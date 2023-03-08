---
layout: post
title: Analyzing Singapore's HDB flats resale price
date:   2023-03-08 12:00:00 +0700
tags: analysis visualization
featured_img: /assets/images/posts/01-predict-hdb-resale_files/color-map.png

---

In this series I will be modeling Singapore's HDB flats resale price. The first part is building a model with standard ML problem process (EDA, Feature Engineering, Split train/test data, Fit model, Evaluation). The second part (coming soon™) will be about mlops. I will use `mlflow` to track and manage experiments & models.

Part 1 (This post)
- Train different models to predict resale price for Singapore's HDB

Part 2 (Coming soon™)
- Run & Log experiments and models using `mlflow`
- Save models to model registry
- Load and serve the best model

<details>
<summary>What are HDB flats?</summary>
    <ul>
    <li> HDB (Housing and Development Board) buildings are public housing blocks in Singapore. They were built and managed by the Housing and Development Board (HDB), a statutory board under the Ministry of National Development. 
    <li>
    HDB flats range from studio apartments to executive apartments, and are available for purchase or rent. Today 80% of Singapore's population live in HDB flats.
    </li>
    </li>
    <li>
    HDB flats are typically located in housing estates, which are self-contained communities with amenities such as schools, markets, and parks. The HDB also manages and maintains the estates, ensuring that they remain safe, clean and well-maintained.
    </li>
    </ul>
</details>


## 0. Import and read data


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
from shapely.geometry import Point
import folium

from math import radians, cos, sin, asin, sqrt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
```


```python
df = pd.read_csv('./data/processed/intermediate-data.csv')
df['town'] = df['town'].replace({'KALLANG/WHAMPOA': 'KALLANG'})
df.shape
```




    (133473, 18)



## 1. Exploratory Data Analysis

Preview dataset


```python
df.head().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>month</th>
      <td>2017-01</td>
      <td>2017-01</td>
      <td>2017-01</td>
      <td>2017-01</td>
      <td>2017-01</td>
    </tr>
    <tr>
      <th>town</th>
      <td>ANG MO KIO</td>
      <td>ANG MO KIO</td>
      <td>ANG MO KIO</td>
      <td>ANG MO KIO</td>
      <td>ANG MO KIO</td>
    </tr>
    <tr>
      <th>flat_type</th>
      <td>2 ROOM</td>
      <td>3 ROOM</td>
      <td>3 ROOM</td>
      <td>3 ROOM</td>
      <td>3 ROOM</td>
    </tr>
    <tr>
      <th>block</th>
      <td>406</td>
      <td>108</td>
      <td>602</td>
      <td>465</td>
      <td>601</td>
    </tr>
    <tr>
      <th>street_name_x</th>
      <td>ANG MO KIO AVE 10</td>
      <td>ANG MO KIO AVE 4</td>
      <td>ANG MO KIO AVE 5</td>
      <td>ANG MO KIO AVE 10</td>
      <td>ANG MO KIO AVE 5</td>
    </tr>
    <tr>
      <th>storey_range</th>
      <td>10 TO 12</td>
      <td>01 TO 03</td>
      <td>01 TO 03</td>
      <td>04 TO 06</td>
      <td>01 TO 03</td>
    </tr>
    <tr>
      <th>floor_area_sqm</th>
      <td>44.0</td>
      <td>67.0</td>
      <td>67.0</td>
      <td>68.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>flat_model</th>
      <td>Improved</td>
      <td>New Generation</td>
      <td>New Generation</td>
      <td>New Generation</td>
      <td>New Generation</td>
    </tr>
    <tr>
      <th>lease_commence_date</th>
      <td>1979</td>
      <td>1978</td>
      <td>1980</td>
      <td>1980</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>remaining_lease</th>
      <td>61 years 04 months</td>
      <td>60 years 07 months</td>
      <td>62 years 05 months</td>
      <td>62 years 01 month</td>
      <td>62 years 05 months</td>
    </tr>
    <tr>
      <th>resale_price</th>
      <td>232000.0</td>
      <td>250000.0</td>
      <td>262000.0</td>
      <td>265000.0</td>
      <td>265000.0</td>
    </tr>
    <tr>
      <th>key</th>
      <td>ANG MO KIO AVENUE 10</td>
      <td>ANG MO KIO AVENUE 4</td>
      <td>ANG MO KIO AVENUE 5</td>
      <td>ANG MO KIO AVENUE 10</td>
      <td>ANG MO KIO AVENUE 5</td>
    </tr>
    <tr>
      <th>postal</th>
      <td>560406.0</td>
      <td>560108.0</td>
      <td>560602.0</td>
      <td>560465.0</td>
      <td>560601.0</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>1.362005</td>
      <td>1.370966</td>
      <td>1.380709</td>
      <td>1.366201</td>
      <td>1.381041</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>103.85388</td>
      <td>103.838202</td>
      <td>103.835368</td>
      <td>103.857201</td>
      <td>103.835132</td>
    </tr>
    <tr>
      <th>street_name_y</th>
      <td>ANG MO KIO AVENUE 10</td>
      <td>ANG MO KIO AVENUE 4</td>
      <td>ANG MO KIO AVENUE 5</td>
      <td>ANG MO KIO AVENUE 10</td>
      <td>ANG MO KIO AVENUE 5</td>
    </tr>
    <tr>
      <th>building</th>
      <td>HDB-ANG MO KIO</td>
      <td>KEBUN BARU HEIGHTS</td>
      <td>YIO CHU KANG GREEN</td>
      <td>TECK GHEE HORIZON</td>
      <td>YIO CHU KANG GREEN</td>
    </tr>
    <tr>
      <th>address</th>
      <td>406 ANG MO KIO AVENUE 10 HDB-ANG MO KIO SINGAP...</td>
      <td>108 ANG MO KIO AVENUE 4 KEBUN BARU HEIGHTS SIN...</td>
      <td>602 ANG MO KIO AVENUE 5 YIO CHU KANG GREEN SIN...</td>
      <td>465 ANG MO KIO AVENUE 10 TECK GHEE HORIZON SIN...</td>
      <td>601 ANG MO KIO AVENUE 5 YIO CHU KANG GREEN SIN...</td>
    </tr>
  </tbody>
</table>
</div>



Check of null/NaN data. In this case our % and count of rows having missing data are low so it's safe to drop them


```python
df.isnull().mean()
```




    month                  0.000000
    town                   0.000000
    flat_type              0.000000
    block                  0.000000
    street_name_x          0.000000
    storey_range           0.000000
    floor_area_sqm         0.000000
    flat_model             0.000000
    lease_commence_date    0.000000
    remaining_lease        0.000000
    resale_price           0.000000
    key                    0.000000
    postal                 0.018513
    latitude               0.018513
    longitude              0.018513
    street_name_y          0.018513
    building               0.018513
    address                0.018513
    dtype: float64




```python
df = df.drop(df[df['latitude'].isnull()].index)
```

Now let's look at the overall distribution of target variable. The distribution is right-skewed with few but very high values. The median value (SGD 440k) is lower than average (SGD 470k)


```python
fig, ax = plt.subplots(figsize=(9, 4))
df['resale_price'].hist(
    ec='white',
    linewidth=.5,
    bins=[_ * 50_000 for _ in range(30)],
    grid=False,
    ax=ax)

ax.axvline(df['resale_price'].mean(), ls='--', c='red', label=f"avg = SGD {df['resale_price'].mean():,.0f}")
ax.axvline(df['resale_price'].median(), ls='--', c='k', label=f"median = SGD {df['resale_price'].median():,.0f}")

ax.set_title(f'HDB Resale Price')
ax.legend()
sns.despine(ax=ax)
```


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_12_0.png)
    


#### Numerical variables

Correlations between each numeric variables and the target value give us a good sense of how much predicting power they have.

Unsurprisingly, bigger and newer flats are positively correlated with resale price


```python
fig, ax = plt.subplots()
df.drop('resale_price', axis=1) \
    .corrwith(df['resale_price'], numeric_only=True) \
    .plot(kind='barh', ax=ax)
ax.axvline(x=0, c='k')
ax.set_title('Correlation with resale price')
sns.despine(ax=ax)
```


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_15_0.png)
    


We can also study how much feature variables correlate with each other. If we solely care about having a good prediction, then it's ok to have correlated features! However, if we want to study the impact of each variable, it's helpful to remove redundant dimensions.

In the heatmap below, `latitude` and `postal` are highly correlated, we will drop at least 1 of them before fitting the models. However for now we will keep them to create a more useful feature.


```python
corr_matrix = df.corr(numeric_only=True).round(3)

# remove the top triangle of the matrix to simplify the heatmap
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap=sns.color_palette("vlag_r", as_cmap=True),
    ax=ax)
```




    <AxesSubplot: >




    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_17_1.png)
    


#### Categorical variables

`block`, `building`, `address` contains too many categories. This might cause overfitting or computational issues then we will not be using those variables.

Perhaps the most useful feature we can use here is `flat_type`. The possible values are:
- 2 ROOM
- 3 ROOM
- 4 ROOM
- 5 ROOM
- Executive
- Multi-generationla

There are a few ways to deal with this data. For simplicity, I will convert them to number of rooms so we can have a nice numeric value.


```python
df.select_dtypes('object').nunique()
```




    month                68
    town                 26
    flat_type             6
    block              2580
    street_name_x       556
    storey_range         17
    flat_model           21
    remaining_lease     653
    key                 556
    street_name_y       556
    building            612
    address            8896
    dtype: int64




```python
sns.boxplot(
    data=df,
    y='resale_price',
    x='flat_type',
    palette="Reds",
    showfliers=False,)
```




    <AxesSubplot: xlabel='flat_type', ylabel='resale_price'>




    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_20_1.png)
    


## 2. Feature engineering

Intuitively, we know that location is one of the most important factor in determining house prices. We will need a way to add this information to our model. To visualize the impact of location, I've created a choropleth map.


```python
sg_map = gpd.read_file('data/geojson/singapore_planning.geojson')
sg_map['PLN_AREA_N'] = sg_map['PLN_AREA_N'].replace({'OUTRAM': 'CENTRAL AREA', 'ROCHOR': 'CENTRAL AREA', 'DOWNTOWN CORE': 'CENTRAL AREA'})
sg_map = sg_map.dissolve('PLN_AREA_N')
sg_map.to_file('data/geojson/singapore_planning_clean.geojson', driver='GeoJSON')

mrt_map = gpd.read_file('data/geojson/singapore-mrt.min.geojson')
mrt_map = mrt_map.drop(['network', 'wikipedia_url', 'wikipedia_image_url', 'name_zh', 'name_hi'], axis=1, errors='ignore')
mrt_map = mrt_map[mrt_map['type']=='station']
mrt_map = mrt_map.reset_index(drop=True)

sg_map['avg 4 ROOM'] = sg_map.index.map(df[df['flat_type']=='4 ROOM'].groupby('town')['resale_price'].mean().to_dict())
sg_map['PLAN_AREA_N'] = sg_map.index

town_map = folium.Map(location=[1.35, 103.8], zoom_start=11, tiles='CartoDB positron')

folium.Choropleth(
    geo_data='data/geojson/singapore_planning_clean.geojson',
    name="4 ROOM",
    data=sg_map,
    columns=['PLAN_AREA_N', 'avg 4 ROOM'],
    key_on="feature.properties.PLN_AREA_N",
    fill_color="RdYlGn_r",
    nan_fill_color="None",
    fill_opacity=0.75,
    bins=[_ * 50_000 for _ in range(6, 17)],
    line_opacity=.1,
    legend_name="Average HDB resale value (SGD)",
    highlight=True,
).add_to(town_map)

style_function = lambda x: {
    'fillColor': '#ffffff',
    'color':'#000000', 
    'fillOpacity': 0.1, 
    'weight': 0.1}

highlight_function = lambda x: {
    'fillColor': '#000000', 
    'color':'#000000', 
    'fillOpacity': 0.50, 
    'weight': 0.1}
    
NIL = folium.features.GeoJson(
    'data/geojson/singapore_planning_clean.geojson',
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['PLN_AREA_N'],  # use fields from the json file
        aliases=['Town:'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)

town_map.add_child(NIL)
town_map.keep_in_front(NIL)


folium.LayerControl().add_to(town_map)
# town_map.save("assets/town_map.html")
town_map
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-1.12.4.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_8fd77da61d7c8fecbc6b4bdc374ae787 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js&quot;&gt;&lt;/script&gt;

                    &lt;style&gt;
                        .foliumtooltip {
                            background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;
                        }
                       .foliumtooltip table{
                            margin: auto;
                        }
                        .foliumtooltip tr{
                            text-align: left;
                        }
                        .foliumtooltip th{
                            padding: 2px; padding-right: 8px;
                        }
                    &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_8fd77da61d7c8fecbc6b4bdc374ae787&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_8fd77da61d7c8fecbc6b4bdc374ae787 = L.map(
                &quot;map_8fd77da61d7c8fecbc6b4bdc374ae787&quot;,
                {
                    center: [1.35, 103.8],
                    crs: L.CRS.EPSG3857,
                    zoom: 11,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );


            function objects_in_front() {
                    geo_json_a46079314baad62b3ecb17352e893b86.bringToFront();
            };
            map_8fd77da61d7c8fecbc6b4bdc374ae787.on(&quot;overlayadd&quot;, objects_in_front);
            $(document).ready(objects_in_front);



            var tile_layer_513f46c14716640c5b5504d32980067f = L.tileLayer(
                &quot;https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors \u0026copy; \u003ca target=\&quot;_blank\&quot; href=\&quot;http://cartodb.com/attributions\&quot;\u003eCartoDB\u003c/a\u003e, CartoDB \u003ca target=\&quot;_blank\&quot; href =\&quot;http://cartodb.com/attributions\&quot;\u003eattributions\u003c/a\u003e&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 18, &quot;maxZoom&quot;: 18, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            ).addTo(map_8fd77da61d7c8fecbc6b4bdc374ae787);


            var choropleth_9b3dbaa289f178330bbeb3e6b1fb8ebe = L.featureGroup(
                {}
            ).addTo(map_8fd77da61d7c8fecbc6b4bdc374ae787);


        function geo_json_77df15357c67e335f7c50573becd81b4_styler(feature) {
            switch(feature.properties.PLN_AREA_N) {
                case &quot;ANG MO KIO&quot;: case &quot;BEDOK&quot;: case &quot;PASIR RIS&quot;: case &quot;PUNGGOL&quot;: case &quot;SERANGOON&quot;: case &quot;TAMPINES&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#a6d96a&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;BISHAN&quot;: case &quot;CLEMENTI&quot;: case &quot;GEYLANG&quot;: case &quot;TOA PAYOH&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#fee08b&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;BUKIT BATOK&quot;: case &quot;BUKIT PANJANG&quot;: case &quot;HOUGANG&quot;: case &quot;JURONG EAST&quot;: case &quot;SENGKANG&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#66bd63&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;BUKIT MERAH&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#f46d43&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;BUKIT TIMAH&quot;: case &quot;KALLANG&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#fdae61&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;CENTRAL AREA&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#a50026&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;CHOA CHU KANG&quot;: case &quot;JURONG WEST&quot;: case &quot;SEMBAWANG&quot;: case &quot;WOODLANDS&quot;: case &quot;YISHUN&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#1a9850&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;MARINE PARADE&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#d9ef8b&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                case &quot;QUEENSTOWN&quot;: 
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;#d73027&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
                default:
                    return {&quot;color&quot;: &quot;black&quot;, &quot;fillColor&quot;: &quot;None&quot;, &quot;fillOpacity&quot;: 0.75, &quot;opacity&quot;: 0.1, &quot;weight&quot;: 1};
            }
        }
        function geo_json_77df15357c67e335f7c50573becd81b4_highlighter(feature) {
            switch(feature.properties.PLN_AREA_N) {
                default:
                    return {&quot;fillOpacity&quot;: 0.95, &quot;weight&quot;: 3};
            }
        }

        function geo_json_77df15357c67e335f7c50573becd81b4_onEachFeature(feature, layer) {
            layer.on({
                mouseout: function(e) {
                    if(typeof e.target.setStyle === &quot;function&quot;){
                        geo_json_77df15357c67e335f7c50573becd81b4.resetStyle(e.target);
                    }
                },
                mouseover: function(e) {
                    if(typeof e.target.setStyle === &quot;function&quot;){
                        const highlightStyle = geo_json_77df15357c67e335f7c50573becd81b4_highlighter(e.target.feature)
                        e.target.setStyle(highlightStyle);
                    }
                },
            });
        };
        var geo_json_77df15357c67e335f7c50573becd81b4 = L.geoJson(null, {
                onEachFeature: geo_json_77df15357c67e335f7c50573becd81b4_onEachFeature,

                style: geo_json_77df15357c67e335f7c50573becd81b4_styler,
        });

        function geo_json_77df15357c67e335f7c50573becd81b4_add (data) {
            geo_json_77df15357c67e335f7c50573becd81b4
                .addData(data)
                .addTo(choropleth_9b3dbaa289f178330bbeb3e6b1fb8ebe);
        }



    var color_map_db60926755705b8355961189342ca772 = {};


    color_map_db60926755705b8355961189342ca772.color = d3.scale.threshold()
              .domain([300000.0, 301002.004008016, 302004.00801603205, 303006.0120240481, 304008.0160320641, 305010.0200400802, 306012.0240480962, 307014.02805611223, 308016.03206412826, 309018.0360721443, 310020.0400801603, 311022.04408817634, 312024.04809619236, 313026.05210420844, 314028.05611222447, 315030.0601202405, 316032.0641282565, 317034.06813627254, 318036.07214428857, 319038.0761523046, 320040.0801603206, 321042.0841683367, 322044.0881763527, 323046.09218436875, 324048.0961923848, 325050.1002004008, 326052.10420841683, 327054.10821643285, 328056.1122244489, 329058.1162324649, 330060.120240481, 331062.124248497, 332064.12825651304, 333066.13226452906, 334068.1362725451, 335070.1402805611, 336072.14428857714, 337074.14829659316, 338076.1523046092, 339078.1563126253, 340080.1603206413, 341082.1643286573, 342084.16833667335, 343086.1723446894, 344088.1763527054, 345090.1803607214, 346092.1843687375, 347094.1883767535, 348096.19238476956, 349098.1963927856, 350100.2004008016, 351102.20440881763, 352104.20841683366, 353106.2124248497, 354108.2164328657, 355110.2204408818, 356112.2244488978, 357114.22845691384, 358116.23246492987, 359118.2364729459, 360120.2404809619, 361122.24448897794, 362124.24849699397, 363126.25250501, 364128.2565130261, 365130.2605210421, 366132.2645290581, 367134.26853707415, 368136.2725450902, 369138.2765531062, 370140.2805611222, 371142.2845691383, 372144.2885771543, 373146.29258517036, 374148.2965931864, 375150.3006012024, 376152.30460921844, 377154.30861723446, 378156.3126252505, 379158.3166332665, 380160.3206412826, 381162.32464929856, 382164.32865731465, 383166.3326653307, 384168.3366733467, 385170.3406813627, 386172.34468937875, 387174.3486973948, 388176.3527054108, 389178.3567134269, 390180.3607214429, 391182.36472945893, 392184.36873747496, 393186.372745491, 394188.376753507, 395190.38076152303, 396192.38476953906, 397194.3887775551, 398196.39278557117, 399198.3967935872, 400200.4008016032, 401202.40480961924, 402204.40881763527, 403206.4128256513, 404208.4168336673, 405210.4208416834, 406212.42484969937, 407214.42885771545, 408216.4328657315, 409218.4368737475, 410220.4408817635, 411222.44488977955, 412224.4488977956, 413226.4529058116, 414228.4569138277, 415230.46092184365, 416232.46492985974, 417234.46893787576, 418236.4729458918, 419238.4769539078, 420240.48096192384, 421242.48496993986, 422244.4889779559, 423246.49298597197, 424248.496993988, 425250.501002004, 426252.50501002005, 427254.5090180361, 428256.5130260521, 429258.5170340681, 430260.52104208415, 431262.5250501002, 432264.52905811626, 433266.5330661322, 434268.5370741483, 435270.54108216433, 436272.54509018036, 437274.5490981964, 438276.5531062124, 439278.5571142285, 440280.56112224446, 441282.56513026054, 442284.56913827657, 443286.5731462926, 444288.5771543086, 445290.58116232464, 446292.5851703407, 447294.5891783567, 448296.5931863728, 449298.59719438874, 450300.6012024048, 451302.60521042085, 452304.6092184369, 453306.6132264529, 454308.6172344689, 455310.621242485, 456312.625250501, 457314.62925851706, 458316.633266533, 459318.6372745491, 460320.64128256514, 461322.64529058116, 462324.6492985972, 463326.6533066132, 464328.6573146293, 465330.66132264526, 466332.66533066134, 467334.6693386773, 468336.6733466934, 469338.6773547094, 470340.68136272545, 471342.68537074147, 472344.6893787575, 473346.6933867736, 474348.69739478955, 475350.70140280563, 476352.70541082165, 477354.7094188377, 478356.7134268537, 479358.71743486973, 480360.7214428858, 481362.7254509018, 482364.72945891786, 483366.73346693383, 484368.7374749499, 485370.74148296594, 486372.74549098196, 487374.749498998, 488376.753507014, 489378.7575150301, 490380.76152304607, 491382.76553106215, 492384.7695390781, 493386.7735470942, 494388.7775551102, 495390.78156312625, 496392.7855711423, 497394.7895791583, 498396.7935871744, 499398.79759519035, 500400.80160320643, 501402.80561122246, 502404.8096192385, 503406.8136272545, 504408.81763527053, 505410.82164328656, 506412.8256513026, 507414.82965931867, 508416.83366733463, 509418.8376753507, 510420.84168336674, 511422.84569138277, 512424.8496993988, 513426.8537074148, 514428.8577154309, 515430.86172344687, 516432.86573146295, 517434.8697394789, 518436.873747495, 519438.877755511, 520440.88176352705, 521442.8857715431, 522444.8897795591, 523446.8937875752, 524448.8977955912, 525450.9018036072, 526452.9058116232, 527454.9098196393, 528456.9138276554, 529458.9178356713, 530460.9218436873, 531462.9258517034, 532464.9298597195, 533466.9338677354, 534468.9378757515, 535470.9418837675, 536472.9458917836, 537474.9498997997, 538476.9539078156, 539478.9579158317, 540480.9619238477, 541482.9659318638, 542484.9699398797, 543486.9739478958, 544488.9779559118, 545490.9819639279, 546492.9859719439, 547494.9899799599, 548496.993987976, 549498.997995992, 550501.002004008, 551503.006012024, 552505.0100200401, 553507.0140280561, 554509.0180360721, 555511.0220440882, 556513.0260521042, 557515.0300601203, 558517.0340681362, 559519.0380761523, 560521.0420841683, 561523.0460921844, 562525.0501002003, 563527.0541082164, 564529.0581162325, 565531.0621242485, 566533.0661322644, 567535.0701402805, 568537.0741482966, 569539.0781563127, 570541.0821643287, 571543.0861723446, 572545.0901803607, 573547.0941883768, 574549.0981963928, 575551.1022044088, 576553.1062124248, 577555.1102204409, 578557.114228457, 579559.118236473, 580561.1222444889, 581563.126252505, 582565.1302605211, 583567.134268537, 584569.1382765531, 585571.1422845691, 586573.1462925852, 587575.1503006013, 588577.1543086172, 589579.1583166332, 590581.1623246493, 591583.1663326654, 592585.1703406814, 593587.1743486974, 594589.1783567134, 595591.1823647295, 596593.1863727455, 597595.1903807615, 598597.1943887775, 599599.1983967936, 600601.2024048097, 601603.2064128257, 602605.2104208417, 603607.2144288577, 604609.2184368738, 605611.2224448898, 606613.2264529058, 607615.2304609218, 608617.2344689379, 609619.2384769539, 610621.24248497, 611623.246492986, 612625.250501002, 613627.254509018, 614629.2585170341, 615631.2625250501, 616633.266533066, 617635.2705410821, 618637.2745490982, 619639.2785571143, 620641.2825651303, 621643.2865731462, 622645.2905811623, 623647.2945891784, 624649.2985971944, 625651.3026052103, 626653.3066132264, 627655.3106212425, 628657.3146292586, 629659.3186372746, 630661.3226452905, 631663.3266533066, 632665.3306613227, 633667.3346693387, 634669.3386773546, 635671.3426853707, 636673.3466933868, 637675.3507014029, 638677.3547094188, 639679.3587174348, 640681.3627254509, 641683.366733467, 642685.3707414829, 643687.374749499, 644689.378757515, 645691.3827655311, 646693.3867735472, 647695.3907815631, 648697.3947895791, 649699.3987975952, 650701.4028056113, 651703.4068136273, 652705.4108216433, 653707.4148296593, 654709.4188376754, 655711.4228456914, 656713.4268537074, 657715.4308617234, 658717.4348697395, 659719.4388777555, 660721.4428857716, 661723.4468937876, 662725.4509018036, 663727.4549098196, 664729.4589178357, 665731.4629258517, 666733.4669338677, 667735.4709418837, 668737.4749498998, 669739.4789579159, 670741.4829659319, 671743.4869739478, 672745.4909819639, 673747.49498998, 674749.498997996, 675751.503006012, 676753.507014028, 677755.5110220441, 678757.5150300602, 679759.5190380762, 680761.5230460921, 681763.5270541082, 682765.5310621243, 683767.5350701403, 684769.5390781562, 685771.5430861723, 686773.5470941884, 687775.5511022045, 688777.5551102204, 689779.5591182364, 690781.5631262525, 691783.5671342686, 692785.5711422845, 693787.5751503005, 694789.5791583166, 695791.5831663327, 696793.5871743488, 697795.5911823647, 698797.5951903807, 699799.5991983968, 700801.6032064129, 701803.6072144288, 702805.6112224449, 703807.6152304609, 704809.619238477, 705811.623246493, 706813.627254509, 707815.631262525, 708817.6352705411, 709819.6392785572, 710821.6432865731, 711823.6472945892, 712825.6513026052, 713827.6553106213, 714829.6593186373, 715831.6633266533, 716833.6673346693, 717835.6713426854, 718837.6753507014, 719839.6793587175, 720841.6833667335, 721843.6873747495, 722845.6913827655, 723847.6953907816, 724849.6993987976, 725851.7034068136, 726853.7074148296, 727855.7114228457, 728857.7154308618, 729859.7194388778, 730861.7234468937, 731863.7274549098, 732865.7314629259, 733867.7354709419, 734869.7394789578, 735871.7434869739, 736873.74749499, 737875.7515030061, 738877.755511022, 739879.759519038, 740881.7635270541, 741883.7675350702, 742885.7715430862, 743887.7755511021, 744889.7795591182, 745891.7835671343, 746893.7875751504, 747895.7915831663, 748897.7955911823, 749899.7995991984, 750901.8036072145, 751903.8076152304, 752905.8116232464, 753907.8156312625, 754909.8196392786, 755911.8236472947, 756913.8276553106, 757915.8316633266, 758917.8356713427, 759919.8396793588, 760921.8436873747, 761923.8476953908, 762925.8517034068, 763927.8557114229, 764929.859719439, 765931.8637274549, 766933.8677354709, 767935.871743487, 768937.875751503, 769939.879759519, 770941.8837675351, 771943.8877755511, 772945.8917835671, 773947.8957915832, 774949.8997995992, 775951.9038076152, 776953.9078156312, 777955.9118236473, 778957.9158316634, 779959.9198396794, 780961.9238476953, 781963.9278557114, 782965.9318637275, 783967.9358717435, 784969.9398797594, 785971.9438877755, 786973.9478957916, 787975.9519038077, 788977.9559118237, 789979.9599198396, 790981.9639278557, 791983.9679358718, 792985.9719438878, 793987.9759519037, 794989.9799599198, 795991.9839679359, 796993.987975952, 797995.991983968, 798997.9959919839, 800000.0])
              .range([&#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#006837ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#1a9850ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#66bd63ff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#a6d96aff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#d9ef8bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fee08bff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#fdae61ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#f46d43ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#d73027ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;, &#x27;#a50026ff&#x27;]);


    color_map_db60926755705b8355961189342ca772.x = d3.scale.linear()
              .domain([300000.0, 800000.0])
              .range([0, 450 - 50]);

    color_map_db60926755705b8355961189342ca772.legend = L.control({position: &#x27;topright&#x27;});
    color_map_db60926755705b8355961189342ca772.legend.onAdd = function (map) {var div = L.DomUtil.create(&#x27;div&#x27;, &#x27;legend&#x27;); return div};
    color_map_db60926755705b8355961189342ca772.legend.addTo(map_8fd77da61d7c8fecbc6b4bdc374ae787);

    color_map_db60926755705b8355961189342ca772.xAxis = d3.svg.axis()
        .scale(color_map_db60926755705b8355961189342ca772.x)
        .orient(&quot;top&quot;)
        .tickSize(1)
        .tickValues([300000, &#x27;&#x27;, 400000, &#x27;&#x27;, 500000, &#x27;&#x27;, 600000, &#x27;&#x27;, 700000, &#x27;&#x27;, 800000, &#x27;&#x27;]);

    color_map_db60926755705b8355961189342ca772.svg = d3.select(&quot;.legend.leaflet-control&quot;).append(&quot;svg&quot;)
        .attr(&quot;id&quot;, &#x27;legend&#x27;)
        .attr(&quot;width&quot;, 450)
        .attr(&quot;height&quot;, 40);

    color_map_db60926755705b8355961189342ca772.g = color_map_db60926755705b8355961189342ca772.svg.append(&quot;g&quot;)
        .attr(&quot;class&quot;, &quot;key&quot;)
        .attr(&quot;transform&quot;, &quot;translate(25,16)&quot;);

    color_map_db60926755705b8355961189342ca772.g.selectAll(&quot;rect&quot;)
        .data(color_map_db60926755705b8355961189342ca772.color.range().map(function(d, i) {
          return {
            x0: i ? color_map_db60926755705b8355961189342ca772.x(color_map_db60926755705b8355961189342ca772.color.domain()[i - 1]) : color_map_db60926755705b8355961189342ca772.x.range()[0],
            x1: i &lt; color_map_db60926755705b8355961189342ca772.color.domain().length ? color_map_db60926755705b8355961189342ca772.x(color_map_db60926755705b8355961189342ca772.color.domain()[i]) : color_map_db60926755705b8355961189342ca772.x.range()[1],
            z: d
          };
        }))
      .enter().append(&quot;rect&quot;)
        .attr(&quot;height&quot;, 40 - 30)
        .attr(&quot;x&quot;, function(d) { return d.x0; })
        .attr(&quot;width&quot;, function(d) { return d.x1 - d.x0; })
        .style(&quot;fill&quot;, function(d) { return d.z; });

    color_map_db60926755705b8355961189342ca772.g.call(color_map_db60926755705b8355961189342ca772.xAxis).append(&quot;text&quot;)
        .attr(&quot;class&quot;, &quot;caption&quot;)
        .attr(&quot;y&quot;, 21)
        .text(&quot;Average HDB resale value (SGD)&quot;);

        function geo_json_a46079314baad62b3ecb17352e893b86_styler(feature) {
            switch(feature.properties.PLN_AREA_N) {
                default:
                    return {&quot;color&quot;: &quot;#000000&quot;, &quot;fillColor&quot;: &quot;#ffffff&quot;, &quot;fillOpacity&quot;: 0.1, &quot;weight&quot;: 0.1};
            }
        }
        function geo_json_a46079314baad62b3ecb17352e893b86_highlighter(feature) {
            switch(feature.properties.PLN_AREA_N) {
                default:
                    return {&quot;color&quot;: &quot;#000000&quot;, &quot;fillColor&quot;: &quot;#000000&quot;, &quot;fillOpacity&quot;: 0.5, &quot;weight&quot;: 0.1};
            }
        }

        function geo_json_a46079314baad62b3ecb17352e893b86_onEachFeature(feature, layer) {
            layer.on({
                mouseout: function(e) {
                    if(typeof e.target.setStyle === &quot;function&quot;){
                        geo_json_a46079314baad62b3ecb17352e893b86.resetStyle(e.target);
                    }
                },
                mouseover: function(e) {
                    if(typeof e.target.setStyle === &quot;function&quot;){
                        const highlightStyle = geo_json_a46079314baad62b3ecb17352e893b86_highlighter(e.target.feature)
                        e.target.setStyle(highlightStyle);
                    }
                },
            });
        };
        var geo_json_a46079314baad62b3ecb17352e893b86 = L.geoJson(null, {
                onEachFeature: geo_json_a46079314baad62b3ecb17352e893b86_onEachFeature,

                style: geo_json_a46079314baad62b3ecb17352e893b86_styler,
        });

        function geo_json_a46079314baad62b3ecb17352e893b86_add (data) {
            geo_json_a46079314baad62b3ecb17352e893b86
                .addData(data)
                .addTo(map_8fd77da61d7c8fecbc6b4bdc374ae787);
        }



    geo_json_a46079314baad62b3ecb17352e893b86.bindTooltip(
    function(layer){
    let div = L.DomUtil.create(&#x27;div&#x27;);

    let handleObject = feature=&gt;typeof(feature)==&#x27;object&#x27; ? JSON.stringify(feature) : feature;
    let fields = [&quot;PLN_AREA_N&quot;];
    let aliases = [&quot;Town:&quot;];
    let table = &#x27;&lt;table&gt;&#x27; +
        String(
        fields.map(
        (v,i)=&gt;
        `&lt;tr&gt;
            &lt;th&gt;${aliases[i]}&lt;/th&gt;

            &lt;td&gt;${handleObject(layer.feature.properties[v])}&lt;/td&gt;
        &lt;/tr&gt;`).join(&#x27;&#x27;))
    +&#x27;&lt;/table&gt;&#x27;;
    div.innerHTML=table;

    return div
    }
    ,{&quot;className&quot;: &quot;foliumtooltip&quot;, &quot;sticky&quot;: true});


            var layer_control_42a3e612f4f2b53ac2ee41693186bf0f = {
                base_layers : {
                    &quot;cartodbpositron&quot; : tile_layer_513f46c14716640c5b5504d32980067f,
                },
                overlays :  {
                    &quot;4 ROOM&quot; : choropleth_9b3dbaa289f178330bbeb3e6b1fb8ebe,
                },
            };
            L.control.layers(
                layer_control_42a3e612f4f2b53ac2ee41693186bf0f.base_layers,
                layer_control_42a3e612f4f2b53ac2ee41693186bf0f.overlays,
                {&quot;autoZIndex&quot;: true, &quot;collapsed&quot;: true, &quot;position&quot;: &quot;topright&quot;}
            ).addTo(map_8fd77da61d7c8fecbc6b4bdc374ae787);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



To determine how central a location it, I will calculate the straight line distance to a center point. For Singapore, I've selected the point having coordinate value `CITY_CENTER = (1.2801990449115896, 103.85175675603243)`


```python
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    
    # constant used to convert to km
    r = 6371
    
    return c * r


CITY_CENTER = (1.2801990449115896, 103.85175675603243)

df['distance_from_center'] = df.apply(
    lambda x: haversine(
        x['longitude'],
        x['latitude'],
        CITY_CENTER[1],
        CITY_CENTER[0]), axis=1)
```

### Distacne to nearest MRT station

Other than distance to city center, close priximity to amenities and public transportations could also make the flats more attractive and hence having higher price. I've also added distance (in km) to the nearst MRT station.

To make calculations faster, I've (1) estimated the nearest location using Pythagorean theorem, and then (2) use the haversine formula above to determine actual distance in kilometers.


```python
# hdb_ll -> np.ndarray

# latitude is the x-axis
# longitude is the y-axis
hdb_ll = df[['longitude', 'latitude']].values

# find nearst mrt
x_delta = (mrt_map.geometry.x.values.reshape(-1, 1) - hdb_ll[:, 0])
y_delta = (mrt_map.geometry.y.values.reshape(-1, 1) - hdb_ll[:, 1])
delta = ((x_delta ** 2) + (y_delta)**2) ** .5

df['nearest_mrt_id'] = delta.argmin(axis=0)
df['nearest_mrt_location'] = df['nearest_mrt_id'].map(pd.Series(mrt_map.geometry.values, index=mrt_map.index).to_dict())
df['nearest_mrt_longitude'] = df['nearest_mrt_location'].apply(lambda val: val.x)
df['nearest_mrt_latitude'] = df['nearest_mrt_location'].apply(lambda val: val.y)

df['distance_to_mrt'] = df.apply(lambda x:
    haversine(
        x['longitude'],
        x['latitude'],
        x['nearest_mrt_longitude'],
        x['nearest_mrt_latitude']),
        axis=1)

df = df.drop(['nearest_mrt_longitude', 'nearest_mrt_latitude', 'nearest_mrt_id'], axis=1, errors='ignore')
```

### Other features

Here's the remaining features to process:
* Convert remaining lease duration from year-month format to months
* Create a new feature called number of bedrooms from flat type (explained above)
* Averaged storey number since the original dataset list a range instead of exact floor number
* Drop values for rare types
* Drop highly correlated features
* Use boxlots to determine appropriate feature range. Optionally drop outliers.


```python
# convert `remaining_lease_months` from years months to months
df['remaining_lease'].str.contains('year').value_counts()
df['remaining_lease_months'] = df['remaining_lease'].str[:2].astype(int) * 12 + df['remaining_lease'].str[-9:-7].astype(int)


# convert flat type to number of rooms
# https://www.hdb.gov.sg/residential/buying-a-flat/finding-a-flat/types-of-flats
df['num_bedrooms'] = df['flat_type'].map({
    '1 ROOM': 0.5,
    '2 ROOM': 1,
    '3 ROOM': 2,
    '4 ROOM': 3,
    '5 ROOM': 4,
    'EXECUTIVE': 5,
    'MULTI-GENERATION': 6,
})

df = df.drop(df[df['num_bedrooms'].isnull()].index)

# convert storey range to mean of range
df['storey_range_feature'] = (df['storey_range'].str[:2].astype(int) + df['storey_range'].str[-2:].astype(int)) / 2

# drop highly correlated features
df = df.drop(['lease_commence_date', 'postal'], axis=1, errors='ignore')


numerical_feature_cols = [
    'num_bedrooms',
    'floor_area_sqm',
    'storey_range_feature',
    'remaining_lease_months',
    'distance_to_mrt',
    'distance_from_center',
    'floor_area_sqm',
    ]

fig, axes = plt.subplots(1, len(numerical_feature_cols), figsize=(16, 6))

for idx, col_name in enumerate(numerical_feature_cols):
    ax = axes[idx]
    sns.boxplot(df[col_name], ax=ax)
    sns.despine(ax=ax)
    ax.set_title(col_name.replace('_', '\n'), loc='left')

fig.tight_layout()
```


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_30_0.png)
    



```python
# remove outliers
df = df[df['floor_area_sqm'] <= df['floor_area_sqm'].quantile(.95)]
df = df.dropna()
df = df.reset_index(drop=True)
```


```python
# correlation between numeric variables and target
fig, ax = plt.subplots()
df.drop('resale_price', axis=1) \
    .corrwith(df['resale_price'], numeric_only=True) \
    .sort_values(ascending=True) \
    .plot(kind='barh', ax=ax)
ax.axvline(x=0, c='k')
ax.set_title('Correlation with resale price')
sns.despine(ax=ax)
```


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_32_0.png)
    



```python
# save data
df.to_csv('./data/final/cleaned-data.csv', index=False)
```

## 3. Split train-test data


```python
X = df[numerical_feature_cols]
y = df['resale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

## 4. Fit models

#### 4.1 Linear Regresssion (baseline)

I always want to have a simple base estimator to use for comparing performance


```python
model = LinearRegression()
print(f'fitting {model.__class__.__name__}')
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting LinearRegression
    rmse=78,280


#### 4.2 Decision Trees, Random Forest, GradientBoosting, SVR


```python
model = RandomForestRegressor(max_depth=20, n_estimators=100)
print(f'fitting {model.__class__.__name__}')
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting RandomForestRegressor
    rmse=33,511



```python
model = GradientBoostingRegressor()
print(f'fitting {model.__class__.__name__}')
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting GradientBoostingRegressor
    rmse=60,247



```python
model = LinearSVR()
print(f'fitting {model.__class__.__name__}')
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting LinearSVR
    rmse=81,673


Out of the few models above, RandomForests perform the best. With a RMSE of 33.4k. In real terms, for a property that cost on average close to 500k, being off by 33k is decent.

However, there are a lot more that we can do here:
    1. Selecting different models
    2. Changing hyperparameters


Without mlflow, we will have to manually keep track of model parameteres, data sources, metrics, etc (such as in a google sheet). This is prone to errors and hard to keep track.

In the next post I will use mlflow to manage experiments and models.

## 5. Summary (so far)

- I've built some simple models to predict HDB resale prices. In addition to the given features (eg. living area, town name, flat type), I've added 2 features to measure centrality and proximity to MRT stations.

- RandomForest performs the best. The number of relatively uncorrelated trees operating as a committee will outperform most single complex model. We can improve the prediction by adding more relevant features such as to good schools, shopping malls, other amenities, etc.

- Conceptually, there could be other ways to predict house price, such as using a Time Series model (such as ARIMA)

_That's the end of part 1. In the next post of this series I will write about mlflow_
---
layout: post
title: Analyzing Singapore's HDB flats resale price
date:   2023-03-08 12:00:00 +0700
tags: analysis visualization
featured_img: /assets/images/posts/01-predict-hdb-resale_files/town_map.html
description: Explore & model Singapore's HDB flats resale price using different prediction models. Engineering new features such as centrality and proximity to MRT stations. Out of the box, RandomForest performs better than single complex models with minimal hyperparameters tuning

---
In this series I will be modeling Singapore's HDB flats resale price. The first part is about building a model, following a standard ML problem process (EDA, Feature Engineering, Split train/test data, Fit model, Evaluation). The second part (coming soon™) will be about mlops. I will use `mlflow` to track and manage experiments & models.

Part 1 (This post)
- Train different models to predict resale price for Singapore's HDB

Part 2 (Coming soon™)
- Run & Log experiments and models using `mlflow`
- Save models to model registry
- Load and serve the best model

<details><summary>What are HDB flats?</summary>
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

plt.style.use('seaborn-v0_8-white')
```


```python
df = pd.read_csv('./data/intermediate/intermediate-data.csv')
df = df.drop('Unnamed: 0', axis=1)
df['town'] = df['town'].replace({'KALLANG/WHAMPOA': 'KALLANG'})

df.shape
```




    (133473, 16)



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



Checking for null/NaN data. In this case the percentage of rows having missing data are low, so it's safe to drop them


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
    postal                 0.018513
    latitude               0.018513
    longitude              0.018513
    building               0.018513
    address                0.018513
    dtype: float64




```python
df = df.drop(df[df['latitude'].isnull()].index)
```

Now let's look at the overall distribution of resale price, henceforth also refered to as the target variable.

The distribution is right-skewed with few but very high values. The median value (SGD 440k) is lower than average (SGD 470k)


```python
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, height_ratios=[1, 6])

df['resale_price'].hist(
    ec='white',
    linewidth=.5,
    bins=[_ * 50_000 for _ in range(30)],
    grid=False,
    ax=axes[1])

axes[1].axvline(df['resale_price'].mean(), ls='--', c='red', label=f"avg = SGD {df['resale_price'].mean():,.0f}")
axes[1].axvline(df['resale_price'].median(), ls='--', c='k', label=f"median = SGD {df['resale_price'].median():,.0f}")
axes[1].set_xticks([_ * 100_000 for _ in range(14)], labels=[_ * 100 for _ in range(14)])
axes[1].set_xlabel('Unit: thousands SGD')
axes[1].legend(fontsize=14)
axes[0].axis('off')
sns.despine(ax=axes[1])

sns.boxplot(df['resale_price'], orient="h", showfliers=False, ax=axes[0])
sns.despine(ax=axes[0], left=True, bottom=True)
fig.suptitle('HDB Resale Price Distribution')
fig.tight_layout()
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




    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_17_1.png)
    


#### Categorical variables

`block`, `building`, `address` contains too many categories. This might cause overfitting or computational issues then we will not be using those variables.

Perhaps the most useful feature we can use here is `flat_type`. The possible values are: `2 ROOM` , `3 ROOM`, `4 ROOM`, `5 ROOM`, `Executive`, `Multi-generationl`. There are a few ways to deal with this data. For simplicity, I will convert them to number of rooms so we can have a nice numeric value.


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
    building            612
    address            8896
    dtype: int64




```python
fig, ax = plt.subplots(figsize=(9, 4))

sns.boxplot(
    data=df,
    x='resale_price',
    y='flat_model',
    palette="Reds",
    showfliers=False,
    ax=ax).set_title('`flat_model` has too many categories')
```


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_20_1.png)
    



```python
fig, ax = plt.subplots(figsize=(9, 4))

sns.boxplot(
    data=df,
    x='resale_price',
    y='flat_type',
    palette="Reds",
    showfliers=False,
    ax=ax).set_title('`flat_type` has a few categories and has little overlap. Making this a good feature')
```


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_21_1.png)
    


## 2. Feature engineering

Intuitively, we know that location is one of the most important factor in determining house prices. We will need a way to add this information to our model. To visualize the impact of location, I've created a choropleth map.

It's visible from the map that the further towns have lower price range. 

<details>
<summary>Show code</summary>

<script src="https://gist.github.com/dobeok/6a7a2251a949b3404baf72e546e4ee55.js"></script>
</details>


<div style="width:100%;">
    <div style="position:relative;width:100%;height:0;padding-bottom:60%;">
        <span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span>
            <iframe src="/assets/images/posts/01-predict-hdb-resale_files/town_map.html" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen>
            </iframe>
    </div>
</div><br>

To determine how central a location it, I will calculate the straight line distance to a center point. For Singapore, I've selected the point having coordinate value `CITY_CENTER = (1.28019, 103.85175)`. This point was picked by eyeballing on google map.


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


df['distance_from_center'] = df.apply(
    lambda x: haversine(
        x['longitude'],
        x['latitude'],
        CITY_CENTER[1],
        CITY_CENTER[0]), axis=1)
```

### Distance to the nearest MRT station

Other than distance to city center, close priximity to amenities and public transportations could also make the flats more attractive and hence having higher price. I will now calculate distance from the nearest MRT station to each of the HDB block.

To make calculations faster, I've (1) estimated the nearest location using Pythagorean theorem, and then (2) use the haversine formula above to determine actual distance in kilometers. This will save some calculation time.


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


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_32_0.png)
    



```python
# remove outliers
df = df[df['floor_area_sqm'] <= df['floor_area_sqm'].quantile(.95)]
df = df.dropna()
df = df.reset_index(drop=True)


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


    
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_34_0.png)
    


## 3. Split train-test data


```python
X = df[numerical_feature_cols]
y = df['resale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

## 4. Fit models

#### 4.1 Linear Regresssion (baseline)

I will first fit a simple estimator, which will be used as baseline.


```python
lr_model = LinearRegression()
print(f'fitting {lr_model.__class__.__name__}')
lr_model.fit(X_train, y_train)
preds = lr_model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting LinearRegression
    rmse=78,280


#### 4.2 Decision Trees, Random Forest, GradientBoosting, SVR


```python
rf_model = RandomForestRegressor(max_depth=20, n_estimators=100)
print(f'fitting {rf_model.__class__.__name__}')
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting RandomForestRegressor
    rmse=33,531



```python
gb_model = GradientBoostingRegressor()
print(f'fitting {gb_model.__class__.__name__}')
gb_model.fit(X_train, y_train)
preds = gb_model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting GradientBoostingRegressor
    rmse=60,247



```python
svr_model = LinearSVR()
print(f'fitting {svr_model.__class__.__name__}')
svr_model.fit(X_train, y_train)
preds = svr_model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f'{rmse=:,.0f}')
```

    fitting LinearSVR
    rmse=81,687


#### Making predictions on sample data

To get more specific sense of each model's performace, we can select random test data and check for each model's predictions


```python
# select random indices
idx = np.random.choice(X_test.index, size=3)

sample_predictions = {}

for model in [lr_model, rf_model, gb_model, svr_model]:
    sample_predictions[model.__class__.__name__] = model.predict(X_test.loc[idx])

sample_predictions['Ground Truth'] = y_test[idx]
sample_predictions = pd.DataFrame(sample_predictions)

pd.DataFrame(sample_predictions).plot(
    kind='bar',
    rot=0,
    xlabel='flat id',
    ylabel='SGD') \
    .legend(
        loc='center left',
        bbox_to_anchor=(1.0, 0.5))
```
![png](/assets/images/posts/01-predict-hdb-resale_files/01-predict-hdb-resale_47_1.png)
    


Out of the few models above, RandomForests perform the best. With a RMSE of 33.4k. In real terms, for a property that cost on average close to 500k, being off by 33k is decent.

However, there are a lot more that we can do here:
    1. Selecting different models
    2. Optimizing hyperparameters, for example, using GridSearch or Hyperopt.

Without mlflow, we will have to manually keep track of model parameteres, data sources, metrics, etc (such as in a google sheet). This is prone to errors and hard to keep track.

## 5. Summary (so far)


I created basic HDB resale price prediction models with two additional features for centrality and proximity to MRT stations, on top of existing features such as living area, town name, and flat type.

RandomForest outperforms single complex models due to its use of many relatively uncorrelated trees as a committee. Further improvement to predictions can be achieved by adding more relevant features, such as proximity to good schools, shopping malls, and other amenities.

House prices could also be predicted using a Time Series model like ARIMA, among other potential approaches.

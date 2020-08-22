---
layout: post
title: 'Cohort and Retention analysis'
date: 2020-08-22 21:00:00 +0700
---

Cohort analysis provides insights into users behaviors by segmenting them into mutually exclusive groups and observe the differences. Though there are multiple ways to define a cohort, the most common is grouping users by acquisition date.

Most serious analytics software have built-in cohort analysis tools. But knowing how to create one using python can be handy too.

Input data is a generic sales log including 3 columns: unique customer id, order date and orders quantity

```python
df.head()
```

<img src="/assets/cohort-1.png" alt="cohort-1" width="300"/>



The most suitable time range depends on your product: the choice could be day/week/month. I'm using week because days tends to be too short and months could be too long.

```python
df['order_week'] = df['order_date'].dt.isocalendar().week
df['cohort'] = df.groupby('customer_id')['order_week'].transform('min')
df['weeks_since_first_order'] = df['order_week'] - df['cohort']
```

<img src="/assets/cohort-2.png" alt="cohort-2" width="600"/>



```python
cohort_data = df.groupby(['cohort', 'weeks_since_first_order'])['customer_id'].apply(pd.Series.nunique).reset_index()
cohort_data.head()
```

<img src="/assets/cohort-3.png" alt="cohort-3" width="300"/>

 

```python
cohort_count = cohort_data.pivot_table(index='cohort',
                                       columns='weeks_since_first_order',
                                       values='customer_id')
cohort_count.head()
```

<img src="/assets/cohort-4.png" alt="cohort-4" width="600"/>

```python
retention = cohort_count.divide(cohort_count.iloc[:,0], axis=0).round(3) * 100
```

<img src="/assets/cohort-5.png" alt="cohort-5" width="600"/>

Plotting cohort using seaborn

```python
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title('Customer retention based on first order week')
sns.heatmap(data=retention,
            annot=True,
            fmt = '.0%',
            vmin=0,
            vmax=.8,
            cmap='YlGnBu', ax=ax)

fig.savefig('retention-heatmap.png', transparent=True)
```

<img src="/assets/retention-heatmap.png" alt="heatmap" width="900"/>


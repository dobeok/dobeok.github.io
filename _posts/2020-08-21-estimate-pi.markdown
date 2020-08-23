---
layout: post
title:  "Estimate pi using Monte-Carlo simulation"
date:   2020-08-21 12:11:27 +0700
---

### Idea

In this post I estimate pi using Monte-Carlo simulation. This is done by generating random points in a 1x1 square. The probability of the point falling insize the circle is propotional to the area of the circle.

We can estimate the probability above by counting the actual number of points. To check if the point is inside the circle, we use the Euclidean distance between the point and origin (0, 0)

### The code

```python
import matplotlib.pyplot as plt

import random

# for reproducibility
random.seed(42)
```



```python
fig, ax = plt.subplots()
ax.set_aspect('equal')
circle1 = plt.Circle((0, 0), 1, color='#f0f0f0')
ax.add_artist(circle1)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_title('$1/4$ Area of circle / Area of square = $(\pi * r^2) / 4r^2 = \pi$')
```

![estimate-pi](/assets/estimate-pi.png)

```python
# euclidean distance
def distance_from_origin(point):
    return (point[0]**2 + point[1]**2) ** .5


num_points = 500_000
points = []

for i in range(num_points):
    x, y = random.random(), random.random()
    points.append((x,y))

# check if point is inside circle:
distances = [distance_from_origin(pt) for pt in points]

num_points_inside = len(list(distance for distance in distances if distance<=1))

assert len(list(distance for distance in distances if distance>1)) == num_points - num_points_inside

# estimate pi
print(4 * num_points_inside / num_points)
# 3.125


# how number of points affect pi?
# intuition: larger n gives more accurate pi

# n = 1_000 -> pi ~ 3.156
# n = 10_000 -> pi ~ 3.126
# n = 100_000 -> pi ~ 3.13988
# n = 500_000 -> pi ~ 3.142856
```

---
layout: post
title:  Estimate $\pi$ (pi) by simulation 
date:   2020-08-21 12:11:27 +0700
tags: shorts maths
featured_img: /assets/images/posts/estimate-pi/est-pi-animation.gif
---

### Introduction

I found this interesting maths puzzle: **Given a random uniform function returning values between 0 and 1,  estimate the value of Pi ($\pi$).**

### Idea

Recall that the area $A$ of a cicle with radius $r$ is $A = \pi * r^2$. Hence, if we know both $A$ and $r$, then we can easily determine $\pi$

Since $\pi$ is a constant, we can simplify the problem by letting $r=1$, and try to estimate $A$


To estimate $A$, we do the following:
* Draw a circle with radius $r=1$
* Draw a square with side of length 2
* Align the 2 shapes such that their center overlap

$$\frac{Area\ of\ cirle}{Area\ of\ square} = \frac{\pi * 1^2}{2^2} = \frac{\pi}{4}$$

We then generate random points. The probability that any point falls inside the circle is proportional to the area of the circle. We can estimate that probability  by counting the number of points. To check if the point is inside the circle or not, we calculate the distance between the point and the origin (0, 0)


Thus, the estimated value of $\pi$ is
$$\pi \approx \frac{4 * Number\ of\ points\ in\ circle}{Total\ number\ of\ points}$$


Since the area is estimated using count of points, our intuition is that using more points will yield more accurate approximations.

![estimate-pi](/assets/images/posts/estimate-pi/estimate-pi.png)

### The code

```python
import matplotlib.pyplot as plt

import random

# for reproducibility
random.seed(42)

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

### End
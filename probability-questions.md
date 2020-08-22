---
layout: page
title: Probability Questions
permalink: /probability-questions/
---


## Probability and logic questions

#### 0. How do you simulate a fair coin toss with a biased coin (probability of getting head or tail is unknown and != .5)

_Answer:_

![image](/assets/biased-coin.png)

Instead of tossing the coin once, we toss it twice. The idea is to create different outcomes that have the same probabilities.

There are 4 possibilities: HH, TT, HT, TH

If the outcomes are HH or TT, we ignore.

Since the tosses are independent, the probabilities of getting HT or TH are the same. Hence we can have a fair toss.



#### 1. You and your friends play russian roulette. There are 3 bullets in consecutive chambers. Your friend go first, he pulls the trigger and nothing happens. It's your turn. You are given the choice to spin the cylinder before pulling the trigger. What do you do?

_Answer:_

The probability of surviving if you spin the cylinder first is 3/6 = .50​

If you don't spin, then the probability of getting an empty chamber is 2/3 = .66​

![image](/assets/russian-roulette.png)

If your friend did not get a bullet, then he must have fired chamber D or E of R.

- If your friend fired chamber F, and you don't spin, you will get a bullet (chamber A)

- If your friend fired chamber E or D,  and you don't spin, then you survive.

Hence the better choice is don't spin if you want to maximize your chance of surviving.


#### 2. Three ants are standing on each of the 3 corners of a triangle. If each ant moves along an edge towards a randomly chosen corner, what is the chance that none of them collide?

_Answer:_

If none of the ants collide, then they must all move in the same direction: either close-wise or counter-clockwise.
Hence, there are 2 different ways none of the ants collide.

The total number of possible movements is 2^3 = 8​

Hence the probability that none collide is 2/8 = .25​

Extension: for any polygon with n edges and n vertices, the probability of having no ants colliding is 2/(2^n​)

#### 3. A certain couple have 2 children, at least 1 of which is a girl. What is the probability that they have 2 girls?

_Answer:_

If the couple have 2 children, then the are 4 equal possibilities: BB, BG, GG, GB (B=Boy, G=Girl)

Since the couple have at least 1 girl, the remaining possibilities are BG, GG, GB. Thus the probability of having 2 girls is 1/3



#### 4. How do you estimate pi with a function that returns a number between 0 and 1 uniformly?

_Hint:_

- Estimate pi by dividing the area of a circle by its enclosing square.
- We can estimate this area by generating random points in a square. And then count the number of points that are inside the circle vs. those that are outside of it
- Check if a point is inside a circle by calculating the distance from origin using Euclidian distance. Eg. for point (x, y), the distance is sqrt(x^2 + y^2)



#### 5. In a 8-player tennis championship with single-elimination format, what is the probability that the 2nd best player ends up as 2nd place. Assuming that the better player always win?

_Anwser_:

- Wrong answer is 1/2
- Correct answer: The 2nd best player get 2nd place if he ends up at the final game. This requires him to be placed in a different bracket branch from the best player. After the best player takes up a spot randomly, then there are 4/7 chances the 2nd best player avoids the same branch as the best player. Hence the probability of him getting 2nd place is 4/7 (better than 1/2)
- Extension: For a championship consisting of 2^n players, the probability of 2nd player getting 2nd place is (2^(n-1))/ ((2^n) -1)



#### 6. In a sock drawer, there are 6 blue socks, 8 brown socks and 10 black socks. In complete darkness, how many socks would you need to pull out go get a guaranteed matching pairs?

_Answer:_

There are 3 different colors, so you need to pull out at most 4 socks. Proof using pigeonhole principle: there are 4 socks with 3 colors, then exist at least 1 color having more than 1 socks.

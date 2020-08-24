---
layout: post
title:  "Simulating the secretary problem"
date:   2020-08-23 15:30:00 +0700
tags: python simulations
---

The secretary problem is one that demonstrates a scenario involving optimal stopping theory.

The basic form of the problem is as follow: Imagine an administrator who wants to hire the best secretary out of n rankable applicants for a position. The applicants are interviewed one by one in random order. A decision about each particular applicant is to be made immediately after the interview. Once rejected, an applicant cannot be recalled.

If the decision can be deferred to the end, this can be solved trivially. The difficulty is that the decision must be made immediately.

The solution is to interview the first 37% of applicants without selecting anyone. After that, continue interviewing the remaining applicants and immediately stop if you find one who is better than all of the previous applicants.

As the number of applicants increases, the chance of success converges to 37%. Though this is less than 50% of success (choosing the best applicant), it is important to note that if we were to hire randomly in a pool of 100 applicants, the chance of selecting the best is a mere 1%.

We can simulate the algorithm as below

```python
from collections import Counter
import random

# for reproducibility
random.seed(42)
```

```python
num_candidates = 100

# initialise a list of 100 candidates
# scores corresponding to their skill level
# higher scores are better candidates
# in the best scenario, we'd hire candidates #99
candidates = list(range(num_candidates))

# cut-off
first_37_percent = round(num_candidates*.37)
```


```python
# number of simulations
m = 1_000

# store the scores of applicants selected
ranks = []

for sim in range(m):
    # rearrange candidates
    random.shuffle(candidates)

    
    # interview the first 1/3 of all candidates without accepting anyone
    # but keep track of the highest score
    best_of_third = max(candidates[:first_37_percent])
    
    # for the next 2/3 of candidates, keep interviewing, but offer the job
    # to anyone better than the previous 1/3 candidates
    for score in candidates[first_37_percent:]:
        if score > best_of_third:
            selected_candidate = score
            ranks.append(selected_candidate)
            break


# counting the number of applicants with score of 99 out of all simulations
Counter(ranks)[99] / m
# .365 (~37%)

Counter(ranks)
# even in the case of missing the best applicant
# we got very close to the best one
# Counter({99: 365,
#         98: 124,
#         95: 22,
#         96: 39,
#         97: 65,
#         94: 10,
#         93: 4,
#         91: 3,
#         89: 1,
#         92: 1})
```


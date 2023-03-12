---
layout: post
title:  Web scraping with requests and BeautifulSoup
date:   2020-08-20 12:11:27 +0700
tags: web-scraping
description: Basic web scraping with Requests and BeautifulSoup
---

### Introduction

Web scraping is useful for when we need data that's not nicely formatted in csv.
In this post I will share how to scrape headlines from a news title. There can be many potential use cases. For example, one can make a news aggregator and perhaps filter out on headline topics that they do not care about.
Or make a short sentiment analysis of the news to find out which publishers are more 'positive' :')


![CNA](/../assets/images/posts/web-scraping/img-1-cna.png)

### Code

The libraries used are [requests](https://pypi.org/project/requests/){:target="_blank"} and [BeautifulSoup](https://pypi.org/project/beautifulsoup4/){:target="_blank"}
* `requests` allows you to send HTTP requests
* `BeautifulSoup` provides a convenient interface to navigate and extract the HTML content.

Requirements
```bash
pip install requests
pip install beautifulsoup4
```

Most news websites can be scraped directly by making a request to the url. For example, here's one that fetchs the headlines from [Channel News Asia](https://www.channelnewsasia.com/news/international){:target="_blank"}




```python
import requests
from bs4 import BeautifulSoup

url = 'https://www.channelnewsasia.com/international'

# content contains the raw html of the page
content = requests.get(url).content

# BeautifulSoup parse the raw html. We use 'lmxl' parser here.
# More information about BeautifulSoup's parsers can be found in the
# documentation at https://www.crummy.com/software/BeautifulSoup/bs4/doc/
soup = BeautifulSoup(content, 'lxml')



# For the example, we will extract all teasers in the news' home page.
# Teasers are the articles' short titles.
# Chrome's Inspect tools show that teasers are <h3> headings with 
# class name "teaser__heading"
teasers = soup.find_all('a', class_='h6__link list-object__heading-link')


# Getting the first 5 entries
for idx, teaser in enumerate(teasers[:10]):
    print(idx, teaser.text.strip())
```

Ouput

```
0 'Everybody makes mistakes': Swimming fraternity shocked by Schooling, Lim's cannabis case but rally behind them
1 Exclusive-JD.com, Yum China among Chinese firms chosen for US audit inspection -sources
2 Not end of the line for the Joseph Schooling brand despite Olympic champ's drug use, say experts
3 From Olympics gold to drug confession: The Joseph Schooling timeline
4 Japan further eases border controls for tourism, allowing non-guided package tours
5 Toyota Motor to invest $5.3 billion in Japan and US for EV battery supply
6 Sheila Sim is pregnant with her second child; her 2-year-old daughter is sure the baby will be a boy
7 Hedge fund Bridgewater, Citadel Securities expand Asian footprints
8 South Korea to end pre-departure COVID-19 test requirement for international arrivals
9 Former Malaysia PM Mahathir in hospital after testing positive for COVID-19
```

In this example I printed the first 10 headlines. You can save results into a text file that can then be used for later analysis.

### End
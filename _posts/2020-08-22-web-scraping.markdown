---
layout: post
title:  "Web scraping with requests and BeautifulSoup"
date:   2020-08-22 12:11:27 +0700
---

Web scraping is something I regularly do at work. This post provides a bare bone example of the workflow. The libraries used are `requests` and `BeautifulSoup`

`requests` allows you to make HTTP requests, and `BeautifulSoup` provides a convenient interface to navigate and extract the HTML content.

Installations
```
pip install requests
pip install beautifulsoup4
```

The code
```python
import requests
from bs4 import BeautifulSoup

url = 'https://www.channelnewsasia.com/news/international'

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
teasers = soup.find_all('h3', class_='teaser__heading')

# Getting the first 5 entries
for teaser in teasers[:5]:
    print(teaser.text)

# Output:
# South Korea to ramp up coronavirus curbs over fears of second wave
# Australia's Victoria state records 13 new COVID-19 deaths, stable infections
# China reports 22 coronavirus cases, 6th day without local transmission
# New Zealand aware of Singapore's intent to establish travel, advisory to residents remains unchanged
# Malaysia deports Bangladeshi man who criticised treatment of migrants in documentary
```

The results can then be exported to a csv for further analysis.
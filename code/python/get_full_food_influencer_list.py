# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from bs4 import BeautifulSoup
import pandas as pd
import itertools
from pathlib import Path

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
data = root / 'data'
data_in = data / 'in'
data_out = data / 'out'
temp = data / 'temp'

# %%
with open(data_in / 'food_influencers_webpage.html', encoding = 'utf-8') as fp:
    soup = BeautifulSoup(fp, 'html.parser')
    profiles = soup.find_all(class_ = "trow trow-wrap")
    usernames = [p.a.text.strip()[1:] for p in profiles]
    followers = [int(next(itertools.islice(p.a.next_siblings, 2, 3)).strip().replace(',', '')) for p in profiles]

# %%
print(repr(followers))

# %%
influencers = pd.DataFrame({'username': usernames, 'num_followers': followers})

# %%
missing = ['kp_ingitsimple']
rename = {'streetsmartrd': 'streetsmart.rd',
         'carlieeeeats': 'carlie.eats',
         'memphiswings615': 'originalmemphiswings',
         'marathonnutritionist': 'marathon.nutritionist',
         'cleanfooddirtycity': 'lilydoran'}

# %%
influencers = influencers[~influencers['username'].isin(missing)]

# %%
influencers['username'] = influencers['username'].replace(to_replace = rename)

# %%
influencers.to_csv(data_in / 'list_influencers.csv', encoding = 'utf-8', index = False)

# %%
len(influencers)

# %%
len(influencers[(influencers['num_followers'] > 50000) & (influencers['num_followers'] < 200000)])

# %%

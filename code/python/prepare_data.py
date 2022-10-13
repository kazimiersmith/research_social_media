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
import pandas as pd
from pathlib import Path
from ast import literal_eval
from datetime import datetime
import re

# %%
pd.options.display.max_rows = 1000

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
data = root / 'data'
data_in = data / 'in'
data_out = data / 'out'
csv = data_out / 'csv'
temp = data / 'temp'

# %%
# CSV files with scraped post data
files = csv.glob('*.csv')

dfs = []
for f in files:
    dfs.append(pd.read_csv(f))
    
posts = pd.concat(dfs)
posts = posts.reset_index()

# %%
# Since I scrape the 7 most recent posts up to one week ago, I may scrape some posts
# multiple times
posts = posts.drop_duplicates(subset = 'shortcode', keep = 'first')

# %%
# Make sure there are no NaN values when I convert these columns to lists later
posts = posts.fillna({'caption_hashtags': '[]',
                      'sponsors': '[]',
                      'caption_mention': '[]',
                      'owner_comment_hashtags': '[]'})

# %%
posts = posts.rename(columns = {'sponsored': 'paid_partnership',
                                'sponsors': 'paid_partners'})

# %%
# Parse columns whose values are lists
posts['caption_hashtags'] = posts['caption_hashtags'].apply(literal_eval)
posts['caption_mention'] = posts['caption_mention'].apply(literal_eval)
posts['paid_partners'] = posts['paid_partners'].apply(literal_eval)
posts['owner_comment_hashtags'] = posts['owner_comment_hashtags'].apply(literal_eval)

# %%
# Terms to use to classify sponsored posts

# Terms that must match exactly. Matching all words that start with these terms would
# give a lot of false positives
exact_match = ['ad', 'sp', 'spon']

# Terms to match at the beginning. We don't want to match foodnetwork, postworkout, preworkout, etc.
start_match = ['work']

# Terms to match anywhere
any_match = ['partner', 'sponsor', 'paid']


# %%
# Determine whether a word w indicates a sponsored posts
def classify_sponsored_word(w):
    re_exact = re.compile('|'.join(exact_match), re.IGNORECASE)
    re_start = re.compile('|'.join(start_match), re.IGNORECASE)
    re_any = re.compile('|'.join(any_match), re.IGNORECASE)
    
    return (bool(re_exact.fullmatch(w)) or bool(re_start.match(w)) or bool(re_any.findall(w)))


# %%
posts['caption_hashtags_sponsored'] = [[h for h in tags if classify_sponsored_word(h)] for tags in posts['caption_hashtags']]

# %%
# Whether the post is sponsored (boolean)
posts['sponsored'] = posts.apply(lambda p: len(p['caption_hashtags_sponsored']) > 0 or p['paid_partnership'], axis = 1)


# %%
# Convert date column from str to datetime object
def parse_date(date_str):
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        # Date is not in ISO format
        return datetime.strptime(date_str, '%m/%d/%Y %H:%M')
    except TypeError:
        # Date is NaN
        return pd.NaT


# %%
posts['date'] = posts['date'].apply(parse_date)

# %%
# Sort chronologically
posts = posts.sort_values(by = 'date')

# %%
# Drop posts without a date
posts = posts[posts['date'].notnull()]

# %%
# Drop everything before July 1, 2022 (there are a few posts before that, but not many)
cutoff = datetime(2022, 7, 1)
posts = posts[posts['date'] >= cutoff]
posts['date']

# %%

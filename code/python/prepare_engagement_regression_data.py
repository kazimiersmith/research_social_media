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
# Prepare data for regressing engagement on sponsorship

import pandas as pd
from pathlib import Path
from ast import literal_eval
import re

pd.options.display.max_rows = 500

# %%
# Directories
cwd = Path.cwd()
root = cwd / '..' / '..'
data = root / 'data'
out = data / 'out'
out_csv = out / 'csv'
temp = data / 'temp'

# %%
# Post data
posts = pd.read_csv(out_csv / 'data_initial_regressions.csv')

# Make sure there are no NaN values when I convert these columns to lists later
posts = posts.fillna({'caption_hashtags': '[]',
                      'sponsors': '[]',
                      'caption_mention': '[]',
                      'owner_comment_hashtags': '[]'})
posts

# %%
# The column originally called "sponsored" is true if the post has an explicit "paid partnership with @xyz" disclosure,
# and the column originally called "sponsors" contains the list of partners. Rename these columns so the names are more
# descriptive
posts = posts.rename(columns = {'sponsored': 'paid_partnership',
                                'sponsors': 'paid_partners'})
posts

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
posts['caption_hashtags_sponsored'] = [[h for h in tags if classify_sponsored_word(h)] 
                                       for tags in posts['caption_hashtags']]
posts['owner_comment_hashtags_sponsored'] = [[h for h in tags if classify_sponsored_word(h)] 
                                             for tags in posts['owner_comment_hashtags']]


# %%
# Return one if the post has a sponsored hashtag or is marked as a paid partnership, zero otherwise
def is_sponsored(p):
    spon_hashtag = len(p['caption_hashtags_sponsored']) > 0 or len(p['owner_comment_hashtags_sponsored']) > 0
    paid = p['paid_partnership'] or len(p['paid_partners']) > 0
    
    return int(spon_hashtag or paid)


# %%
# Indicator for whether the post is sponsored
posts['sponsored'] = posts.apply(is_sponsored, axis = 1)

# %%
# Calculate a standard measure of engagement
posts['engagement'] = (posts['comments_num'] + posts['likes_num']) / posts['followers_num']

# %%
# Want to see whether posts with more mentions get more or less engagement
posts['num_caption_mentions'] = posts['caption_mention'].apply(lambda l: len(l))

# %%
cols = ['likes_num', 'comments_num', 'followers_num', 'sponsored', 'engagement', 'num_caption_mentions']

# %%

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
import numpy as np

# %%
pd.options.display.max_rows = 1000

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
data = root / 'data'
data_in = data / 'in'
data_out = data / 'out'
csv = data_out / 'csv'
estimation = data_out / 'estimation'
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
len(posts)

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
len(posts)

# %%
# Drop posts that hide likes count
posts = posts[posts['likes_num'] >= 0]
len(posts)

# %%
# Calculate a measure of engagement
posts['engagement'] = (posts['likes_num'] + posts['comments_num']) / posts['followers_num']
#posts[['likes_num', 'comments_num', 'followers_num', 'engagement']]

# %%
#posts.sort_values(by = 'followers_num', ascending = False)

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
posts = posts.sort_values(by = ['date', 'profile_username'])

# %%
# Drop posts without a date
posts = posts[posts['date'].notnull()]

# %%
len(posts)

# %%
# Drop everything before July 31, 2022 (there are a few posts before that, but not many)
cutoff = datetime(2022, 7, 31)
posts = posts[posts['date'] >= cutoff]
#posts['date']

# %%
len(posts)

# %%
len(posts['profile_username'].unique())

# %%
# Set index to date
posts = posts.set_index('date')

# %%
#posts.columns

# %%
# Convert sponsored from boolean to 0 or 1
posts['sponsored'] = posts['sponsored'].apply(lambda s: int(s))

# %%
# Engagement on sponsored and non-sponsored posts
posts['engagement_sponsored'] = posts.apply(lambda p: np.nan if p['sponsored'] == 0 else p['engagement'], axis = 1)
posts['engagement_not_sponsored'] = posts.apply(lambda p: np.nan if p['sponsored'] == 1 else p['engagement'], axis = 1)
#posts

# %%
# Branded posts: for now, any post that tags another account
posts['branded'] = posts.apply(lambda p: 1 if len(p['caption_mention']) > 0 else 0, axis = 1)

# %%
posts.to_csv(estimation / 'posts_all.csv', index = False)

# %%
# Convert to influencer-week data
# Note: including count as one of the aggregation methods for likes_num
# is just a way to count the number of posts for a given influencer in a given week
agg_methods = {'likes_num': ['mean', 'count'],
               'comments_num': 'mean',
               'followers_num': 'mean',
               'engagement': 'mean',
               'sponsored': 'sum',
               'engagement_sponsored': 'mean',
               'engagement_not_sponsored': 'mean'}
posts_panel = posts.groupby('profile_username').resample('W').agg(agg_methods)
#posts_panel

# %%
posts_panel['followers_next_period'] = posts_panel.groupby('profile_username').shift(-1)['followers_num']

# %%
len(posts_panel)

# %%
#posts_panel

# %%
posts_panel.columns

# %%
posts_panel.columns = posts_panel.columns.map('_'.join)
posts_panel = posts_panel.reset_index()
#posts_panel

# %%
posts_panel = posts_panel.rename(columns = {'date': 'week',
                                            'likes_num_mean': 'likes',
                                            'likes_num_count': 'posts',
                                            'comments_num_mean': 'comments',
                                            'followers_num_mean': 'followers',
                                            'engagement_mean': 'engagement',
                                            'sponsored_sum': 'sponsored_posts',
                                            'engagement_sponsored_mean': 'engagement_sponsored',
                                            'engagement_not_sponsored_mean': 'engagement_not_sponsored',
                                            'followers_next_period_': 'followers_next_period'})
posts_panel['fraction_sponsored'] = posts_panel['sponsored_posts'] / posts_panel['posts']
posts_panel['change_followers'] = posts_panel['followers_next_period'] - posts_panel['followers']

# %%
posts_panel.to_csv(estimation / 'posts_panel.csv', index = False)


# %%
# Prepare a dataframe for the regression used to estimate the transition function
def log_zero(x):
    if x > 0:
        return np.log(x)
    else:
        return 0
    
posts_transition = posts_panel.copy()
posts_transition['log_posts'] = posts_transition['posts'].apply(log_zero)
posts_transition['log_frac_spon'] = posts_transition['fraction_sponsored'].apply(log_zero)
posts_transition['log_engagement'] = posts_transition['engagement'].apply(log_zero)
posts_transition['log_followers'] = posts_transition['followers'].apply(log_zero)
posts_transition['log_followers_next'] = posts_transition['followers_next_period'].apply(log_zero)
posts_transition['change_log_followers'] = posts_transition['followers_next_period'] - posts_transition['followers']
posts_transition['log_likes'] = posts_transition['likes'].apply(log_zero)
posts_transition['log_comments'] = posts_transition['comments'].apply(log_zero)
posts_transition['log_spon_posts'] = posts_transition['sponsored_posts'].apply(log_zero)

# %%
posts_transition.to_csv(estimation / 'transition_estimation_data.csv', index = False)

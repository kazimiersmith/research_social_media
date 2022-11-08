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
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.max_rows = 500

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
estimation = root / 'data' / 'out' / 'estimation'
fig = root / 'fig'

# %%
posts = pd.read_csv(estimation / 'posts_all.csv')


# %%
# Categorize posts as organic, sponsored disclosed, or branded undisclosed
def classify_post(p):
    if p['sponsored'] == 1:
        return 'sponsored_disclosed'
    elif p['branded_undisclosed'] == 1:
        return 'branded_undisclosed'
    else:
        return 'organic'
    
posts['post_type'] = posts.apply(classify_post, axis = 1)
posts.groupby('post_type')[['followers_num', 'likes_num', 'comments_num', 'engagement']].mean()

# %%
posts.groupby('post_type')[['followers_num', 'likes_num', 'comments_num', 'engagement']].count()


# %%
# Create bins for summary stats by number of followers
def bin_posts_followers(p):
    if p['followers_num'] < 50000:
        return 'Less than 50,000'
    elif 50000 <= p['followers_num'] < 100000:
        return '50,000-100,000'
    elif 100000 <= p['followers_num'] < 200000:
        return '100,000-200,000'
    elif 200000 <= p['followers_num'] < 300000:
        return '200,000-300,000'
    elif 300000 <= p['followers_num'] < 400000:
        return '300,000-400,000'
    elif 400000 <= p['followers_num']:
        return 'More than 400,000'
    
posts['followers_bin'] = posts.apply(bin_posts_followers, axis = 1)
posts.groupby('followers_bin')[['sponsored', 'branded_undisclosed', 'organic']].sum()

# %%
posts.groupby('followers_bin')['profile_username'].nunique()

# %%
posts.groupby('followers_bin')[['likes_num', 'comments_num', 'engagement']].mean()

# %%
posts_panel = pd.read_csv(estimation / 'posts_panel.csv')

# %%
posts_panel.columns

# %%
min_followers = posts_panel['followers'].min()
max_followers = posts_panel['followers'].max()

# %%
min_followers

# %%
max_followers

# %%
len(posts_panel['profile_username'].unique())

# %%
min_date = posts_panel['week'].min()

# %%
min_date

# %%
max_date = posts_panel['week'].max()

# %%
max_date

# %%
# Average across influencers, to show descriptive statistics
posts_influencer_avg = posts_panel.groupby('week').mean()
posts_influencer_avg = posts_influencer_avg.reset_index()
posts_influencer_avg

# %%
posts_influencer_avg.plot(x = 'week', y = 'posts', label = 'Posts')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig(fig / 'posts.png')
plt.show()

# %%
posts_influencer_avg.plot(x = 'week', y = 'followers', label = 'Followers')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig(fig / 'followers.png')
plt.show()

# %%
posts_influencer_avg.plot(x = 'week', y = 'engagement', label = 'Engagement')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig(fig / 'engagement.png')
plt.show()

# %%
posts_influencer_avg.plot(x = 'week', y = 'sponsored_posts', label = 'Sponsored posts')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig(fig / 'sponsored_posts.png')
plt.show()

# %%
posts_influencer_avg.plot(x = 'week', y = ['engagement_sponsored', 'engagement_not_sponsored'],
                         label = ['Engagement (spon. posts)', 'Engagement (organic posts)'])
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig(fig / 'engagement_sponsored_not.png')
plt.show()

# %%
posts_influencer_avg.plot(x = 'week', y = 'fraction_sponsored', label = 'Fraction spon. posts')
plt.xticks(rotation = 30)
plt.tight_layout()
plt.savefig(fig / 'fraction_sponsored.png')
plt.show()

# %%

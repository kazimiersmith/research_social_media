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
pd.options.display.max_rows = 500

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
estimation = root / 'data' / 'out' / 'estimation'
fig = root / 'fig'

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

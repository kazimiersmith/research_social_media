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
import instaloader
import re #regular expressions
import pandas as pd
import unicodedata
import numpy as np
from pathlib import Path
import os
import datetime
import warnings
import itertools
from datetime import datetime

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
data = root / 'data'
data_in = data / 'in'
data_out = data / 'out'
today = datetime.now().strftime('%Y%m%d')

# %%
with open(data_in / 'instagram_mobile_user_agent.txt', 'r') as f:
    mobile_user_agent = f.read()
mobile_user_agent

# %%
# Get instance. Use iphone user agent so requests to Instagram look less suspicious
L = instaloader.Instaloader(user_agent = mobile_user_agent)

# Login to Instagram using session file created with instaloader --login in terminal.
# Each person running the code needs to do this themselves. Instagram seems to require being logged in to access most information.
# Don't use an account you care a lot about - it could get banned due to scraping
L.load_session_from_file('kazimiersmith')

# %%
influencer_list_full = pd.read_csv(data_in / 'list_influencers.csv', encoding = 'utf-8')
#influencer_list_full = pd.read_csv('list_influencers_5.csv', encoding = 'utf-8')

# %%
# For the initial regression of engagement on sponsorship, use influencers with
# 50,000 to 200,000 followers
influencer_list = influencer_list_full[(influencer_list_full['num_followers'] > 50000) 
                                       & (influencer_list_full['num_followers'] < 200000)]['username']
#influencer_list = influencer_list_full['username']

# %%
def user_to_json(influencer, num_posts, replace_json = True):
    print('Downloading posts from', influencer)
    Profile = instaloader.Profile
    profile = Profile.from_username(L.context, influencer)

    # get_posts returns posts in the order you see them when you visit the profile. This generally means
    # it returns the most recent posts firsts, but if there are pinned posts, it will return those.
    posts = profile.get_posts()
    
    # Ignore pinned posts, since they might be from a long time ago
    # TODO maybe actually check the dates? Probably won't make much difference
    posts_non_pinned = itertools.filterfalse(lambda p: p.is_pinned, posts)
    
    for post in itertools.islice(posts_non_pinned, num_posts):
        shortcode = post.shortcode
        L.save_metadata_json(str(data_out / 'json' / today / shortcode), post) 


# %%
for influencer in influencer_list:
    user_to_json(influencer, 5)


# %%
# Function to grab objects of interest from post object

def objects_from_post(post):
    # Shortcode
    shortcode = 'https://www.instagram.com/p/' + post.shortcode
    
    # Date
    postdate = post.date
    
    # Profile
    profile = post.owner_profile
    
    # Username of post's owner
    profile_username = post.owner_username
    
    # Extract location
    location = post.location
    
    if not location:
        loc_name = float('nan')
        loc_lng = float('nan')
        loc_lat = float('nan')
    else:
        loc_name = location.name
        loc_lng = location.lng
        loc_lat = location.lat
        
    # Extract image URL
    image_url = post.url
    
    # Number of likes
    likes_num = post.likes 
    
    # Number of comments
    comments_num = post.comments
    
    # Find ID of likes
    #postlikes = []
    #for likes in post.get_likes():
    #    postlikes.append(likes)
    
    # Extract caption
    # Caption
    caption = post.caption
    
    # Caption hashtag
    caption_hashtags = post.caption_hashtags
    
    # Caption mentions (profiles mentioned in caption)
    caption_mention = post.caption_mentions
    
    # Whether the post is sponsored (i.e. "Paid partnership with...")
    sponsored = post.is_sponsored
    
    # List of the post's sponsors (usernames)
    sponsors = [p.username for p in post.sponsor_users]
    
    # Number of followers
    followers_num = profile.followers
    
    # Chronologically earliest comment, to search for hashtags. Influencers sometimes put hashtags in a separate comment,
    # usually the first comment on the post. Note that post.get_comments() does not necessarily return
    # the chronologically earliest comment as the first item.
#     start = time.time()
#     first_comment = min(post.get_comments(), key = lambda p: p.created_at_utc)
#     if first_comment:
#         first_comment_text = first_comment.text
        
#         # Is the first comment by the owner of the original post?
#         first_comment_by_owner = (first_comment.owner.username == profile_username)
        
#         # If the first comment is by the owner of the original post, get the (unique) hashtags from the first comment
#         first_comment_hashtags = list(set(part[1:] for part in first_comment_text.split() if part.startswith('#')))
#     else:
#         first_comment_text = float('nan')
#         first_comment_by_owner = float('nan')
#         first_comment_hastags = float('nan')
        
#     end = time.time()
#     print('Getting first comment hashtags took', str(end - start), 'seconds')
    
    if post.comments > 0:
        owner_comments = (c for c in post.get_comments() if c.owner.username == profile_username)
        owner_comment_hashtags = [part[1:] for c in owner_comments for part in c.text.split() if part.startswith('#')]
        owner_comment_hashtags_unique = list(set(owner_comment_hashtags))
    else:
        owner_comment_hashtags_unique = float('nan')
    
    data = {'shortcode': shortcode,
            'date': postdate,
            'profile_username': profile_username,
            'location_name': loc_name,
            'location_lat': loc_lat,
            'location_lng': loc_lng,
            'image_url': image_url,
            'likes_num': likes_num,
            'comments_num': comments_num,
            'caption': caption,
            'caption_hashtags': caption_hashtags,
            'caption_mention': caption_mention,
            'sponsored': sponsored,
            'sponsors': sponsors,
            'followers_num': followers_num,
            'owner_comment_hashtags': owner_comment_hashtags_unique}
        
    return data


# %%
def empty_dict(shortcode):
    the_dict = {'shortcode': shortcode, 
                'date': float('nan'),
                'profile_username': float('nan'),
                'location_name': float('nan'),
                'location_lat': float('nan'),
                'location_lng': float('nan'),
                'image_url': float('nan'),
                'likes_num': float('nan'),
                'comments_num': float('nan'),
                'caption': float('nan'),
                'caption_hashtags': float('nan'),
                'caption_mention': float('nan'),
                'sponsored': float('nan'),
                'sponsors': float('nan'),
                'followers_num': float('nan'),
                'owner_comment_hashtags': float('nan')}
    return the_dict


# %%
json_today = data_out / 'json' / today
json_files = list(json_today.glob('*.json.xz'))
count = 0
list_dicts = []
for file in json_files:
    count += 1
    print('Post {} of {}\n'.format(count, len(json_files)))
    print(file)

    post = instaloader.load_structure_from_file(L.context, str(file))
    # To handle a postexception error
    try:
        list_dicts.append(objects_from_post(post))
    except Exception as e:
        print('Error getting post information:', e)
        list_dicts.append(empty_dict(post.shortcode))

df = pd.DataFrame(list_dicts)

# %%
# Encoding needs to be UTF8-sig, otherwise apostrophes, emojis etc. get messed up
outfile = today + '.csv'
outpath = data_out / 'csv' / outfile
df.to_csv(outpath, encoding = 'utf-8-sig', index = False) 

# %%

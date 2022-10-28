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
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from scipy.special import logsumexp

pd.options.display.max_rows = 500
np.set_printoptions(threshold = 100000)

# %%
root = Path('C:/Users/kas1112/Documents/research_social_media')
estimation = root / 'data' / 'out' / 'estimation'

# %%
default_discount_factor = 0.9

# Transition function intercept and coefficients, estimated via OLS
constant = 12.31954
gamma_sponsored = -0.12789
gamma_engagement = 0.08518

# Initial guess for sponsored post revenue
initial_alpha = 0.01

# Initial guess for cost function coefficients:
# c(p) = theta_1 * p + theta_2 * p^2
initial_theta1 = 1
initial_theta2 = 1

# Bins for discretizing number of followers
num_bins = 20

# Maximum number of posts in a given period. This defines the influencer's choice set.
# For now a period is a week.
max_posts = 7

# %%
posts_panel = pd.read_csv(estimation / 'posts_panel.csv')

# %%
# Max and min number of followers in my data
min_followers = posts_panel['followers'].min()
max_followers = posts_panel['followers'].max()

# %%
# Label the bins with numbers rather than values corresponding to the actual values in the bin
bins = np.linspace(min_followers, max_followers, num = num_bins)
bins


# %%
def assign_bin(x):
    for i, b in enumerate(bins):
        if i < num_bins - 1:
            if b <= x < bins[i + 1]:
                return i
        else:
            if b <= x:
                return i
    return np.nan

posts_panel['followers_binned'] = posts_panel['followers'].apply(assign_bin)
posts_panel['followers_next_binned'] = posts_panel['followers_next_period'].apply(assign_bin)
posts_panel = posts_panel.dropna(axis = 0, subset = 'followers_binned')
posts_panel = posts_panel.dropna(axis = 0, subset = 'followers_next_binned')

# %%
transition = []
for f in range(num_bins):
    # Observations where number of followers (binned) is f
    df_f = posts_panel[posts_panel['followers_binned'] == f]
    
    # Get frequencies of each value of next period followers; normalize = True makes them probabilities
    transition_probs_nonzero = df_f.value_counts(subset = 'followers_next_binned', normalize = True)
    
    transition_probs = np.zeros(num_bins)
    for b, prob in transition_probs_nonzero.iteritems():
        transition_probs[int(b)] = prob
        
    #print(transition_probs)
    
    # Check whether all transition probabilities are zero
    if not np.any(transition_probs):
        transition_probs[int(f)] = 1
        
    transition.append(transition_probs)
    
# Transition matrix
transition = np.array(transition)
transition


# %%
# Influencer's utility
def utility(followers, posts, sponsored_posts, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    return alpha * sponsored_posts * followers + theta1 * posts + theta2 * posts * posts


# %%
# One iteration of the Bellman operator.
def bellman_iteration(**kwargs):
    # Previous value of the expected value function; this should be a vector of length num_bins
    # where each element is the expected value function evaluated at that bin (i.e. number of followers)
    prev_ev = kwargs.get('prev_ev', np.zeros((num_bins, max_posts + 1, max_posts + 1)))
    
    beta = kwargs.get('beta', default_discount_factor)
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    new_ev = np.zeros((num_bins, max_posts + 1, max_posts + 1))
    for followers in range(num_bins):
        for posts in range(max_posts + 1):
            for spon in range(posts + 1):
                # Must have sponsored posts <= posts
                #print(followers, posts, spon)
                integral_summands = []
                for b in range(num_bins):
                    # f is the number of followers corresponding to bin b
                    # I use the midpoints of the bins, except for the last point, where I use the left endpoint
                    if b < num_bins - 1:
                        f = (bins[b] + bins[b + 1]) / 2
                    else:
                        f = bins[b]

                    exp = []
                    for p in range(max_posts + 1):
                        for s in range(p + 1):
                            # Influencer can make s sponsored posts, where 0 <= s <= p
                            exp.append(utility(f, p, s, alpha = alpha, theta1 = theta1, theta2 = theta2) + beta * prev_ev[b][p][s])

                    integral_summands.append(logsumexp(exp) * transition[followers][b])

                new_ev[followers][posts][spon] = np.sum(integral_summands)
                         
    return new_ev


# %%
def iterate_bellman(**kwargs):
    print('Iterate Bellman')
    epsilon_tol = 0.000001
    
    beta = kwargs.get('beta', default_discount_factor)
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    prev_exp_vf = np.zeros((num_bins, max_posts + 1, max_posts + 1))
    new_exp_vf = bellman_iteration(prev_ev = prev_exp_vf, beta = beta, alpha = alpha, theta1 = theta1, theta2 = theta2)
    epsilon = np.amax(np.abs(new_exp_vf - prev_exp_vf))
    while epsilon > epsilon_tol:
        prev_exp_vf = np.copy(new_exp_vf)
        new_exp_vf = bellman_iteration(prev_ev = prev_exp_vf, beta = beta, alpha = alpha, theta1 = theta1, theta2 = theta2)
        epsilon = np.amax(np.abs(new_exp_vf - prev_exp_vf))
        print('Epsilon = ', epsilon)
        
    return np.array(new_exp_vf)


# %%
def get_ccps(exp_vf):
    print('CCPs')
    # TODO ignore entries in exp_vf where sponsored posts > posts
    denominators = np.array([np.full((max_posts + 1, max_posts + 1), np.sum(np.exp(exp_vf[b][:][:]))) for b in range(num_bins)])
    replace_probs = np.exp(exp_vf) / denominators
    
    return replace_probs


# %%
# Params are alpha, theta1, theta2
def neg_log_likelihood(params):
    print('Negative log likelihood')
    exp_vf = iterate_bellman(alpha = params[0], theta1 = params[1], theta2 = params[2])
    ccps = get_ccps(exp_vf)
    obs_data = np.array(posts_panel[['followers_binned', 'posts', 'sponsored_posts']]).astype(int)
    
    return -np.sum(np.log([ccps[o[0]][o[1]][o[2]] for o in obs_data]))


# %%
# Estimate theta1 and the replacement cost using MLE
def estimate_mle():
    print('Estimate MLE')
    # Minimize the negative log-likelihood, i.e. maximize the log-likelihood
    return minimize(neg_log_likelihood,
                    [initial_alpha, initial_theta1, initial_theta2],
                    method = 'BFGS')


# %%
estimates = estimate_mle()

# %%
estimates

# %%
# Utility estimates
# def get_utility(val_func, **kwargs):
#     beta = kwargs.get('beta', default_beta)
#     t0 = kwargs.get('transition0', transition0)
#     t1 = kwargs.get('transition1', transition1)
    
#     # log(exp(v(x, 0)) + exp(v(x, 1)))
#     vf_logsumexp = np.log(np.exp(val_func[:, 0]) + np.exp(val_func[:, 1])).reshape(-1, 1)
    
#     v0 = val_func[:, 0].reshape(-1, 1) - beta * t0 @ vf_logsumexp
#     v1 = val_func[:, 1].reshape(-1, 1) - beta * t1 @ vf_logsumexp
    
#     return np.array([v0.flatten(), v1.flatten()]).T

# # Estimates: theta1 = 688.6650, RC = 146.9811
# mle_estimates = estimate_mle()

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
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from scipy.special import logsumexp

pd.options.display.max_rows = 500
np.set_printoptions(threshold = 100000)

# %%
root = Path('C:/Users/kas1112/Dropbox/my_research_social_media')
estimation = root / 'data' / 'out' / 'estimation'

# %%
discount_factor = 0.9

carrying_capacity = 3500000

# Coefficients in follower growth equation
beta_0 = 1
beta_sponsored = -0.5
beta_organic = 1
beta_engagement = 1

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

# Degree of Chebyshev polynomial for value function approximation. Should be >= num_grid_points
chebyshev_degree = 20

# Number of grid points to use for value function approximation
num_grid_points = chebyshev_degree + 1

# Number of samples to use for Monte Carlo integration
monte_carlo_samples = 20

# Assume the error term in the follower growth equation is normally distributed with the following mean and stanard deviation:
follower_error_mean = 0
follower_error_std_dev = 100

# %%
posts_panel = pd.read_csv(estimation / 'posts_panel.csv')

# %%
# Max and min number of followers in my data
min_followers = posts_panel['followers'].min()
max_followers = posts_panel['followers'].max()

# %%
# Grid points suggested in RMT 4th edition, citing Judd (1996, 1998)
chebyshev_zeros = np.array([np.cos((2 * k - 1) / (2 * num_grid_points) * np.pi) for k in range(1, num_grid_points + 1)])


# %%
# Scale Chebyshev zeros to obtain grid points (Chebyshev zeros are in [-1, 1])
# r_min: lower bound of starting interval
# r_max: upper bound of starting interval
# t_min: lower bound of target interval
# t_max: upper bound of target interval
#: m: number to scale
def scale(r_min, r_max, t_min, t_max, m):
    return (m - r_min) / (r_max - r_min) * (t_max - t_min) + t_min

grid_points = np.array([scale(-1, 1, min_followers, max_followers, z) for z in chebyshev_zeros])
grid_points

# %%
# Chebyshev coefficients. These define the value function completely.
# Choose zero as the initial value for all coefficients.
chebyshev_coefficients = np.zeros(chebyshev_degree + 1)


# %%
# Influencer's utility
# No branded posts for now
def utility(followers, sponsored_posts, organic_posts, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    total_posts = sponsored_posts + organic_posts
    
    return alpha * sponsored_posts * followers - theta1 * total_posts - theta2 * total_posts ** 2


# %%
[5] * 4


# %%
# Calculate the mean of the distribution of the number of followers next period
def followers_next_mean(followers, sponsored_posts, organic_posts, engagement):
    cap_term = followers * (1 - followers / carrying_capacity)
    
    mean = followers
    mean += beta_0 * cap_term
    mean += beta_organic * organic_posts * cap_term
    mean += beta_sponsored * sponsored_posts * cap_term
    mean += beta_engagement * engagement * cap_term
    
    return mean


# %%
def integrate_monte_carlo(g, num_samples, mean, std_dev):
    draws = stats.norm.rvs(loc = mean, scale = std_dev, size = num_samples)
    return np.sum(g(draws)) / num_samples


# %%
# The expression to maximize in the value function
def value_function_objective(followers, sponsored_posts, organic_posts, engagement, chebyshev_coefficients, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    # Evaluate a Chebyshev series with the given coefficients
    def cheb_eval(x):
        return np.polynomial.chebyshev.chebval(x, chebyshev_coefficients)
    
    # Discounted expected value of the value function next period
    mean = followers_next_mean(followers, sponsored_posts, organic_posts, engagement)
    discounted_EV = discount_factor * integrate_monte_carlo(cheb_eval, monte_carlo_samples, mean, follower_error_std_dev)
    
    return utility(followers, sponsored_posts, organic_posts, alpha = alpha, theta1 = theta1, theta2 = theta2) + discounted_EV


# %%
# Given parameters, iterate on the value function approximation algorithm (RMT 4th ed.)
def iterate_approximation(**kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    # Calculate the value function at each grid point
    for g in grid_points:
        


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

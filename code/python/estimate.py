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
from scipy.special import eval_chebyt
import matplotlib.pyplot as plt

pd.options.display.max_rows = 500
np.set_printoptions(threshold = 100000)

# %%
root = Path('C:/Users/kas1112/Dropbox/my_research_social_media')
estimation = root / 'data' / 'out' / 'estimation'

# %%
discount_factor = 0.9

carrying_capacity = 3500000

# Coefficients in follower growth equation
beta_0 = 0.0001
beta_sponsored = -0.00005
beta_organic = 0.0001
beta_engagement = 0.0001

# Initial guess for sponsored post revenue
initial_alpha = 0.5

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
chebyshev_degree = 5

# Number of grid points to use for value function approximation
num_grid_points = 20

# Number of samples to use for Monte Carlo integration
monte_carlo_samples = 200000

# Assume the error term in the follower growth equation is normally distributed with the following mean and stanard deviation:
follower_error_mean = 0
follower_error_std_dev = 10

# Tolerance when iterating value function
epsilon_tol = 0.000001

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
initial_chebyshev_coefficients = np.zeros(chebyshev_degree + 1)


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
# Calculate the mean of the distribution of the number of followers next period
def followers_next_mean(followers, sponsored_posts, organic_posts, engagement):
    cap_term = followers * (1 - followers / carrying_capacity)
    
    #print('cap_term=', str(cap_term))
    
    mean = followers
    mean += beta_0 * cap_term
    mean += beta_organic * organic_posts * cap_term
    mean += beta_sponsored * sponsored_posts * cap_term
    mean += beta_engagement * engagement * cap_term
    
    return mean


# %%
def integrate_monte_carlo(g, num_samples, mean, std_dev):
    draws = stats.norm.rvs(loc = mean, scale = std_dev, size = num_samples)
    #print(draws)
    #print(g(draws))
    return np.sum(g(draws)) / num_samples


# %%
# The expression to maximize in the value function
def value_function_objective(choice_vars, followers, engagement, chebyshev_coefficients, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    sponsored_posts = choice_vars[0]
    organic_posts = choice_vars[1]
    
    # Evaluate a Chebyshev series with the given coefficients
    def cheb_eval(x):
        return np.polynomial.chebyshev.chebval(x, chebyshev_coefficients)
    
    # Discounted expected value of the value function next period
    mean = followers_next_mean(followers, sponsored_posts, organic_posts, engagement)
    print('Mean =', mean)
    # print('Followers =', followers)
    # print('Sponsored posts =', sponsored_posts)
    # print('Organic posts =', organic_posts)
    EV = integrate_monte_carlo(cheb_eval, monte_carlo_samples, mean, follower_error_std_dev)
    discounted_EV = discount_factor * EV
    #print('Discounted EV', str(discounted_EV))
   
    # Minimize the negative
    return -utility(followers, sponsored_posts, organic_posts, alpha = alpha, theta1 = theta1, theta2 = theta2) - discounted_EV


# %%
initial_alpha = 0.0001
beta_0 = 0.001
beta_sponsored = -100000
beta_organic = 0.001
beta_engagement = 0.001
spon = np.linspace(0, 45000, 100)
vf = [-value_function_objective([s, 7], 200000, 0.001, [0.09, 0.04, 0.03, 0.02, 0.01]) for s in spon]
plt.plot(spon, vf)

# %%
org = np.linspace(0, 100, 100)
vf = [-value_function_objective([2, o], 200000, 0.001, [0.09, 0.04, 0.03, 0.02, 0.01]) for o in org]
plt.plot(spon, vf)


# %%
# Given parameters, iterate on the value function approximation algorithm (RMT 4th ed.)
def iterate_approximation(initial_chebyshev_coefficients, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
   
    # Initial Chebyshev coefficients
    chebyshev_coefficients = initial_chebyshev_coefficients
    
    # Values of the jth Chebyshev polynomial T_j at each grid point
    chebvals = np.array([[eval_chebyt(j, g) for g in grid_points] for j in range(chebyshev_degree + 1)])
        
    # Denominator when calculating least-squares Chebyshev coefficients    
    denominators = np.array([np.dot(chebvals[j], chebvals[j]) for j in range(chebyshev_degree + 1)])
    
    epsilon = 1000000
    while epsilon > epsilon_tol:
        print('Epsilon =', epsilon)
        # Calculate value function at each grid point
        values = []
        results = []
        for g in grid_points:
            minimize_result = minimize(value_function_objective, [1, 5],
                                       args = (g, 0.01, chebyshev_coefficients),
                                       bounds = [(0, None), (0, None)])
                                       
            #print('Value function maximized at', minimize_result['x'])
            #print('Max value =', -minimize_result['fun'])
            results.append(minimize_result)
            values.append(-minimize_result['fun'])
            
        print('VF at grid points:', results)    
        # Calculate new Chebyshev coefficients
        numerators = np.array([np.dot(values, chebvals[j]) for j in range(chebyshev_degree + 1)])
        chebyshev_coefficients_new = numerators / denominators
        #print('cheb coeffs new:', chebyshev_coefficients_new)
        
        vf_old = np.array([np.polynomial.chebyshev.chebval(g, chebyshev_coefficients) for g in grid_points])
        vf_new = np.array([np.polynomial.chebyshev.chebval(g, chebyshev_coefficients_new) for g in grid_points])
        # print('vf_old =', vf_old)
        # print('vf_new =', vf_new)
        # print(np.abs(vf_old - vf_new))
        chebyshev_coefficients = chebyshev_coefficients_new
        epsilon = np.linalg.norm(vf_old - vf_new)


# %%
iterate_approximation(initial_chebyshev_coefficients)

# %%

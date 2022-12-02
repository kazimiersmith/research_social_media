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
from numpy.polynomial import Chebyshev
from numpy.polynomial.chebyshev import chebpts1
from scipy.integrate import quad

pd.options.display.max_rows = 500
np.set_printoptions(threshold = 100000)

# %%
#root = Path('C:/Users/kas1112/Dropbox/my_research_social_media')
root = Path('~/Dropbox/my_research_social_media')
estimation = root / 'data' / 'out' / 'estimation'

# %%
discount_factor = 0.9

carrying_capacity = 1

# Coefficients in follower growth equation
beta_0 = 0.000001
beta_sponsored = -0.01
beta_organic = 0.02
beta_engagement = 0.00001

# Initial guess for sponsored post revenue
initial_alpha = 0.01

# Initial guess for cost function coefficients:
# c(p) = theta_1 * p + theta_2 * p^2
initial_theta1 = 0.001
initial_theta2 = 0.000014

# Maximum number of posts in a given period. This defines the influencer's choice set.
# For now a period is a week.
#max_posts = 7

# Degree of Chebyshev polynomial for value function approximation. Should be less than or equal to num_grid_points
chebyshev_degree = 5

# Number of grid points to use for value function approximation
num_grid_points = 20

# Assume the error term in the follower growth equation is normally distributed with the following mean and stanard deviation:
follower_error_mean = 0
follower_error_std_dev = 0.1

# Default engagement rate
default_engagement = 0.01

# Tolerance when iterating value function
epsilon_tol = 0.001

# %%
posts_panel = pd.read_csv(estimation / 'posts_panel.csv')

# %%
# Max and min number of followers in my data
min_followers = posts_panel['followers'].min()
max_followers = posts_panel['followers'].max()

# %%
# Grid points suggested in RMT 4th edition, citing Judd (1996, 1998)
grid_points = chebpts1(num_grid_points)


# %%
# Scale Chebyshev zeros to obtain grid points (Chebyshev zeros are in [-1, 1])
# r_min: lower bound of starting interval
# r_max: upper bound of starting interval
# t_min: lower bound of target interval
# t_max: upper bound of target interval
#: m: number to scale
def scale(r_min, r_max, t_min, t_max, m):
    return (m - r_min) / (r_max - r_min) * (t_max - t_min) + t_min


# %%
# Initial Chebyshev series representing the value function
# Choose zero as the initial value for all coefficients.
C_initial = Chebyshev(np.zeros(chebyshev_degree + 1))


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
followers = np.linspace(-1, 1, 100)
spon = np.linspace(0, 8, 100)
org = np.linspace(0, 8, 100)
mean_f = [followers_next_mean(f, 0.25, 3, default_engagement) for f in followers]
mean_s = [followers_next_mean(0.1, s, 3, default_engagement) for s in spon]
mean_o = [followers_next_mean(0.1, 0.25, o, default_engagement) for o in org]

# %%
plt.plot(followers, mean_f)

# %%
plt.plot(spon, mean_s)

# %%
plt.plot(org, mean_o)


# %%
# def integrate_monte_carlo(g, num_samples, mean, std_dev):
#     draws = stats.norm.rvs(loc = mean, scale = std_dev, size = num_samples)
#     #print(draws)
#     #print(g(draws))
#     return np.sum(g(draws)) / num_samples

# %%
# The expression to maximize in the value function
# choice_vars: the number of sponsored and organic posts (eventually, branded as well)
# followers: the influencer's number of followers
# engagement: the influencer's engagement (in a given period)
# C: the current Chebyshev approximation of the value function (the algorithm calculates new coefficients at each iteration)
def value_function_objective(choice_vars, followers, engagement, C, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
    
    sponsored_posts = choice_vars[0]
    organic_posts = choice_vars[1]
    
    # Discounted expected value of the value function next period
    mean = followers_next_mean(followers, sponsored_posts, organic_posts, engagement)
    # print('Mean =', mean)
    
    # Integrand for expected value function
    def integrand(x):
        return C(x) * stats.norm.pdf(x, loc = mean, scale = follower_error_std_dev)
   
    # Discounted expected value of next period's value function.
    # Calculations are done with the number of followers in [-1, 1]; it will be scaled later
    #print(quad(integrand, -1, 1, full_output = 1))
    discounted_EV = discount_factor * quad(integrand, -1, 1)[0]
    #print('Discounted EV', str(discounted_EV))
    current_period_util = utility(followers, sponsored_posts, organic_posts, alpha = alpha, theta1 = theta1, theta2 = theta2)
    #print('Current period utility =', str(current_period_util))
   
    # Minimize the negative
    return -current_period_util - discounted_EV


# %%
# Given parameters, iterate on the value function approximation algorithm (RMT 4th ed.).
# C_initial is the initial Chebyshev approximation of the value function
def iterate_approximation(C_initial, **kwargs):
    alpha = kwargs.get('alpha', initial_alpha)
    theta1 = kwargs.get('theta1', initial_theta1)
    theta2 = kwargs.get('theta2', initial_theta2)
   
    # Initial Chebyshev approximation
    C = C_initial
    
    epsilon = 1000000
    while epsilon > epsilon_tol:
        spon = np.linspace(0, 20, 100)
        org = np.linspace(0, 20, 100)
        vf_s = [-value_function_objective([s, 3], 0.1, default_engagement, C) for s in spon]
        vf_o = [-value_function_objective([0.25, o], 0.1, default_engagement, C) for o in org]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (5, 5))
        ax1.plot(spon, vf_s)
        ax1.set_title('Sponsored posts')
        ax2.plot(org ,vf_o)
        ax2.set_title('Organic posts')
        fig.tight_layout()
        plt.show()
                 
        # Calculate value function at each grid point
        values = []
        results = []
        for g in grid_points:
            minimize_result = minimize(value_function_objective, [0, 0],
                                       args = (g, default_engagement, C),
                                       bounds = [(0, None), (0, None)])
                                       
            #print('Value function maximized at', minimize_result['x'])
            #print('Max value =', -minimize_result['fun'])
            results.append(minimize_result)
            values.append(-minimize_result['fun'])
            
        #print('VF at grid points:', results)    
        
        # New Chebyshev approximation is the least squares fit to the value function evaluated
        # at each grid point
        C_new = C.fit(grid_points, values, deg = chebyshev_degree, domain = [-1, 1])
        
        vf_old = C(grid_points)
        vf_new = C_new(grid_points)
        
        C = C_new
        epsilon = np.linalg.norm(vf_old - vf_new)
        print('Epsilon =', epsilon)
        
    return C


# %%
C_final = iterate_approximation(C_initial)

# %%
C_final

# %%
spon = np.linspace(0, 1, 100)
vf = [-value_function_objective([s, 3], 0.1, default_engagement, C_final) for s in spon]
plt.plot(spon, vf)

# %%
org = np.linspace(0, 8, 100)
vf = [-value_function_objective([0.25, o], 0.1, default_engagement, C_final) for o in org]
plt.plot(spon, vf)

# %%
followers = np.linspace(-1, 1, 100)
vf = C_final(followers)
plt.plot(followers, vf)
plt.ylabel('Value of optimal policy')
plt.xlabel('Number of followers (scaled)')


# %%
# Calculate the optimal policy for a given number of followers
def policy_function(n):
    minimize_result = minimize(value_function_objective, [0, 0],
                               args = (n, default_engagement, C_final),
                               bounds = [(0, None), (0, None)])
    
    return minimize_result['x']


# %%
optimal_policy = np.array([policy_function(n) for n in followers])
optimal_sponsored_posts = optimal_policy.T[0]
optimal_organic_posts = optimal_policy.T[1]
plt.plot(followers, optimal_sponsored_posts, color = 'blue', label = 'Sponsored')
plt.plot(followers, optimal_organic_posts, color = 'red', label = 'Organic')
plt.ylabel('Optimal number of posts')
plt.xlabel('Number of followers (scaled)')
plt.legend()
plt.show()

# %%

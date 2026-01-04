# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:39:35 2025

@author: dile
"""

import math
import numpy as np
from numba import njit

# ... (keep lmax_general, sample_arrival_general, etc.) ...

# --- NEW Numba-Compatible Samplers ---
# These now match the parameters loaded by distros.py

@njit
def sample_exp(params_tuple):
    """
    Samples from exponential.
    params_tuple: (mu,)  (where mu is the rate)
    """
    mu = params_tuple[0]
    return -math.log(np.random.rand()) / mu

@njit
def sample_bpar(params_tuple):
    """
    Samples from Bounded Pareto using inverse transform.
    params_tuple: (xmin, xmax, shape)
    """
    k, m, a = params_tuple  # k=xmin, m=xmax, a=shape
    
    # Handle edge cases
    if k == m:
        return k
    if a == 0: # Uniform distribution
        return k + np.random.rand() * (m - k)

    u = np.random.rand()
    
    # Inverse transform formula:
    # x = (k^-a - U * (k^-a - m^-a)) ^ (-1/a)
    k_a = k**(-a)
    m_a = m**(-a)
    
    inner = k_a - u * (k_a - m_a)
    
    if inner <= 0: # Should not happen if k < m
        return k
        
    return inner**(-1.0 / a)

@njit
def sample_hyperexp(params_tuple):
    """
    Samples from 2-phase Hyperexponential.
    params_tuple: (rate1, rate2, p_phase1)
    """
    r1, r2, p1 = params_tuple
    
    # pick the branch
    if np.random.rand() < p1:
        # sample from Exp(r1)
        return -math.log(np.random.rand()) / r1
    else:
        # sample from Exp(r2)
        return -math.log(np.random.rand()) / r2

@njit
def sample_erlk(params_tuple):
    """
    Samples from Erlang-k.
    params_tuple: (rate, k)
    
    Uses formula from distros.py: mean = 1/rate
    """
    rate, k = params_tuple
    k_int = int(k)
    
    # sum k exponential samples (with rate 1)
    sum_exp = 0.0
    for _ in range(k_int):
        sum_exp += -math.log(np.random.rand())
    
    # scale by (k * rate)
    divisor = k * rate
    if divisor == 0:
        return 0.0
        
    return sum_exp / divisor

@njit
def lambda_ideal(taus, ps, reqs, s):
    '''
    computes ideal arrival rate if the system were work-conserving
    '''
    return s / np.sum(taus * ps * reqs)


@njit
def fast_overwrite_and_merge(v, alpha, l):
    """
    This function updates the sorted workload vector.
    
    It assumes v is a sorted array (workloads on servers), it simulates
    a new job arriving that needs alpha servers, the job's workload is based on
    alpha-th server. It removes the alpga smallest workloads (servers 0 to alpha-1)
    and adds the new workload into the remining list, maintaining sorted oreder.    
    """
    # find the insertion point for the new workload
    insert_pos = np.searchsorted(v[alpha:], l) + alpha
    
    # builds the new sorted array efficiently 
    out = np.empty_like(v)
    out[:insert_pos - alpha] = v[alpha:insert_pos - alpha + alpha]
    out[insert_pos - alpha:insert_pos] = l
    out[insert_pos:] = v[insert_pos - alpha + alpha:]
    return out

@njit
def sample_arrival_general(probs, cumsum_probs, sampler, sigma_params_list, requests, r_class):
    """
    This function samples one arrival. 
    
    Params:
        probs (np.ndarray): Array of class probabilities.
        cumsum_probs (np.ndarray): Cumulative sum of `probs`.
        sampler (numba.Dispatcher): The Numba-jitted sampler function to call.
        sigma_params_list (list[tuple]): List of parameter tuples for each class.
        requests (np.ndarray): Array of server counts ('alpha') for each class.
        r_class (float): A single random number [0, 1) for class selection.
        
    Returns:
        (float, int): (sigma, alpha) - The service time and server request count.
    """
    # uses the passed-in random number for class selection
    r = r_class 
    c = 0
    for i in range(len(probs)):
        if r < cumsum_probs[i]:
            c = i
            break
            
    # gets the specific parameters for the chosen class
    params_for_class_c = sigma_params_list[c]
    
    # sampler will generate its own random numbers.
    sigma = sampler(params_for_class_c)

    # returns the service time and the server count
    return sigma, requests[c]

@njit
def lmax_general(s, classes, sampler, sigma_params_list, taus, requests, probs, perc, initl):
    """
    Finds the maximum stable arrival rate (lambda_max) by simulating the
    system and finding its steady-state drift `gamma`.
    
    Uses a doubling scheme (similar to perfect sampling) to run the
    simulation until gamma converges.
    
    The max arrival rate is the inverse of the drift: lmax = 1 / gamma.
    """
    cumsum_probs = np.cumsum(probs)
    w = np.zeros(s) # workload vector, sorted
    lprev = 0       # previous simulation length
    l = initl       # current simulation length
    gammaprev = -math.inf
    gammacur = 0.0  # the system drift (converges to 1 / lmax)
    oldmin = 0.0    # tracks the subtracted minimum workload

    # compute ideal arrival rate and the convergence tolerance
    lideal = lambda_ideal(taus, probs, requests, s)
    epsilon = 1 / lideal * (perc / 100)

    # keeps doubling ell until gamma converges
    while abs(gammacur - gammaprev) > epsilon:
        gammaprev = gammacur
        for i in range(lprev, l):
            
            
            # random number for class selection
            r_class = np.random.rand() 

            # get sigma and alpha for this arriva
            sigma, alpha = sample_arrival_general(probs, cumsum_probs, 
                                                  sampler, sigma_params_list, 
                                                  requests, r_class)
            # workload update
            w = fast_overwrite_and_merge(w, alpha, w[alpha - 1] + sigma)
        
        # update the long-run average-drift (gamma)
        
        # normalise the workload vector to prevent it from
        # drifting to infinity and causing precision errors,
        # we subtract the minimum and track it in `oldmin`.
        gammacur = (w[s - 1] + oldmin) / l
        oldmin += w[0]
        w -= w[0]
        
        # prepare for the next doubling iteration
        lprev = l
        print("Current Gamma:", gammacur, "for l =", l)
        l *= 2

    lmax = 1 / gammacur
    u = 0.0
    for i in range(classes):
        u += lmax * probs[i] * taus[i] * requests[i]
    u /= s

    return lmax, lideal, u
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:22:42 2025

@author: dile
"""

import csv
import numpy as np
from modsgen import distros
from numpy.random import SeedSequence, default_rng

# if cupy is not available, fall back to numpy and use cp as the alias
# to allow the code to run on systems with or without a compatible GPU.
try:
    import cupy as cp
except ImportError:
    import numpy as cp


def one_sampling(B, L, N, W, sigma, tau, alpha, wasted):
    """
    Performs one step of the backward simulation for B tries simultaneously.

    Params:
        B (int): Batch size (current_Ntries)
        L (int): Sequence length (chunk_size)
        N (int): Number of servers considered in the system
        W (cp.ndarray): Current workload, shape (B, N) 
        sigma (cp.ndarray): Service times, shape (B, L).
        tau (cp.ndarray): Interarrival times, shape (B, L).
        alpha (cp.ndarray): Class sizes, (B, L).
        wasted (cp.ndarray): recrds wasted servers, shape (B).

    Returns:
        Tuple[cp.ndarray, cp.ndarray]: Updated workload and wasted arrays.
    """

    # creates erver indices with shape (1, N, 1) which then allows 
    # broadcasting against (B, N, L) arrays
    indices = cp.arange(N, dtype=cp.float32).reshape(1, N, 1)
    
    # expand dimensions of time-dependent inputs to (B, 1; L)
    # which prepares them to then be broadcasted over the N servers
    alpha_exp = alpha[:, cp.newaxis, :]
    sigma_exp = sigma[:, cp.newaxis, :]
    tau_exp = tau[:, cp.newaxis, :]

    # compute helper matrices M, A, M2 with explicit batch dimension,
    # and shape (B, N, L)
    # M_new -> mask is 1 if server i is affected by an arrival
    # (i.e. i < alpha), 0 otherwise.
    M_new = (indices < alpha_exp).astype(cp.float32)
    # M2_new -> mask is 1 if server i is the last server affected
    # (i.e. n == alpha - 1)
    M2_new = (indices == (alpha_exp - 1)).astype(cp.float32)
    # A_vals is the workload change matrix,
    # affected servers (M_new=1) get (sigma - tau)
    # unaffected servers (M_new=0) get (-tau)
    A_vals = M_new * (sigma_exp - tau_exp) + (1 - M_new) * (-tau_exp)
    
    # prepend a zero column for the t=0 step in the loop
    zeros_col = cp.zeros((B, N, 1), dtype=cp.float32)
    M = cp.concatenate((zeros_col, M_new), axis=2)      # shape: (B, N, L+1)
    A = cp.concatenate((zeros_col, A_vals), axis=2)     # shape: (B, N, L+1)
    M2 = cp.concatenate((zeros_col, M2_new), axis=2)    # shape: (B, N, L+1)

    # iterate through each time step (from 0 to L)
    for t in range(L + 1):
        M_t = M[:, :, t]    # shape: (B, N) - affected servers 
        A_t = A[:, :, t]    # shape: (B, N) - workload change
        M2_t = M2[:, :, t]  # shape: (B, N) - last affected servre

        # update wasted capacity (batched)
        sum_M_t = cp.sum(M_t, axis=1) # number of affected servers (per batch)
        sum_W_M2_t = cp.sum(W * M2_t, axis=1) # workload of last afftected server
        sum_W_M_t = cp.sum(W * M_t, axis=1) # total workload of all affected servers
        
        # apply specific wasted capacity expression
        wasted = wasted + (sum_M_t * sum_W_M2_t - sum_W_M_t)

        # find the max workload among the affected servers for each batch
        mx = cp.max(W * M_t, axis=1, keepdims=True) # shape: (B, 1)

        # update W for each try in the batch (apply backward simulation update)
        # i.e., i) unaffected servers (1-M_t) keep their workload, ii) affected 
        # servers (M_t) take the max workload (mx), and iii) apply the workload 
        # change A_t to all servers.
        W = W * (1 - M_t) + M_t * mx + A_t
        
        # workload cannot be negative so crop to 0
        W = cp.maximum(W, 0)
        
        # sort workloads as the algorithm assumes servers are ordered 
        # by workload (lowest to highest)
        W = cp.sort(W, axis=1)
        
    return W, wasted


def generate_chunk(Ntries, start_offset, chunk_size, ps, lambdas,
                   sizes, base_seeds, compute_sigmas, sigma_params,
                   num_sigma_streams=1):
    """
    Generates a chunk of random samples (sigmas, taus, alphas) for a batch.

    This function generates random numbers on the CPU for reproducible seeding
    and then converts them to the target distributions (sigma, tau, alpha)
    on the GPU (using cupy).
    
    Params:
        Ntries (int): Batch size (current_Ntries).
        start_offset (int): Offset used for seeding to ensure unique streams.
        chunk_size (int): Number of samples to generate per trial (L).
        ps (np.ndarray): Class probabilities.
        lambdas (np.ndarray): Arrival rate (for taus).
        sizes (cp.ndarray): Array of possible class sizes.
        base_seeds (np.ndarray): Base seeds for this batch, shape (Ntries,).
        compute_sigmas (callable): GPU-aware function to calculate service times.
        sigma_params (dict): Parameters for the `compute_sigmas` function.
        num_sigma_streams (int): Number of independent random streams needed
                                 by `compute_sigmas`.

    Returns:
        Tuple[cp.ndarray, cp.ndarray, cp.ndarray]: sigmas, taus, alphas
    """
    
    # initialize data strcutures
    rand_vals_np = np.empty((Ntries, chunk_size), dtype=np.float32)
    tau_rands_np = np.empty((Ntries, chunk_size), dtype=np.float32)
    
    # create a list to hold the numpy sigma arrays
    # (some distributions require more than one)
    sigma_rands_np_list = [
        np.empty((Ntries, chunk_size), dtype=np.float32) 
        for _ in range(num_sigma_streams)
    ]
    
    # generate random numbers on CPU, trial by trial, for robust seeding
    for i in range(Ntries):
        
        # create a unique seed sequence for this trial and time offset
        ss = SeedSequence(base_seeds[i] + start_offset)
        # spawn independent RNGs for class, tau, and each sigma stream
        child_rngs = ss.spawn(2 + num_sigma_streams)
        
        # generate uniform randoms [0,1)
        rng_class = default_rng(child_rngs[0])
        rng_tau   = default_rng(child_rngs[1])
        
        # class and tau random generation
        rand_vals_np[i] = rng_class.random(chunk_size)
        tau_rands_np[i] = rng_tau.random(chunk_size)

        # generate all required sigma random number arrays
        for j in range(num_sigma_streams):
            rng_sigma_j = default_rng(child_rngs[2 + j])
            sigma_rands_np_list[j][i] = rng_sigma_j.random(chunk_size)
            
    # transferring all data to cupy 
    rand_vals_cp = cp.asarray(rand_vals_np)
    tau_rands_cp = cp.asarray(tau_rands_np)
    sigma_rands_cp_list = [cp.asarray(arr) for arr in sigma_rands_np_list]

    # compute cumulative probabilities for class selection
    cum_ps = cp.asarray(ps).cumsum()
    
    # determine the class for each arrival 
    class_indices = cp.searchsorted(cum_ps, rand_vals_cp)
    
    # map class indices to their corresponding sizes
    alphas = cp.asarray(sizes)[class_indices]
    
    # generate exponential interarrival times
    taus = -cp.log(tau_rands_cp) / cp.asarray(lambdas)
    
    # call cupy-based function to compute sigmas (service times)
    sigmas = compute_sigmas(
        sigma_rands_cp_list,  # Pass the list of random arrays
        class_indices, 
        **sigma_params
    )
    
    return sigmas, taus, alphas


def perfect_sampling(chunk_size, ell, N, Ntries, sizes, ps, lambdas,
                     fname, compute_sigmas, sigma_params, default_seed=42,
                     batch_size=10000, n_streams = 1):
    
    """
    Performs the main perfect sampling simulation using a backward simulation approach.

    The simulation runs in batches (batch_size) for a total of Ntries.
    It uses a "doubling" scheme, starting from an initial simulation length `ell`
    and doubling it (ell, 2*ell, 4*ell, ...) until the workload (W) converges
    between two successive iterations (Wcurr == Wprev). This ensures the
    simulation has reached its stationary distribution (sub-perfect sampling).

    Results are written to a CSV file incrementally after each batch.

    Params:
        chunk_size (int): Length of sequence (L) processed in each `one_sampling` call.
        ell (int): Initial simulation length to check for convergence.
        N (int): Number of servers.
        Ntries (int): Total number of independent simulations (trials) to run.
        sizes (np.ndarray): Array of possible class sizes.
        ps (np.ndarray): Probabilities for each class.
        lambdas (np.ndarray): Arrival rate (for tau generation and final wasted calc).
        fname (str): Output CSV file name.
        compute_sigmas (callable): Function to calculate service times.
        sigma_params (dict): Parameters for the `compute_sigmas` function.
        default_seed (int): Base seed for the main RNG.
        batch_size (int): Number of trials to process in a single GPU batch.
        n_streams (int): Number of random streams needed by `compute_sigmas`.
    """
    
    # initialize the main ranfom number generator (RNG) on CPU
    parent_rng = np.random.default_rng(default_seed)
    
    # number of features to save: [ID, wait_time_1, ..., wait_time_K, wasted, ell]
    num_features = sizes.shape[0] + 3

    # move static simulation parameters to GPU
    arr_rate = cp.array(lambdas, dtype=cp.float32)    
    sizes_cp = cp.asarray(sizes)
    sigma_params_cp = {key: cp.asarray(val) for key, val in sigma_params.items()}
    
    # compute total number of batches needed
    num_batches = (Ntries + batch_size - 1) // batch_size
    
    # pre-generate unique base seeds for all trials (on CPU)
    base_seeds = parent_rng.integers(low=0, high=2**60, size=Ntries)


    for batch_index in range(num_batches):
        
        # determine start/end indices for the current batch
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, Ntries)
        current_Ntries = end_idx - start_idx
                
        # get the pre-generated seeds for this batch
        batch_base_seeds = base_seeds[start_idx:end_idx]
        
        # initialize workload arrays to zeros for this batch
        Wcurr = cp.zeros((current_Ntries, N), dtype=cp.float32)
        Wprev = cp.zeros((current_Ntries, N), dtype=cp.float32)

        
        current_ell = int(ell) # reset simulation length for each new batch
        first = True  # flag for the first iteration of the doubling loop
        
        # loop until workload converges (Wcurr == Wprev)
        while first or not cp.allclose(Wcurr, Wprev):
            
            # store the last iteration's workload for comparison
            Wprev = Wcurr.copy()
            # reset wasted capacity for this new (longer) simulation run
            Wcurr = cp.zeros((current_Ntries, N), dtype=cp.float32)
            wasted_curr = cp.zeros(current_Ntries, dtype=cp.float32)
            
            if current_ell > ell:
                first = False
            
            # divide the total simulation length into manageable 
            # chunks to avoid memory overflow
            num_chunks = current_ell // chunk_size
            
            for i in range(num_chunks):
                
                # compute the start_offset for seeding, ensuring each chunk or
                # doubling iteration gets a unique, non-overlapping set of random numbers.           
                start_offset = (2**32) - current_ell + chunk_size * i
                
                # generate the random variables for this chunk
                sigmas, taus, alphas = generate_chunk(current_Ntries, start_offset, chunk_size, 
                                                      ps, lambdas, sizes_cp, batch_base_seeds,
                                                      compute_sigmas, sigma_params_cp, n_streams)
                
                # run one step of the backward simulation for this chunk
                Wcurr, wasted_curr = one_sampling(current_Ntries, chunk_size, N, Wcurr, sigmas, taus, alphas, wasted_curr)
            
            # double the simulation length for the next iteration
            current_ell *= 2
        
        # current_ell was doubled one last time after convergence
        # so the actual simulation length that converged is half of that     
        final_ell = current_ell / 2
        
        # normalise the wasted capacity by the simulation length and arrival rate
        wasted_final = arr_rate * (wasted_curr / final_ell)

      
        # saving results 
        storage_tensor = cp.zeros((current_Ntries, num_features), dtype=cp.float32)
        
        # vectorized extraction of waiting times
        wait_times = Wcurr[:, sizes_cp - 1]
        
        # populate the storage tensor  
        storage_tensor[:, 0] = cp.arange(start_idx + 1, end_idx + 1)
        storage_tensor[:, 1:-2] = wait_times
        storage_tensor[:, -2] = wasted_final
        storage_tensor[:, -1] = final_ell

        storage_numpy = cp.asnumpy(storage_tensor)
        with open(fname, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(storage_numpy)
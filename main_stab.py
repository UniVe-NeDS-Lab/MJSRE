# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:59:25 2025

@author: dile
"""
import time
import numpy as np
from modsgen import distros
from modsgen import algostab as stab

try:
    import cupy as cp
    def to_numpy(x):
        """Converts cupy array to numpy, or leaves it."""
        return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x
except ImportError:
    import numpy as cp
    
    def to_numpy(x):
        """Arrays are already NumPy, so just return them."""
        return x


# --- Main Execution ---

start_time = time.time()

# --- DEFINE YOUR PARAMETERS HERE ---
fname = 'synth'
distro = 'exp'  # 'exp', 'bpar', 'hyperexp', or 'erlk'
initl = 500
perc = 0.1
s = 1024
# -----------------------------------

input_file_name = f'{fname}_{s}_{distro}'
print(f"Starting simulation for '{distro}' from file: {input_file_name}")

# 1. --- LOAD AND CONFIGURE ---
if distro == 'exp':
    # Load data
    parser_func = distros.exp_parser
    params_dict_cp, classes = distros.load_params(input_file_name, parser_func)
    params_dict_np = {key: to_numpy(val) for key, val in params_dict_cp.items()}
    requests = params_dict_np.pop('sizes')
    probs = params_dict_np.pop('probs')
    
    # Configure Numba params
    print("Configuring for Exponential...")
    sampler = stab.sample_exp
    mu_vals = params_dict_np['mus']
    sigma_params_list = [(mu,) for mu in mu_vals]
    taus = 1.0 / mu_vals  # Mean = 1 / rate

elif distro == 'bpar':
    # Load data
    parser_func = distros.bpar_parser
    params_dict_cp, classes = distros.load_params(input_file_name, parser_func)
    params_dict_np = {key: to_numpy(val) for key, val in params_dict_cp.items()}
    requests = params_dict_np.pop('sizes')
    probs = params_dict_np.pop('probs')

    # Configure Numba params
    print("Configuring for B-Pareto...")
    sampler = stab.sample_bpar
    xmins = params_dict_np['xmins']
    xmaxs = params_dict_np['xmaxs']
    shapes = params_dict_np['shapes']
    sigma_params_list = list(zip(xmins, xmaxs, shapes))
    
    # Calculate B-Pareto means
    taus = np.zeros_like(shapes)
    for i in range(len(shapes)):
        k, m, a = xmins[i], xmaxs[i], shapes[i]
        if a == 1:
            if m == k: taus[i] = k
            else: taus[i] = (np.log(m) - np.log(k)) / (k**-1 - m**-1)
        else:
            num = a * (k**(1-a) - m**(1-a))
            den = (1-a) * (k**-a - m**-a)
            if den == 0: taus[i] = k
            else: taus[i] = num / den

elif distro == 'hyperexp':
    # Load data
    parser_func = distros.hyperexp_parser
    params_dict_cp, classes = distros.load_params(input_file_name, parser_func)
    params_dict_np = {key: to_numpy(val) for key, val in params_dict_cp.items()}
    requests = params_dict_np.pop('sizes')
    probs = params_dict_np.pop('probs')

    # Configure Numba params
    print("Configuring for Hyperexponential (2-phase)...")
    sampler = stab.sample_hyperexp
    r1 = params_dict_np['rates_ph1']
    r2 = params_dict_np['rates_ph2']
    p1 = params_dict_np['pphases']
    sigma_params_list = list(zip(r1, r2, p1))
    taus = p1 * (1.0 / r1) + (1.0 - p1) * (1.0 / r2)  # Mean

elif distro == 'erlk':
    # Load data
    parser_func = distros.erlk_parser
    params_dict_cp, classes = distros.load_params(input_file_name, parser_func)
    params_dict_np = {key: to_numpy(val) for key, val in params_dict_cp.items()}
    requests = params_dict_np.pop('sizes')
    probs = params_dict_np.pop('probs')

    # Configure Numba params
    print("Configuring for Erlang-k...")
    sampler = stab.sample_erlk
    rates = params_dict_np['rates']
    ks = params_dict_np['ks']
    sigma_params_list = list(zip(rates, ks))
    taus = 1.0 / rates  # Mean = 1 / rate

else:
    raise NotImplementedError(f"Distro '{distro}' not implemented yet")

# --- Check Taus ---
if np.isinf(taus).any():
    print("ERROR: Mean (tau) is infinite for at least one class. Stopping.")
    exit()

# 2. --- RUN GENERALIZED SIMULATION ---
print("Running generalized Numba simulation...")
lmax, ltop, u = stab.lmax_general(
    s=s, 
    classes=classes, 
    sampler=sampler,           # The @njit sampler function
    sigma_params_list=sigma_params_list, # The list of param tuples
    taus=taus,                 # The calculated mean service times
    requests=requests, 
    probs=probs, 
    perc=perc, 
    initl=initl
)
    
end_time = time.time()

# 3. --- PRINT RESULTS ---
print("\n___________________________________________")
print(f"Ideal arrival rate: {ltop:.6f}")
print(f"Maximum arrival rate: {lmax:.6f}")
print(f"Maximum utilization: {u:.6f}")

loads = np.array([0.75])
arrivals = [round(v, 4) for v in loads * lmax]
print("lambdas=({})".format(" ".join(map(str, arrivals))))
print(f"Execution time: {end_time - start_time:.4f} seconds")
print("___________________________________________\n")
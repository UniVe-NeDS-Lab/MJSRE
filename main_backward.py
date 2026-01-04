# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:51:04 2025

@author: dile
"""

from modsgen import perfsampl_backward as pesa  # <-- Use the CuPy-based version
from modsgen import distros
import cupy as cp
import argparse
import csv

parser = argparse.ArgumentParser(description="Perfect Sampling")

parser.add_argument("-dist", type=str, required=True, choices=['exp', 'bpar', 'erlk', 'hyperexp'], 
                    help="Distribution type: 'exp' (Exponential) or 'bpar' (Bounded Pareto)")

parser.add_argument("-device", type=str)
parser.add_argument("-in_fname", type=str)
parser.add_argument("-N", type=int)            # number of servers
parser.add_argument("-lam", type=float)        # arrival rate
parser.add_argument("-nsamples", type=int)
parser.add_argument("-L", type=int)
parser.add_argument("-split_size", type=int)   # Splitting L to avoid GPU OOM
parser.add_argument("-batch_size", type=int)   # Splitting Nsamples to avoid CPU OOM
parser.add_argument("-seed", type=int)

args = parser.parse_args()


# --- DISTRO SELECTION ---
# Based on the -dist argument, select the correct functions and labels
if args.dist == 'exp':
    parser = distros.exp_parser  # Assumes this name from our prev. refactor
    in_data, nClasses = distros.load_params(args.in_fname, parser)
    num_streams = 1
    compute_sigmas_foo = distros.compute_sigmas_exp
    distro_label = 'exp'
elif args.dist == 'bpar':
    parser = distros.bpar_parser # Assumes this name from our prev. refactor
    in_data, nClasses = distros.load_params(args.in_fname, parser)
    num_streams = 1
    compute_sigmas_foo = distros.compute_sigmas_bpar
    distro_label = 'bpar'
elif args.dist == 'hyperexp':
    parser = distros.hyperexp_parser # Assumes this name from our prev. refactor
    in_data, nClasses = distros.load_params(args.in_fname, parser)
    num_streams = 2
    compute_sigmas_foo = distros.compute_sigmas_hyperexp
    distro_label = 'hyperexp'
elif args.dist == 'erlk':
    parser = distros.erlk_parser # Assumes this name from our prev. refactor
    in_data, nClasses = distros.load_params(args.in_fname, parser)
    num_streams = int(cp.max(in_data['ks']))
    compute_sigmas_foo = distros.compute_sigmas_erlk
    distro_label = 'erlk'
else:
    # This case should be unreachable due to 'choices' in argparse
    print(f"Error: Unknown distribution '{args.dist}'")



# --- SETUP ---

# Select GPU device for CuPy explicitly
dev_id = int(args.device.split(":")[1]) if "cuda" in args.device else 0
cp.cuda.Device(dev_id).use()

# Load parameters 


sizes = in_data.pop('sizes') 
ps = in_data.pop('probs') 
distro_params = in_data

# Convert simulation parameters to CuPy arrays
N        = int(args.N)
lambdas  = float(args.lam)
Ntries   = int(args.nsamples)
initl    = int(args.L)
split_size = int(args.split_size)
batch_size = int(args.batch_size)
seed     = args.seed

# Prepare output CSV
Tstrings = [f'T{sizes[i]}' for i in range(len(sizes))]
out_fname = f'results/msjre_backward_{distro_label}_N{N}_Classes{nClasses}_lam{args.lam:.4f}_L{initl}_Nsamples{Ntries}.csv'
with open(out_fname, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Samples'] + Tstrings + ['Wasted Servers', 'L Max'])
    
    
print(f"Starting simulation for '{distro_label}' distribution.")
print(f"Output will be saved to: {out_fname}")
print(f"Lambda: {lambdas}")
# 6. --- RUN GENERALIZED SIMULATION ---
# Call the one, unified function with the selected strategies
pesa.perfect_sampling(
    chunk_size=split_size, 
    ell=initl, 
    N=N, 
    Ntries=Ntries, 
    sizes=sizes, 
    ps=ps, 
    lambdas=lambdas, 
    fname=out_fname,
    compute_sigmas = compute_sigmas_foo, 
    sigma_params = distro_params,
    default_seed=seed, 
    batch_size=batch_size,
    n_streams=num_streams
)

print("Simulation complete.")    


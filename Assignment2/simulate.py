#%%
import numpy as np
import simpy
import matplotlib.pyplot as plt
from util import save_data
from system import *
from simulation_functions import run_iteration, run_prio_iteration, run_simulation, simulate_different_n_jobs, simulate_different_servers

# pyright: reportPrivateImportUsage=false

# %%
run_iteration(lmd=10, mu=1, n_servers=1, n_jobs=200 ,debug=2)
# %%
run_simulation(rho=10,mu=1, n_servers=1, n_jobs=100, n_sims=100, debug=0)
# %%
simulate_different_servers([1,2,4],mu=10, rho=0.9999, n_jobs=5000, n_sims=200, debug=0)
# %%
n_jobs_list = np.arange(100,400,100)
results = simulate_different_n_jobs(n_jobs_list, 1, mu=10, rho=0.999, n_sims=100, debug=1)

# TODO: Ask how to statistically prove that the system has converged

# FLoor: first arbitrarily small threshold
# and then once you reach
# %%

# %%
run_prio_iteration(lmd=10,mu=1,n_servers=1,n_jobs=200, debug=2)

#%%
run_iteration(lmd=10, mu=1,n_servers=1,n_jobs=200,debug=2)

#%% Plot distributions of watining times  for 4 values of rho
n_sims = 200
n_jobs = 2000
mu = 0.95
rho_list = [0.8, 0.85, 0.90, 0.95]
FIFOdatalist = []

for load in rho_list:
    FIFOdatalist.append(run_simulation(load,mu, n_servers=1, n_jobs=n_jobs,n_sims=n_sims,debug=1, iteration_function=run_iteration, return_data=True))

#%%
SJFdatalist = []
for load in rho_list:
    SJFdatalist.append(run_simulation(load, mu, n_servers=1, n_jobs=n_jobs, n_sims=n_sims, debug=1, iteration_function=run_prio_iteration,return_data=True))

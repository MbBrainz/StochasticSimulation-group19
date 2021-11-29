#%%
import numpy as np
import simpy
import matplotlib.pyplot as plt
from system_simulation import *
# pyright: reportPrivateImportUsage=false


# %%
run_iteration(lmd=10, mu=1, n_servers=1, n_jobs=200 ,debug=1)
# %%
run_simulation(lmd=10,mu=1, n_servers=1, n_jobs=100, n_sims=100, debug=0)
# %%
simulate_different_servers([1,2],mu=10, rho=0.9999, n_jobs=1000, n_sims=100, debug=0)
# %%
n_jobs_list = np.arange(100,3000,100)
results = simulate_different_n_jobs(n_jobs_list, 1, mu=10, rho=0.999, n_sims=100, debug=1)

y_val = [x[0] for x in results]
std_val = [x[1] for x in results]

#%%
import matplotlib.pyplot as plt

plt.style.use("seaborn")
fig, ax = plt.subplots(1, 1, )
ax.plot(
    n_jobs_list,
    y_val,
    label='average waiting time',
    color='r',
    linestyle='-',
    marker='.',
)
ax.set(
    xlabel='n_jobs',
    ylabel='waiting time',
    title='Average wating time per n_jobs'
)

# TODO: Ask how to statistically prove that the system has converged

# FLoor: first arbitrarily small threshold
# and then once you reach
# %%

def run_iteration_diff(lmd, mu, n_servers, n_jobs, debug=0):
    system = System(simpy.Environment(), n_servers, mu)
    system.env.process(job_source(system, lmd, n_jobs, debug))
    system.env.run()
    mean_i = np.mean(system.wait_times)
    if debug == 2: print(f'Mean waiting time: {mean_i}')

    return system
# %%
run_prio_iteration(lmd=10,mu=1,n_servers=2,n_jobs=1000)
# %%

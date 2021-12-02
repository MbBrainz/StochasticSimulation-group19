import numpy as np
from numpy.core.fromnumeric import mean
from util import confidence_interval
from system import System, PrioSystem
import simpy

# pyright: reportPrivateImportUsage=false

#%%
def run_iteration(lmd, mu, n_servers, n_jobs, debug=0):
    system = System(simpy.Environment(), n_servers, mu, debug=debug)
    mean_i = system.run(lmd,n_jobs)
    if debug == 2: print(f'Mean waiting time: {mean_i}')
    return system, mean_i

def run_prio_iteration(lmd, mu, n_servers, n_jobs, debug=0):
    system = PrioSystem(simpy.Environment(), n_servers, mu, debug=debug)
    mean_i = system.run(lmd, n_jobs)
    if debug == 2: print(f'Mean waiting time~: {mean_i:.4f}')
    return system, mean_i

def run_simulation(rho, mu, n_servers, n_jobs, n_sims ,debug=0, iteration_function=run_iteration, return_data=False):
    mean_list = []
    lmd = mu / (rho * n_servers)

    for i in range(n_sims):
        system, mean_i = iteration_function(lmd, mu, n_servers, n_jobs, debug)
        mean_list.append(mean_i)

    mean_array = np.array(mean_list)
    mean_value = np.mean(mean_array)
    std_value =  np.std(mean_array, ddof=1)
    c_level = 0.95
    ci = confidence_interval(mean_value, std_value, n_sims, level=c_level)

    if debug==1: print(f'For values lamda({lmd:.2f}), mu({mu:.2f}), n_servers({n_servers:.0f}), njobs({n_jobs:.0f}), nsims({n_sims:.0f}) found a confidence interval of \n\
          {c_level*100:.0f}%: [{ci[0]:.3f}, {ci[1]:.3f}]')
    if return_data == False: mean_list = 0
    return mean_value, std_value, ci, mean_list

def simulate_different_servers(nserver_list, mu, rho, n_jobs, n_sims, debug=0):
    for servers in nserver_list:
        # M/M/n queue and a system load ρ and processor capacity μ than for a single M/M/1 queue with the same load characteristics (and thus an n-fold lower arrival rate).
        lmd_server = mu / (rho * servers)
        # mu_server = rho/lmd
        if debug==1: print(f'for {servers} servers with lamda: {lmd_server:.2f}')
        # print(f'Simulate for # servers: {servers}')
        run_simulation(rho, mu, servers, n_jobs, n_sims, debug=debug)
        print("\n")

def simulate_different_n_jobs(n_jobs_list, n_servers, mu, rho, n_sims, debug):

    results = []
    for n_jobs in n_jobs_list:
        print(n_jobs)
        # lmd_server = mu / (rho * n_servers)
        if debug==1: print(f'start simulation for {n_jobs} jobs...')
        mean_i, std_i, ci_i, _ = run_simulation(rho, mu, n_servers,n_jobs, n_sims, debug=debug)
        results.append((mean_i, std_i, ci_i))

    return results
        #different n server need to be checked as wel for convergence of jnovbs
# mu = 1      # μ – the capacity of each of n equal servers.
# %%

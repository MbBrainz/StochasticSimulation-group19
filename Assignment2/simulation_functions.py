import numpy as np
import pandas as pd
import simpy
from numpy.core.fromnumeric import mean

from system import PrioSystem, System
from util import confidence_interval, plot_sim_data


# pyright: reportPrivateImportUsage=false
#%%
def FIFO_iteration(rho, mu, n_servers, n_jobs, service_dist="M",debug=0) -> tuple[System,float]:
    system = System(simpy.Environment(), n_servers, mu, service_dist=service_dist, debug=debug)
    lmd = get_lambda(rho, mu, n_servers)
    mean_i = system.run(lmd, n_jobs)
    if debug == 2: print(f'Mean waiting time: {mean_i}')
    return system, mean_i

def get_lambda(rho, mu, n_servers): return rho * mu * n_servers

def SJF_iteration(rho, mu, n_servers, n_jobs, service_dist="M", debug=0):
    if service_dist != "M": print(f"priority system has no implementation for service dist type '{service_dist}'. Falls back to 'M' ")
    system = PrioSystem(simpy.Environment(), n_servers, mu, debug=debug)
    mean_i = system.run(get_lambda(rho, mu, n_servers), n_jobs)
    if debug == 2: print(f'Mean waiting time~: {mean_i:.4f}')
    return system, mean_i

def run_simulation(rho, mu, n_servers, n_jobs, n_sims, service_dist="M",debug=0, iteration_function=FIFO_iteration, return_data=False):
    mean_list = run_sim(rho, mu, n_servers, n_jobs, n_sims, service_dist=service_dist,debug=debug, iteration_function=iteration_function)
    mean_array = np.array(mean_list)
    mean_value = np.mean(mean_array)
    std_value =  np.std(mean_array, ddof=1)
    c_level = 0.95
    ci = confidence_interval(mean_value, std_value, n_sims, level=c_level)
    if debug==1: print(f'For service {service_dist} with values rho({rho:.2f}), mu({mu:.2f}), n_servers({n_servers:.0f}), njobs({n_jobs:.0f}), nsims({n_sims:.0f}) found a confidence interval of \n\
          {c_level*100:.0f}%: [{ci[0]:.3f}, {ci[1]:.3f}]')
    if return_data == False: mean_list = 0
    return mean_value, std_value, ci, mean_list

def plot_iteration(ax,lmd, mu, n_servers, n_jobs, service_dist="M" ,debug=0, iteration_function=FIFO_iteration):
    system, mean = FIFO_iteration(lmd ,mu,n_servers, n_jobs,service_dist=service_dist, debug=debug)
    plot_sim_data(ax, system.wait_times, service_dist)

def plot_simulation(ax, rho, mu, n_servers, n_jobs, n_sims, service_dist="M" ,debug=0, iteration_function=FIFO_iteration, return_data=False):
    mean, std, ci, data = run_simulation(rho, mu, n_servers, n_jobs, n_sims, service_dist=service_dist, debug=debug, iteration_function=iteration_function, return_data=True)
    plot_sim_data(ax, data, service_dist)


def simulate_different_servers(nserver_list, mu, rho, n_jobs, n_sims, debug=0):
    for n_servers in nserver_list:
        # M/M/n queue and a system load ρ and processor capacity μ than for a single M/M/1 queue with the same load characteristics (and thus an n-fold lower arrival rate).
        lmd_server = get_lambda(rho,mu,n_servers)
        # mu_server = rho/lmd
        if debug==1: print(f'for {n_servers} servers with lamda: {lmd_server:.2f}')
        # print(f'Simulate for # servers: {servers}')
        run_simulation(rho, mu, n_servers, n_jobs, n_sims, debug=debug)
        print("\n")


def simulate_different_n_jobs(n_jobs_list, n_servers, mu, rho, n_sims, debug):
    results = []
    for n_jobs in n_jobs_list:
        print(n_jobs)
        if debug==1: print(f'start simulation for {n_jobs} jobs...')
        mean_i, std_i, ci_i, _ = run_simulation(rho, mu, n_servers, n_jobs, n_sims, debug=debug)
        results.append((mean_i, std_i, ci_i))

    return results
        #different n server need to be checked as wel for convergence of jnovbs
# mu = 1      # μ – the capacity of each of n equal servers.
# %%
def run_sim(rho, mu, n_servers, n_jobs, n_sims, service_dist="M",debug=0, iteration_function=FIFO_iteration):
    mean_wait_list = []

    for i in range(n_sims):
        system, mean_i = iteration_function(rho, mu, n_servers, n_jobs,service_dist, debug)
        mean_wait_list.append(np.mean(system.wait_times))

    calculate_stats(mean_wait_list,rho,mu,service_dist,n_servers,n_jobs,debug)
    return mean_wait_list

def run_sim_as_df(rho, mu, n_servers, n_jobs, n_sims, service_dist="M",debug=0, iteration_function=FIFO_iteration) -> pd.DataFrame:
    mean_wait_list = run_sim(rho, mu, n_servers, n_jobs, n_sims, service_dist=service_dist,debug=debug, iteration_function=FIFO_iteration)
    df_list = [(mean_wait_list[i], service_dist, str(n_servers), str(rho), str(mu), str(iteration_function.__name__).removesuffix("_iteration"))
          for i in range(len(mean_wait_list))]
    df = pd.DataFrame(df_list, columns=get_dataframe_columns(),)
    if debug > 1: print(df.head())
    return df

def get_dataframe_columns():
    return ["mean waiting time","service", "n servers", "rho", "mu", "treatment"]
# %%
def calculate_stats(mean_list,rho,mu,service_dist,n_servers, n_jobs, debug=0):
    mean_array = np.array(mean_list)
    n_sims = len(mean_array)
    mean_value = np.mean(mean_array)
    std_value =  np.std(mean_array, ddof=1)
    c_level = 0.95
    ci = confidence_interval(mean_value, std_value, n_sims, level=c_level)
    if debug>0: print(f'For service {service_dist} with values rho({rho:.2f}), mu({mu:.2f}), n_servers({n_servers:.0f}), njobs({n_jobs:.0f}), nsims({n_sims:.0f}) found a confidence interval of \n\
          {c_level*100:.0f}%: [{ci[0]:.3f}, {ci[1]:.3f}]')

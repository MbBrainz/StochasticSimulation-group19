"""
Bank renege example

Covers:

- Resources: Resource
- Condition events

Scenario:
  A counter with a random service time and customers who renege. Based on the
  program bank08.py from TheBank tutorial of SimPy 2. (KGM)

"""
#%%
import random
import numpy as np
import simpy
from scipy import stats

# INCORRECT ERROR SUPPRESSION:
# pyright: reportPrivateImportUsage=false

# λ – the arrival rate into the system as a whole.
# μ – the capacity of each of n equal servers.
# ρ represents the system load. In a single server system, it will be: ρ=λ/μ
# In a multi-server system (one queue with n equal servers, each with capacity μ), it will be ρ=λ/(nμ).

class System(object):
    def __init__(self, env: simpy.Environment, n_servers, mu) -> None:
        self.env = env
        self.mu = mu
        self.server = simpy.Resource(env,capacity=n_servers)
        self.wait_times = []

    def get_job_time(self):
        return np.random.exponential(scale=1/self.mu)



wait_times = []

def job_source(system:System, lmd, n_jobs, debug=0):
    if debug>0: print(f"Job source setup with {n_jobs} jobs with arrival Rate = {lmd} ")
    for i in range(n_jobs):
        inter_arrival = np.random.exponential(scale=1/lmd)
        yield system.env.timeout(inter_arrival)

        system.env.process(job(system, i,debug=debug))

def job(system:System, id, debug=0):
    arrive = system.env.now
    if debug == 2: print(f'[{arrive}] Job{id} arrives')
    with system.server.request() as req:
        yield req
        yield system.env.timeout(system.get_job_time())
    wait = system.env.now - arrive
    if debug == 2: print(f'job finished with id {id} after {wait:.2f}')
    system.wait_times.append(wait)

#%% statistics functions
def confidence_interval(mean_value, std_value, N, level):
    ci = stats.norm.interval( # type: ignore
        level,
        loc=mean_value,
        scale=std_value/np.sqrt(N))
    return ci


#%%
def run_iteration(lmd, mu, n_servers, n_jobs, debug=0):
    system = System(simpy.Environment(), n_servers, mu)
    system.env.process(job_source(system, lmd, n_jobs, debug))
    system.env.run()
    mean_i = np.mean(system.wait_times)
    if debug == 1: print(f'Mean waiting time: {mean_i}')
    return system.wait_times

def run_simulation(lmd, mu, n_servers, n_jobs, n_sims ,debug=0):
    mean_list = []
    for i in range(n_sims):
        wait_times = run_iteration(lmd,  mu, n_servers, n_jobs, debug)
        mean_list.append(np.mean(wait_times))

    mean_array = np.array(mean_list)

    mean_value = np.mean(mean_array)
    std_value =  np.std(mean_array, ddof=1)
    c_level = 0.95
    ci = confidence_interval(mean_value, std_value, n_sims, level=c_level)

    print(f'For values lamda({lmd:.2f}), mu({mu:.2f}), n_servers({n_servers:.0f}), njobs({n_jobs:.0f}), nsims({n_sims:.0f}) found a confidence interval of \n\
          {c_level*100:.0f}%: [{ci[0]:.3f}, {ci[1]:.3f}]')

def simulate_different_servers(nserver_list, mu, rho, n_jobs, n_sims, debug=0):
    for servers in nserver_list:

        # M/M/n queue and a system load ρ and processor capacity μ than for a single M/M/1 queue with the same load characteristics (and thus an n-fold lower arrival rate).
        lmd_server = mu / (rho * servers)
        # mu_server = rho/lmd
        if debug==0: print(f'for {servers} servers with lamda: {lmd_server:.2f}')
        # print(f'Simulate for # servers: {servers}')
        run_simulation(lmd_server, mu, servers, n_jobs, n_sims, debug=debug)
        print("\n")

lmd = 100     # λ – the arrival rate into the system as a whole.
# mu = 1      # μ – the capacity of each of n equal servers.
# # %%
# run_iteration(lmd, mu=1, n_servers=1, n_jobs=200 ,debug=1)
# # %%
# run_simulation(lmd,mu=1, n_servers=1, n_jobs=100, n_sims=100, debug=0)
# %%

simulate_different_servers([1,2],mu=10, rho=0.9999, n_jobs=10000, n_sims=100, debug=0)
# %%

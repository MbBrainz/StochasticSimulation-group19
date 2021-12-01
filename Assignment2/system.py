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
from os import system, wait
import random
import numpy as np
import simpy
from scipy import stats
from simpy.resources.resource import PriorityResource, Request
from simpy.resources.store import Store, PriorityItem, PriorityStore

from util import confidence_interval

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

class PrioSystem(object):
    def __init__(self, env: simpy.Environment, n_servers, mu, debug) -> None:
        store = PriorityStore(env)
        self.env = env
        self.mu = mu
        self.server = PriorityResource(env,capacity=n_servers,)
        self.wait_times = []
        self.debug = debug
        self.service_dist = "M"

    def priority_job(self, id):
        arrive = self.env.now

        priority = self.get_prio_for_rnddistr()
        if self.debug >= 3: print(f'[{arrive}] Job{id} arrives with prio {priority[0]} and waiting time {priority[1]}')

        with self.server.request(priority=priority[0], preempt=True) as req:
            yield req
            yield self.env.timeout(priority[1])

        wait = self.env.now - arrive
        if self.debug >= 3: print(f'job finished with id {id} after {wait:.2f}')
        self.wait_times.append(wait)

    def get_prio_for_rnddistr(self):
        rand = random.uniform(0,1)
        prio = np.round(100*rand)
        exp_rand = (-1 / self.mu) * np.log(1-rand)
        return prio, exp_rand

    def get_job_time(self):
        if self.service_dist == "M":
            return np.random.exponential(scale=1/self.mu)

        if self.service_dist == "D":
            return self.mu

        if self.service_dist == "H":
            mu = random.choices([self.mu ,5], weights=[0.75,0.25], k=1)
            return np.random.exponential(scale=1/mu[0])
        # TODO: Test functionality
        # TODO: Implement prio cue


    def job_source(self, lmd, n_jobs):
        if self.debug >= 3: print(f'Job source starts with {n_jobs} prioJobs and lambda: {lmd}')
        for i in range(n_jobs):
            inter_arrival = np.random.exponential(scale=1/lmd)
            yield self.env.timeout(inter_arrival)

            self.env.process(self.priority_job(i))

    def run(self, lmd, n_jobs):
        self.env.process(self.job_source(lmd,n_jobs))
        self.env.run()
        mean_i = np.mean(self.wait_times)
        return mean_i

def job_source(system:System, lmd, n_jobs, debug=0):
    if debug > 2: print(f"Job source setup with {n_jobs} jobs with arrival Rate = {lmd} ")
    if debug > 2: print(f"Job source setup with {n_jobs} jobs with arrival Rate = {lmd} ")
    for i in range(n_jobs):
        inter_arrival = np.random.exponential(scale=1/lmd)
        yield system.env.timeout(inter_arrival)

        system.env.process(job(system, i,debug=debug))

def job(system:System, id, debug=0):
    arrive = system.env.now
    if debug == 3: print(f'[{arrive}] Job{id} arrives')
    with system.server.request() as req:
        yield req
        yield system.env.timeout(system.get_job_time())
    wait = system.env.now - arrive
    if debug == 3: print(f'job finished with id {id} after {wait:.2f}')
    system.wait_times.append(wait)

# %%

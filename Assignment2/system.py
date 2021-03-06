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
    def __init__(self, env: simpy.Environment, n_servers, mu, service_dist="M", debug=0) -> None:
        self.env = env
        self.mu = mu
        self.server = simpy.Resource(env,capacity=n_servers)
        self.wait_times = []
        self.in_system_times = []
        self.debug = debug
        self.service_dist = service_dist

    def job(self, id):
        arrival_time = self.env.now
        if self.debug == 3: print(f'[{arrival_time}] Job{id} arrives with quelength {len(self.server.queue)}')

        with self.server.request() as req:
            yield req

            wait = self.env.now - arrival_time
            self.wait_times.append(wait)

            yield self.env.timeout(self.get_service_time())

        in_system_time = self.env.now - arrival_time
        self.in_system_times.append(in_system_time)

        if self.debug == 3: print(f'job finished with id {id} after {wait:.2f}')

    def job_source(self, lmd, n_jobs):
        if self.debug > 2: print(f"Job source setup with {n_jobs} jobs with arrival Rate = {lmd} ")
        for i in range(n_jobs):
            inter_arrival = self.get_arrival_time(lmd)
            yield self.env.timeout(inter_arrival)

            self.env.process(self.job(i))

    def run(self, lmd, n_jobs):
        self.env.process(self.job_source(lmd,n_jobs))
        self.env.run()
        mean_i = np.mean(self.in_system_times)
        return mean_i

    # random job arrival rate and service time getters
    def get_service_time(self):

        if self.service_dist == "M":
            return np.random.exponential(scale=1/self.mu)

        elif self.service_dist == "D":
            return 1/self.mu

        elif self.service_dist == "H":
            c=0.2; p1=0.75; p2=1-p1
            mu1 = self.mu*(1-c) / (1-c-p2) * p1
            mu2 = self.mu*(1+c) / (1+c-p1) * p2
            mu = random.choices([mu1,mu2], weights=[p1,p2], k=1)
            return np.random.exponential(scale=1/mu[0])

        else:
            raise ValueError
        # return np.random.exponential(scale=1/self.mu)

    def get_arrival_time(self, lmd):
        return np.random.exponential(scale=1/lmd)


class PrioSystem(object):
    def __init__(self, env: simpy.Environment, n_servers, mu, debug=0) -> None:
        store = PriorityStore(env)
        self.env = env
        self.mu = mu
        self.server = PriorityResource(env,capacity=n_servers,)
        self.wait_times = []
        self.debug = debug
        self.service_dist = "M"

    def priority_job(self, id):
        arrive = self.env.now

        service_time = self.get_service_time()
        if self.debug >= 3: print(f'[{arrive}] Job{id} arrives with prio')

        with self.server.request(priority=int(service_time*1000000), preempt=True) as req:
            yield req
            yield self.env.timeout(service_time)

        wait = self.env.now - arrive
        if self.debug >= 3: print(f'job finished with id {id} after {wait:.2f}')
        self.wait_times.append(wait)

    def job_source(self, lmd, n_jobs):
        if self.debug >= 3: print(f'Job source starts with {n_jobs} prioJobs and lambda: {lmd}')
        for i in range(n_jobs):

            inter_arrival = self.get_arrival_time(lmd)
            yield self.env.timeout(inter_arrival)

            self.env.process(self.priority_job(i))

    def run(self, lmd, n_jobs):
        self.env.process(self.job_source(lmd,n_jobs))
        self.env.run()
        mean_i = np.mean(self.wait_times)
        return mean_i

    # random job arrival rate and service time getters
    def get_arrival_time(self, lmd):
        inter_arrival = np.random.exponential(scale=1/lmd)
        return inter_arrival

    def get_prio_for_rnddistr(self):
        rand = random.uniform(0,1)
        prio = np.round(100*rand)
        exp_rand = (-1 / self.mu) * np.log(1-rand)
        return prio, exp_rand

    # not used ATM. made for service distr. Not applocable yet to Prio Job
    def get_service_time(self):
        if self.service_dist == "M":
            return np.random.exponential(scale=1/self.mu)

        elif self.service_dist == "D":
            return 1/self.mu

        elif self.service_dist == "H":
            c=0.2; p1=0.75; p2=1-p1
            mu1 = self.mu*(1-c) / (1-c-p2) * p1
            mu2 = self.mu*(1+c) / (1+c-p1) * p2
            mu = random.choices([mu1,mu2], weights=[p1,p2], k=1)
            return np.random.exponential(scale=1/mu[0])

        else:
            raise ValueError
        # return np.random.exponential(scale=1/self.mu)


# %%

# %%

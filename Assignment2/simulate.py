#%%
import numpy as np
import simpy
import matplotlib.pyplot as plt
from util import save_data, plot_errorbar
from system import *
from simulation_functions import *

#%%
# pyright: reportPrivateImportUsage=false

# %%

rho=1
mu=0.95
n_servers=1
n_jobs = 3000

service_dist = ["Exponential", "Deterministic", "Hyperexponential"]
colors=['red','blue','green']
# lmd = mu / (rho * n_servers)
#%%
wait_times = []
system, data = FIFO_iteration(rho=rho, mu=mu, n_servers=1, n_jobs=n_jobs,service_dist="M", debug=2)
wait_times.append(system.wait_times)
system, data = FIFO_iteration(rho=rho, mu=mu, n_servers=1, n_jobs=n_jobs,service_dist="D", debug=2)
wait_times.append(system.wait_times)
system, data = FIFO_iteration(rho=rho, mu=mu, n_servers=1, n_jobs=n_jobs,service_dist="H", debug=2)
wait_times.append(system.wait_times)
#%%
fig, axes = plt.subplots(3,2, constrained_layout=True, sharex=True)
gs = axes[0, 0].get_gridspec()

# remove the underlying axes
data = wait_times[0]
bins = np.arange(min(data), max(data) + int(max(data)/40), int(max(data)/40))
for ax in axes[:, 0]:
    ax.remove()
axbig = fig.add_subplot(gs[:, 0])
for i in range(len(wait_times)):
    axbig.hist(wait_times[i],color=colors[i], bins=bins, alpha=0.7, label=service_dist[i])
    axes[i,1].plot(wait_times[i], colors[i], alpha=0.7, )
    # axes[i,1].set(cmap='tab10')
axbig.set(
    xlabel="wait time",
    ylabel="count",
          )
axes[1,1].set(ylabel="wait time")
axes[2,1].set(
              xlabel="job id")
axbig.legend()
plt.savefig("MDH_iter_hist_rho=1_mu=095_njobs=2000_ns=1.png", dpi=600)

#%%
rho=20
system, data = FIFO_iteration(rho=rho, mu=mu, n_servers=1, n_jobs=n_jobs,service_dist="H", debug=3)
plt.hist(system.wait_times, bins=50)
#%%
rho=0.90
mu=15
n_servers=1
fig, ax = plt.subplots(1, 1, )
plot_simulation(ax, rho=rho, mu=mu, n_servers=n_servers, n_jobs=2000, n_sims=200, service_dist="M", debug=1)
plot_simulation(ax, rho=rho, mu=mu, n_servers=n_servers, n_jobs=2000, n_sims=200, service_dist="D", debug=1)
plot_simulation(ax, rho=rho, mu=mu, n_servers=n_servers, n_jobs=2000, n_sims=200, service_dist="H", debug=1)
plt.title("single server")
plt.legend()
#%%
rho=0.90
mu=15
n_servers=2
fig, ax = plt.subplots(1, 1, )
plot_simulation(ax, rho=rho, mu=mu, n_servers=n_servers, n_jobs=2000, n_sims=100, service_dist="M", debug=1)
plot_simulation(ax, rho=rho, mu=mu, n_servers=n_servers, n_jobs=2000, n_sims=100, service_dist="D", debug=1)
plot_simulation(ax, rho=rho, mu=mu, n_servers=n_servers, n_jobs=2000, n_sims=100, service_dist="H", debug=1)
plt.title("multi server")
plt.legend()
# %%
run_simulation(rho=10,mu=1, n_servers=1, n_jobs=100, n_sims=100, debug=0)
# %%
simulate_different_servers([1,2,4],mu=10, rho=0.9999, n_jobs=5000, n_sims=200, debug=0)
# %%
n_jobs_list = np.arange(100,4000,300)
results = simulate_different_n_jobs(n_jobs_list, 1, mu=10, rho=0.999, n_sims=20, debug=1)
#%%
fig, ax = plt.subplots(1, 1, )
plot_errorbar(x_array=n_jobs_list,datalist=results, label="jobs", ax=ax)

# TODO: Ask how to statistically prove that the system has converged

# FLoor: first arbitrarily small threshold
# and then once you reachs
# %%


# %%

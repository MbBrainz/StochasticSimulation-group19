#%%
import numpy as np
import simpy
import matplotlib.pyplot as plt
from util import save_data, plot_errorbar, SEED
from system import *
from simulation_functions import *

#%%
# pyright: reportPrivateImportUsage=false

# %%

rho=1
mu=0.95
n_servers=1
n_jobs = 3000

service_dist = ["Markov", "Deterministic", "Hyper Exponential"]
colors=['red','blue','green']
# lmd = mu / (rho * n_servers)
#%%
np.random.seed(SEED)
wait_times = []
for service in service_dist:
    system, data = FIFO_iteration(rho=rho, mu=mu, n_servers=1, n_jobs=n_jobs,service_dist=service_dist[0], debug=2)
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
axbig.set(
    xlabel="wait time",
    ylabel="count",
          )
axes[1,1].set(ylabel="wait time")
axes[2,1].set(
              xlabel="job id")
axbig.legend()
plt.savefig("MDH_iter_hist_rho=1_mu=095_njobs=2000_ns=1.png", dpi=600)


#%% COMPARES DISTRIBUTION OF DMH SERVICE FOR 1,2 AND 4 SERVERS
import pandas as pd
n_sims = 100
n_jobs = 2000
mu = 0.95

services = ["Markov", "Deterministic", "Hyper Exponential"]
server_list = [1,2,4]
rho = 0.9
df = pd.DataFrame(columns=get_dataframe_columns())
np.random.seed(SEED)
for service_dist in services:
    for n_servers in server_list:
        df = df.append(
            run_sim_as_df(rho ,mu,
                          n_servers=n_servers, n_jobs=n_jobs,n_sims=n_sims,debug=1, iteration_function=FIFO_iteration,
                          service_dist=service_dist[0]),
            ignore_index=True
        )
df.head()

# %%
import seaborn as sns
fig, ax = plt.subplots(1, 3, figsize=(10,5))
order=["H", "M", "D"]
sns.violinplot(data=df[df["n servers"] == "1"], y="mean waiting time", x="n servers",hue="service",ax=ax[0],hue_order=order)
sns.violinplot(data=df[df["n servers"] == "2"], y="mean waiting time", x="n servers",hue="service",ax=ax[1],hue_order=order)
sns.violinplot(data=df[df["n servers"] == "4"], y="mean waiting time", x="n servers",hue="service",ax=ax[2],hue_order=order)
ax[0].get_legend().remove()
ax[0].set(xlabel="")
ax[1].get_legend().remove()
ax[1].set(ylabel="")
ax[2].set(ylabel="")
ax[2].set(xlabel="")
plt.tight_layout()

plt.savefig(f"MDH_sim_rho=09,mu=095_njobs=2000_ns={n_sims}.png", dpi=600)
# %%
#%% not in reprt
fig, ax = plt.subplots(1, 3, )

sns.boxplot(data=df[df["n servers"] == "1"], y="mean waiting time", x="n servers",hue="service",ax=ax[0])
sns.boxplot(data=df[df["n servers"] == "2"], y="mean waiting time", x="n servers",hue="service",ax=ax[1])
sns.boxplot(data=df[df["n servers"] == "4"], y="mean waiting time", x="n servers",hue="service",ax=ax[2])
ax[0].get_legend().remove()
ax[0].set(xlabel="")
ax[1].get_legend().remove()
ax[1].set(ylabel="")
ax[2].set(ylabel="")
ax[2].set(xlabel="")
plt.tight_layout()

plt.savefig(f"MDH_sim_boxen_rho=09,mu=095_njobs=2000_ns={n_sims}.png", dpi=600)
# %%

#%% COMPARES DISTRIBUTION OF DMH SERVICE FOR 1,2 AND 4 SERVERS FOR 0.7, 0.8, 0.9
import pandas as pd
n_sims = 10
n_jobs = 2000
mu = 0.95

services = ["Markov", "Deterministic", "Hyper Exponential"]
server_list = [1,2,4]
rho_list = [0.7, 0.8, 0.9]
np.random.seed(SEED)
df = pd.DataFrame(columns=get_dataframe_columns())
for rho in rho_list:
    for service_dist in services:
        for n_servers in server_list:
            df = df.append(
                run_sim_as_df(rho ,mu,
                            n_servers=n_servers, n_jobs=n_jobs,n_sims=n_sims,debug=1, iteration_function=FIFO_iteration,
                            service_dist=service_dist[0]),
                ignore_index=True
            )
df.head()

# %%
import seaborn as sns
fig, ax = plt.subplots(3, 3, figsize=(12,10))
order=["H", "M", "D"]
plt.style.use("default")
sns.set_theme(style="whitegrid")
for i, rho in enumerate(rho_list):
    sns.violinplot(data=df[(df["n servers"] == "1") & (df["rho"]==str(rho))], y="mean waiting time", x="n servers",hue="service",ax=ax[i][0],hue_order=order)
    sns.violinplot(data=df[(df["n servers"] == "2") & (df["rho"]==str(rho))], y="mean waiting time", x="n servers",hue="service",ax=ax[i][1],hue_order=order)
    sns.violinplot(data=df[(df["n servers"] == "4") & (df["rho"]==str(rho))], y="mean waiting time", x="n servers",hue="service",ax=ax[i][2],hue_order=order)
    ax[i][0].get_legend().remove()
    ax[i][0].set(xlabel="")
    ax[i][1].get_legend().remove()
    ax[i][1].set(ylabel="")
    ax[i][0].text(0.01, 0.96, r"$\rho=$"+str(rho), fontsize=14,fontweight="bold",
                  horizontalalignment='left', verticalalignment='top', transform=ax[i][0].transAxes)
    if i!=0:
        ax[i][2].get_legend().remove()
    if i!=2:
        ax[i][0].set(ylabel="")
        ax[i][1].set(xlabel="")
    ax[i][2].set(ylabel="")
    ax[i][2].set(xlabel="")
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.savefig(f"MDH_sim_rho=09,mu=095_njobs=2000_ns={n_sims}.png", dpi=600)
# %%





#%% not in reprt
fig, ax = plt.subplots(1, 3, )

sns.boxplot(data=df[df["n servers"] == "1"], y="mean waiting time", x="n servers",hue="service",ax=ax[0])
sns.boxplot(data=df[df["n servers"] == "2"], y="mean waiting time", x="n servers",hue="service",ax=ax[1])
sns.boxplot(data=df[df["n servers"] == "4"], y="mean waiting time", x="n servers",hue="service",ax=ax[2])
ax[0].get_legend().remove()
ax[0].set(xlabel="")
ax[1].get_legend().remove()
ax[1].set(ylabel="")
ax[2].set(ylabel="")
ax[2].set(xlabel="")
plt.tight_layout()
plt.savefig(f"MDH_sim_boxen_rho=09,mu=095_njobs=2000_ns={n_sims}.png", dpi=600)
# %%

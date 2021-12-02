#%%  run simulations for different loads
# compare load statistics
import numpy as np
from simulation_functions import run_simulation, run_prio_iteration, run_iteration
import matplotlib.pyplot as plt

#%% [markdown]
# ## Check the behaviour of FIFO and SJF
#

#%%
n_sims = 20
n_jobs = 400
mu = 0.95
rho_list = np.arange(0.8, 1.0, 0.02)
FIFOdatalist = []

for load in rho_list:
    FIFOdatalist.append(run_simulation(load,mu, n_servers=1, n_jobs=n_jobs,n_sims=n_sims,debug=1, iteration_function=run_iteration))

#%%
SJFdatalist = []
for load in rho_list:
    SJFdatalist.append(run_simulation(load, mu, n_servers=1, n_jobs=n_jobs, n_sims=n_sims, debug=1, iteration_function=run_prio_iteration))

#%%
from util import save_data
data_dict = {
    "parameters": {
        "simulations": n_sims,
        "jobs": n_jobs,
        "mu": mu,
        "system_type":"MM1",
        "desc": "Simulation with multiple values for the load(rho)"
    },
    "xdata": list(rho_list),
    "ydata": {
        "FIFO": FIFOdatalist,
        "SJF": SJFdatalist
    }
}

# save_data(data_dict, f"load_chr_FIFO_SJF_errorbar_drho=0.02_mu=095_njobs=2000-2")
# %%
plt.style.use('seaborn')
from util import plot_errorHue


fig, ax = plt.subplots(1, 1, )
plot_errorHue(rho_list, FIFOdatalist, "FIFO", ax)
plot_errorHue(rho_list, SJFdatalist, "SJF", ax)
ax.set(
    xlabel=r'load $\rho)',
    ylabel='average waiting time',
    title='load characeristics of service systems',
    xticks=rho_list
)

#%%



from util import plot_errorbar

fig, ax = plt.subplots(1, 1, )
plot_errorbar(rho_list, FIFOdatalist, "FIFO", ax)
plot_errorbar(rho_list, SJFdatalist, "SJF", ax)
ax.set(
    xlabel=r'$load (\rho)$',
    ylabel='average waiting time',
    title='load characeristics of service systems',
    xticks=rho_list
)

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

#%%
import seaborn as sns
import pandas as pd
plt.style.use("seaborn")

# %% Getting the data in the right format for seaborn plot
meandata_list = []
skip_first_FIFO = 0
skip_first_SJF = 0
for i in range(len(rho_list)):
    for j in range(len(FIFOdatalist[i][3])):
        meandata_list.append((
            FIFOdatalist[i][3][j],
            "FIFO",
            str(rho_list[i])))

for i in range(len(rho_list)):
    for j in range(len(SJFdatalist[i][3])):
        meandata_list.append((
            SJFdatalist[i][3][j],
            "SJF",
            str(rho_list[i])))

df = pd.DataFrame(meandata_list, columns=["time", "service", "rho"],)

df.head()
#%%
sns.displot(df,bins=50,x="time", hue="service",col="rho",kde=True, legend=True)
# plt.savefig("figures/load_chr_FIFO_SJF_kde_4rhos_mu=0.95_njobs=2000_nsim=200.png", dpi=600)

sns.violinplot(data=df,x="rho",y="time",hue="service",legend=True)
# plt.savefig("figures/load_chr_FIFO_SJF_violin_4rhos_mu=0.95_njobs=2000_nsim=200-1.png", dpi=600)


#%% Saving general stats to json
print(F"for rho {[(stat[0], stat[1], stat[2]) for stat in FIFOdatalist]}")
fifostatslist = []
for i in range(len(FIFOdatalist)):
    fifostatslist.append(
        {
            rho_list[i]: {
                "mean": FIFOdatalist[i][0],
                "std": FIFOdatalist[i][1],
                "ci": FIFOdatalist[i][2],
            }
        }
    )
sjfstatslist = []
for i in range(len(SJFdatalist)):
    sjfstatslist.append(
        {
            "rho_" + str(rho_list[i]): {
                "mean": SJFdatalist[i][0],
                "std": SJFdatalist[i][1],
                "ci": SJFdatalist[i][2],
            }
        }
    )

stats_dict = {
    "FIFO": fifostatslist,
    "SJF": sjfstatslist
}
print(stats_dict)
#%%
from util import save_data
# save_data(stats_dict, "load_chr_FIFO_SJF_general_stats")
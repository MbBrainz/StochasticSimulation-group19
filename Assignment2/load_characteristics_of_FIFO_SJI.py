#%%  run simulations for different loads
# compare load statistics
import numpy as np
from simulation_functions import *
import matplotlib.pyplot as plt
from util import SEED

#%% [markdown]
# ## Check the behaviour of FIFO and SJF
#





#%%
np.random.seed(SEED)
n_sims = 200
n_jobs = 20000
mu = 0.95
rho_list = np.arange(0.8, 1.0, 0.01)
FIFOdatalist = []

for load in rho_list:
    FIFOdatalist.append(run_simulation(load,mu, n_servers=1, n_jobs=n_jobs,n_sims=n_sims,debug=1, iteration_function=FIFO_iteration))

#%%
SJFdatalist = []
for load in rho_list:
    SJFdatalist.append(run_simulation(load, mu, n_servers=1, n_jobs=n_jobs, n_sims=n_sims, debug=1, iteration_function=SJF_iteration))

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

save_data(data_dict, f"load_chr_FIFO_SJF_errorbar_drho=0.02_mu=095_njobs=20000-2")
# %%
plt.style.use('seaborn')
from util import plot_errorHue


fig, ax = plt.subplots(1, 1, )
plot_errorHue(rho_list, FIFOdatalist, "FIFO", ax)
plot_errorHue(rho_list, SJFdatalist, "SJF", ax)
ax.set(
    xlabel=r'load ($\rho$)',
    ylabel='average waiting time',
    title='load characeristics of service systems',
    xticks=rho_list[np.arange(0,20,2)]
)
ax.legend(loc="upper left", frameon=True, shadow=True)

# plt.savefig("load_chr_FIFO_SJF_errorHue_drho=0.01_mu=095_njobs=20000-2.png")
#%%


from util import plot_errorbar

fig, ax = plt.subplots(1, 1, )
plot_errorbar(rho_list, FIFOdatalist, "FIFO", ax)
plot_errorbar(rho_list, SJFdatalist, "SJF", ax)
ax.set(
    xlabel=r'load $(\rho)$',
    ylabel='average waiting time',
    title='load characeristics of service systems',
    xticks=rho_list
)
# plt.savefig("figures/load_chr_FIFO_SJF_errorbar_drho=0.02_mu=095_njobs=20000-1.png")

#%% Plot distributions of watining times  for 4 values of rho
n_sims = 500
n_jobs = 20000
mu = 0.95
rho_list = [0.8, 0.85, 0.90]
FIFOdatalist = []

for load in rho_list:
    FIFOdatalist.append(run_simulation(load,mu, n_servers=1, n_jobs=n_jobs,n_sims=n_sims,debug=1, iteration_function=FIFO_iteration, return_data=True))

SJFdatalist = []
for load in rho_list:
    SJFdatalist.append(run_simulation(load, mu, n_servers=1, n_jobs=n_jobs, n_sims=n_sims, debug=1, iteration_function=SJF_iteration,return_data=True))

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
df.to_csv("data/load_chr_FIFO_SJF_dataframe_nsims=200-big.csv")

df.head()
#%%
sns.displot(df,bins=200,x="time", hue="service",col="rho",kde=True, legend=True)
# plt.ylim(top=200)
# plt.xlim(right=40)
plt.savefig("figures/load_chr_FIFO_SJF_kde_3rhos_mu=0.95_njobs=20000_nsim=200-2.png", dpi=600)

#%%
# isRho09or08 = df['rho']== ( 0.9 | 0.8 )
# df_filtered=df[df['rho'].isin([0.8, 0.85, 0.9])]
# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# sns.violinplot(data= df,x="rho",y="time",hue="service",legend=True,ax=ax)
fig, ax = plt.subplots(1, 3, figsize=(10,5))
# order=["H", "M", "D"]
sns.violinplot(data=df[df["rho"] == "0.8"],  y="time", x="rho",hue="service",ax=ax[0] )
sns.violinplot(data=df[df["rho"] == "0.85"], y="time", x="rho",hue="service",ax=ax[1])
sns.violinplot(data=df[df["rho"] == "0.9"],  y="time", x="rho",hue="service",ax=ax[2] )
ax[0].get_legend().remove()
ax[0].set(xlabel="")
ax[1].get_legend().remove()
ax[1].set(ylabel="")
ax[2].set(ylabel="")
ax[2].set(xlabel="")
plt.tight_layout()
plt.savefig("figures/load_chr_FIFO_SJF_violin_3rhos_mu=0.95_njobs=20000_nsim=200-2.png", dpi=600)


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
# from util import save_data
save_data(stats_dict, "load_chr_FIFO_SJF_general_stats-2")
#%%
import numpy as np
from scipy import stats
import json
import matplotlib.pyplot as plt
plt.style.use("seaborn")

STUDENT_NR_MAURITS = 14014777
STUDENT_NR_HARSHITA = 13807609

SEED = STUDENT_NR_MAURITS + STUDENT_NR_HARSHITA

def confidence_interval(mean_value, std_value, N, level):
    ci = stats.norm.interval( # type: ignore
        level,
        loc=mean_value,
        scale=std_value/np.sqrt(N))
    return ci

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def save_data(data, filestring):
    a_file = open(f"data/{filestring}.json", "w")
    json.dump(data, a_file)
    a_file.close()

def plot_errorHue(x_array, datalist, label, ax):
    ax.plot(
    x_array,
    [x[0] for x in datalist],
    label=label
    )
    ax.fill_between(
    x_array,
    [x[0]+x[1] for x in datalist],
    [x[0]-x[1] for x in datalist],
    # label="std",
    alpha=0.5
                )
    ax.legend()

# %%
def plot_errorbar(x_array, datalist, label, ax):
    ax.errorbar(
        x_array,
        [x[0] for x in datalist],
        [x[1] for x in datalist],
        label=label,
        linestyle='--',
        marker='.',
        capsize=3,
        capthick=2
    )

def plot_sim_data(ax, data, label):
    ax.plot(
        data,
        label=label,
        linestyle='-',
        marker='.'
    )
# Result format
# data_dict = {
#     "parameters": {
#         "simulations": n_sims,
#         "jobs": n_jobs,
#         "mu": mu,
#         "system_type":"MM1",
#         "desc": "Simulation with multiple values for the load(rho)"
#     },
#     "xdata": list(rho_list),
#     "ydata": {
#         "FIFO": FIFOdatalist,
#         "SJF": SJFdatalist
#     }
# }
# %%

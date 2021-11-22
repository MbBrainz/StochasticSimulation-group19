# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import qmc

plt.style.use('seaborn')

def plot_random_points_distribution(re_list, im_list, re_lim=[-2,1], im_lim=[-1.25,1.25]):
    """Plot distribution of random points in the specified range

    Args:
        re_list (numpy.ndarray): array of real values of plot
        im_list (numpy.ndarray): array of imaginary values of plot
        re_lim (list, optional): range to plot on real axis. Defaults to [-2,1].
        im_lim (list, optional): range to plot on imaginary axis. Defaults to [-1.25,1.25].
    """
    plt.scatter(re_list, im_list)
    plt.xticks(ticks=np.arange(re_lim[0], re_lim[1], (re_lim[1]-re_lim[0])/len(re_list)))
    plt.yticks(ticks=np.arange(im_lim[0], im_lim[1], (im_lim[1]-im_lim[0])/len(im_list)))
    plt.xlim(re_lim)
    plt.ylim(im_lim)
    plt.xlabel("real axis")
    plt.ylabel("imaginary axis")
    plt.grid(b = True)
    plt.show()

# %%
def generate_pureRandomSample(n,re_lim=[-2,1], im_lim=[-1.25,1.25], show_plot=False):
    """generates a set of n random complex points with a uniform distribution

        Args:
            n (int): number of points generated
            re_lim (list, optional): limit wrt the real domain of the generated numbers. Defaults to [-2,1].
            im_lim (list, optional): limit wrt the imaginary domain of the generated point. Defaults to [-1.25,1.25].
            show_plot (bool, optional): Variable that lets you plot the distribution of the points on a grid. Defaults to False.

        Returns:
            numpy.ndarray: array with the random points. every point is given as (a + bj)
    """

    re = np.random.uniform(re_lim[0], re_lim[1], size=n)
    im = np.random.uniform(im_lim[0], im_lim[1], size=n)

    random_points = re+im*1j

    if show_plot: plot_random_points_distribution(re,im)

    return random_points

# %%
def generate_latinHyperCube(n,re_lim=[-2,1], im_lim=[-1.25,1.25], show_plot=False):
    """generates a set of n random complex points with a distribution based on Latin Hyper Cube sampling.
        See https://en.wikipedia.org/wiki/Latin_hypercube_sampling

        Args:
            n (int): number of points generated
            re_lim (list, optional): limit wrt the real domain of the generated numbers. Defaults to [-2,1].
            im_lim (list, optional): limit wrt the imaginary domain of the generated point. Defaults to [-1.25,1.25].
            show_plot (bool, optional): Variable that lets you plot the distribution of the points on a grid. Defaults to False.

        Returns:
            numpy.ndarray: array with the random points. every point is given as (a + bj)
    """

    width_re = re_lim[1] - re_lim[0]
    width_im = im_lim[1] - im_lim[0]

    re_all = []
    im_all = []

    factor_re = width_re/n
    factor_im = width_im/n

    low_re = re_lim[0]
    high_re = low_re + factor_re

    low_im = im_lim[0]
    high_im = low_im + factor_im

    for i in range(n):
        re = np.random.uniform(low_re, high_re)
        im = np.random.uniform(low_im,high_im)
        # update the range for both real and imaginary axis
        low_re = high_re
        high_re = low_re + factor_re
        low_im = high_im
        high_im = low_im + factor_im
        re_all.append(re)
        im_all.append(im)

    # In-place permutation for imaginary axis
    start = 0
    while(start+1 < n):
        selectIndex = np.random.randint(start,n)
        #swap start and selectIndex
        temp = im_all[selectIndex]
        im_all[selectIndex] = im_all[start]
        im_all[start] = temp
        start = start+1

    re_all = np.array(re_all)
    im_all = np.array(im_all)

    random_points = re_all+im_all*1j

    if show_plot: plot_random_points_distribution(re_all,im_all)

    return random_points

# %%
def generate_Orthogonal(n,re_lim=[-2,1], im_lim=[-1.25,1.25], show_plot=False):
    """generates a set of n random complex points with distribution based on orthogonal sampling. See
        See https://en.wikipedia.org/wiki/Latin_hypercube_sampling

        Args:
            n (int): number of points generated
            re_lim (list, optional): limit wrt the real domain of the generated numbers. Defaults to [-2,1].
            im_lim (list, optional): limit wrt the imaginary domain of the generated point. Defaults to [-1.25,1.25].
            show_plot (bool, optional): Variable that lets you plot the distribution of the points on a grid. Defaults to False.

        Returns:
            numpy.ndarray: array with the random points. every point is given as (a + bj)
    """

    n = int(np.round(np.sqrt(n)))

    width_x = re_lim[1]-re_lim[0]
    width_y = im_lim[1]-im_lim[0]

    num_sq = n*n
    xlist = np.arange(num_sq).reshape((n,n))
    ylist = np.arange(num_sq).reshape((n,n))
    x_scale = width_x/(num_sq)
    y_scale = width_y/(num_sq)
    re = []
    im = []
    for i in range(n):
        xlist[i] = np.random.permutation(xlist[i])
        ylist[i] = np.random.permutation(ylist[i])
    for i in range(n):
        for j in range(n):
            x = re_lim[0] + x_scale*(xlist[i][j] + np.random.uniform())
            y = im_lim[0] + y_scale*(ylist[j][i] + np.random.uniform())
            re.append(x)
            im.append(y)

    re_all = np.array(re)
    im_all = np.array(im)

    if show_plot: plot_random_points_distribution(re_all,im_all)
    return re_all+ im_all*1j

# %%
def generate_sobol(n, re_lim=[-2,1], im_lim=[-1.25,1.25], show_plot=False):
    """generates a set of n random complex points with distribution based on Sobol sampling. See
        See https://en.wikipedia.org/wiki/Sobol_sequence

        Args:
            n (int): number of points generated
            re_lim (list, optional): limit wrt the real domain of the generated numbers. Defaults to [-2,1].
            im_lim (list, optional): limit wrt the imaginary domain of the generated point. Defaults to [-1.25,1.25].
            show_plot (bool, optional): Variable that lets you plot the distribution of the points on a grid. Defaults to False.

        Returns:
            numpy.ndarray: array with the random points. every point is given as (a + bj)
    """

    re_im_sampler = qmc.Sobol(d = 2)
    l_bounds = [re_lim[0], im_lim[0]]
    u_bounds = [re_lim[1], im_lim[1]]

    num_samples = int(np.round(np.log2(n)))

    # 2**num_samples are generated ~ n
    re_im_samples = re_im_sampler.random_base2(num_samples)

    re_im_samples = qmc.scale(re_im_samples, l_bounds, u_bounds)

    if show_plot: plot_random_points_distribution(re_im_samples[:,0],re_im_samples[:,1])


    re_im_samples = re_im_samples[:,0] + re_im_samples[:,1]*1j
    re_im_samples = re_im_samples[:n]

    return re_im_samples

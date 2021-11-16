# %%
import numpy as np
import matplotlib.pyplot as plt

# pyright: reportGeneralTypeIssues=false
plt.style.use('seaborn')

#re_lim=(-2,1)
#im_lim=(-1,1)

# %%
def plot_latin(re_list, im_list, re_lim=[-2,1], im_lim=[-1.25,1.25]):
    plt.scatter(re_list, im_list)
    plt.xticks(ticks=np.arange(re_lim[0], re_lim[1], (re_lim[1]-re_lim[0])/len(re_list)))
    plt.yticks(ticks=np.arange(im_lim[0], im_lim[1], (im_lim[1]-im_lim[0])/len(im_list)))
    plt.xlim(re_lim)
    plt.ylim(im_lim)
    plt.grid(b = True)


# %%
def generate_pureRandomSample(n,re_lim=[-2,1], im_lim=[-1.25,1.25]):

    re = np.random.uniform(re_lim[0], re_lim[1], size=n) # type: ignore
    im = np.random.uniform(im_lim[0], im_lim[1], size=n) # type: ignore

    random_points = re+im*1j

    # plot_latin(re, im)
    # plt.show()

    return random_points

generate_pureRandomSample(9)

# %%
def generate_latinHyperCube(n,re_lim=[-2,1], im_lim=[-1.25,1.25]):

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

    # plot_latin(re_all, im_all)
    # plt.show()

    return random_points


generate_latinHyperCube(9)

# %%
def generate_Orthogonal(n,re_lim=[-2,1], im_lim=[-1.25,1.25]):

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
            x = -2.0 + x_scale*(xlist[i][j] + np.random.uniform())
            y = -1.0 + y_scale*(ylist[j][i] + np.random.uniform())
            re.append(x)
            im.append(y)


    # plot_latin(re,im)
    # plt.show()
    #print(re)
    #print(im

    re_all = np.array(re)
    im_all = np.array(im)
    return re_all+ im_all*1j

generate_Orthogonal(3)

# %%
def generate_Orthogonal_inPlace(n,re_lim=[-2,1], im_lim=[-1.25,1.25]):

    width_x = re_lim[1]-re_lim[0]
    width_y = im_lim[1]-im_lim[0]

    num_sq = n*n
    xlist = np.arange(num_sq).reshape((n,n))
    ylist = np.arange(num_sq).reshape((n,n))
    x_scale = width_x/(num_sq)
    y_scale = width_y/(num_sq)
    re = []
    im = []

    # inplace permutation
    for i in range(n):
        xlist[i] = np.random.permutation(xlist[i])
        ylist[i] = np.random.permutation(ylist[i])
    for i in range(n):
        for j in range(n):
            x = -2.0 + x_scale*(xlist[i][j] + np.random.uniform())
            y = -1.0 + y_scale*(ylist[j][i] + np.random.uniform())
            re.append(x)
            im.append(y)

    plot_latin(re,im)
    plt.show()
    #print(re)
    #print(im)
    re_all = np.array(re)
    im_all = np.array(im)
    return re_all+ im_all*1j

#generate_Orthogonal(3)

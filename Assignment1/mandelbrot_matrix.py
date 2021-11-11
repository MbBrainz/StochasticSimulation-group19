import numpy as np
import matplotlib.pyplot as plt
def mandelbrot_matrix(N=200, re_lim=(-2,1),im_lim=(-1,1), height=400, width=600):
    """Optimized implementation of mandelbrot set using array multiplications instead of for loops.

        Args:
            N (int, optional): Threshold value for iteration. Defaults to 200.
            re_lim (tuple, optional): limit of real axis. $(0) is the minimum value and $(1) is the maximum value. Defaults to (-2,1).
            im_lim (tuple, optional): limir of the imaginary axis. Defaults to (-1,1).
            height (int, optional): height of the image created. used to calculate the step size. Defaults to 400.
            width (int, optional): width of the image. Used to calculate the stepsize of the real axis. Defaults to 600.

        Returns:
            list: [0]The mendelbrot set matrix, [1]The real axis values, [2]The imaginary axis values
    """

    # define the values for the real and imaginary axis
    re_steps = (re_lim[1]-re_lim[0])/width
    im_steps = (im_lim[1]-im_lim[0])/height
    re = np.arange(re_lim[0], re_lim[1], re_steps, dtype=complex)
    im = np.arange(im_lim[0], im_lim[1], im_steps, dtype=complex) * 1j

    # create a matrix with the complex values
    c_matrix =  np.full((height,width), re) + np.full((width,height), im).transpose()

    # create the counter matrix "n" and the z matrix
    n = np.zeros((height,width))
    z = np.zeros((height,width))

    # iterate until the threshold is reached
    for k in range(N):
        # add one to n_ij only if abs(z_ij[k])<=2
        n = np.where(np.abs(z)<=2, n+1, n)
        # calculate z_ij[k+1] only if abs(z_ij[k])<=2
        z = np.where(np.abs(z)<=2, np.square(z) + c_matrix, z)

    return [n, re, im]
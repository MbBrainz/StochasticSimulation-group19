import numpy as np
import matplotlib.pyplot as plt

STUDENT_NR_MAURITS = 14014777
STUDENT_NR_HARSHITA = 13807609

SEED = STUDENT_NR_MAURITS + STUDENT_NR_HARSHITA


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

def random_mandelbrot_points(n_points, re_lim, im_lim, threshold=100):
    """Generates array of uniformly distributed complex numbers and calculates the mandelbrot set for these numbers

        Args:
            n_points (int): number of points taken. Each sample is an complex number
            re_lim (tuple): lower and upperlimit of the real axis
            im_lim (tuple): lower and upperlimit(real numbers) of the imaginary axis.
            threshold (int, optional): threshold used to calculate the mandelbrot set. Defaults to 100.

        Returns:
            list: [0]array of real numbers, [1]Array of imaginary numbers, [2]array of n values from the mandelbrot set.
    """
    # np.random.seed(SEED) #type: ignore
    re = np.random.uniform(re_lim[0], re_lim[1], size=n_points) # type: ignore
    im = np.random.uniform(im_lim[0], im_lim[1], size=n_points) # type: ignore

    random_points = re+im*1j
    # create the counter matrix "n" and the z matrix
    n = np.zeros(n_points)
    z = np.zeros(n_points)

    MAX_ITER = threshold
    # iterate until the threshold is reached
    for k in range(MAX_ITER):
        # add one to n_ij only if abs(z_ij[k])<=2
        n = np.where(np.abs(z) <= 2, n+1, n)
        # calculate z_ij[k+1] only if abs(z_ij[k])<=2
        z = np.where(np.abs(z) <= 2, np.square(z) + random_points, z)

    return re, im, n

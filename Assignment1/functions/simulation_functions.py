from sampling_functions import generate_pureRandomSample
import numpy as np
from tqdm.auto import trange, tqdm
from time import time

class TestResult:
    def __init__(self,
                 sample_mean:float,
                 sample_std: float,
                 trails:float,
                 sim_time: float,
                 funct,
                 confidence_int:float,
                 alpha: float):
        self.confidence_int = confidence_int
        self.mean = sample_mean
        self.std = sample_std
        self.trails = trails
        self.sim_time = sim_time
        self.alpha = alpha
        self.funct = funct

    def dict(self):
        return {
         str(self.funct): {
            "mean": self.mean,
            "std": self.std,
            "confidence_nt": self.confidence_int,
            "n(trails)": self.trails,
            "comp_time": self.sim_time
        }}

    def explain(self):
        """Prints the results to console in a readable format
        """
        print(f"Result of simulation for {self.funct}: \n \
                n_sims \t| \t {self.trails} \n    \
                mean \t| \t {self.mean:.4f} \n    \
                std \t| \t {self.std:.4g} \n    \
                comp. time\t| \t {self.sim_time:.0f} s \n    \
                conf. int.\t| \t {self.confidence_int:.3g} \n    \
                conf % \t| \t {(1-self.alpha)*100} \n    \
              ")
        # print(f"Interval reached after {self.trails} in {self.sim_time:.0g} seconds. mean={self.mean:.4f}, std={self.std:.4g}")
        # print(f"This means for a {(1-self.alpha)*100}% confidence interval, we have a={self.confidence_int:.3g} and X={self.mean:.5f}\n \
        # ")

def compute_mandelbrot_set(n_points, n_iterations=100, re_lim=(-2,1), im_lim=(-1.25,1.25), function=generate_pureRandomSample):
    """Generates array of uniformly distributed complex numbers and calculates the mandelbrot set for these numbers

        Args:
            n_points (int): number of points taken. Each sample is an complex number
            n_iterations (int, optional): threshold used to calculate the mandelbrot set. Defaults to 100.
            re_lim (tuple): lower and upperlimit of the real axis.
            im_lim (tuple): lower and upperlimit(real numbers) of the imaginary axis.
            function: function used to generate the random numbers

        Returns:
            tuple: [1]array of random points, [2]array of n values from the mandelbrot set.
    """
    random_points = function(n_points, re_lim=re_lim, im_lim=im_lim)
    n_points = len(random_points)
    # create the counter matrix "n" and the z matrix
    n = np.zeros(n_points)
    z = np.zeros(n_points)

    MAX_ITER = n_iterations
    # iterate until the threshold is reached
    for k in range(MAX_ITER):
        # add one to n_ij only if abs(z_ij[k])<=2
        n = np.where(np.abs(z) <= 2, n+1, n)
        # calculate z_ij[k+1] only if abs(z_ij[k])<=2
        z = np.where(np.abs(z) <= 2, np.square(z) + random_points, z)

    return random_points, n

def run_simulation(sample_size, n_points, n_iterations=100, re_lim=(-2,1), im_lim=(-1.25,1.25), function=generate_pureRandomSample):
    """Create a sample collection of size *sample_size by calculating the mandelbrot area for each iteration

    Args:
        sample_size (INT): amount of simulations to run
        n_points ): number of generated sample points for the mandelbrot set computation
        n_iterations (int, optional): number of iterations that is used as threshold to determine which points lay inside the mandelbrot set. Defaults to 100.
        re_lim (tuple, optional): upper and lower limit of the real axis. Defaults to (-2,1).
        im_lim (tuple, optional): upper and lower limit of the imaginary axis. Defaults to (-1.25,1.25).
        function ([type], optional): Sampling function that returns the amount of random sapleas that are used as input for the calculation od the M set. Defaults to generate_pureRandomSample.

    Returns:
        tuple: mean_area, standard deviation area, calculation time per iteration, sapmle data used
    """

    sample_data = []
    sample_time = []
    plot_area = (re_lim[1] - re_lim[0]) * (im_lim[1]-im_lim[0])
    for sample in tqdm(range(sample_size)):
        start_time = time()
        # run iteration
        random_points, n = compute_mandelbrot_set(n_points,  n_iterations=n_iterations, re_lim=re_lim, im_lim=im_lim,function=function)
        n_points = len(n)

        # calculate sample data
        points_inside = np.size(np.where(n>=n_iterations))
        mdb_area = points_inside / n_points * plot_area

        sample_data.append(mdb_area)

        calc_time = time() - start_time
        sample_time.append(calc_time)

    mean_area = np.mean(sample_data)
    std_area = np.std(sample_data)
    calc_time = np.mean(sample_time)

    return mean_area, std_area, calc_time, sample_data

def test_sampling_function_max_a(sampling_function, samplestep, max_a, n_iterations, n_points):
    """Tests the given sampling function by running *samplestep simulations and checking that with the max_a value.
        If the confidence interval of the sample data is larger than *max_a then runs another *samplestep iterations.
        If a is within max_a, then finish and return the result in a convenient format.

        Args:
            samplestep (int): amount of simulations to run before the threshold of max_a is tested
            n_points (int): number of generated sample points for the mandelbrot set computation
            max_a (float): max confidence interval used to compute if the simulations stops or continues
            n_iterations (int, optional): number of iterations that is used as threshold to determine which points lay inside the mandelbrot set. Defaults to 100.
            re_lim (tuple, optional): upper and lower limit of the real axis. Defaults to (-2,1).
            im_lim (tuple, optional): upper and lower limit of the imaginary axis. Defaults to (-1.25,1.25).
            function ([type], optional): Sampling function that returns the amount of random sapleas that are used as input for the calculation od the M set. Defaults to generate_pureRandomSample.


        Returns:
            [type]: [description]
    """
    a = 1
    mean, it_count, total_time, std = 0.0 ,0.0 ,0.0 ,0.0
    data = []

    while a > max_a:
        mean, std, sim_time, run_data = run_simulation(sample_size=samplestep,n_points=n_points,n_iterations=n_iterations, function=sampling_function)
        it_count += samplestep

        total_time += sim_time*samplestep
        data.append(run_data)
        std = np.std(data)
        mean = np.mean(data)
        a = get_confidence_interval(1.96, std, it_count)

    result = TestResult(sample_mean=mean,
                      sample_std=std,
                      trails=it_count,
                      sim_time=total_time,
                      confidence_int=a,
                      alpha=0.05,
                      funct=sampling_function.__name__)

    return result, data

def test_sampling_function_nsims(sampling_function, n_simulations, n_iterations, n_points):
    """Test the given sampling function for *n_simulations and returns a summerised result

        Args:
            sampling_function (function): A sampling function that returns an array with the samples in the form of a complex number
            n_simulations (int): amount of simulations to average the results over
            n_iterations (int): amount if iterations as threshold for the computation of the mandelbrot set
            n_points (int): Amount of samples to generate and calculate the mandelbrot set for

        Returns:
            tuple: [0] the resulting statistics of the simulations in a convenient class format Result. [1] Data generated from simulation to optionally add to previously collected data
    """

    print(f"Running simulation for {sampling_function.__name__}:")
    mean, std, sim_time, run_data = run_simulation(sample_size=n_simulations,n_points=n_points,n_iterations=n_iterations, function=sampling_function)
    total_time = sim_time*n_simulations
    a = get_confidence_interval(1.96, std, n_simulations)

    result = TestResult(sample_mean=mean,
                    sample_std=std,
                    trails=n_simulations,
                    sim_time=total_time,
                    confidence_int=a,
                    alpha=0.05,
                    funct=sampling_function.__name__)
    return result, run_data

def get_confidence_interval(lmda, sample_std, sample_size):
    """ calculates confidence interval for a given lamda and sample

    Args:
        lmda ([type]): fractile of confidence letter
        sample_std ([type]): sample standard deviation
        sample_size ([type]): int

    Returns:
        float: Confidence interval of the provided data. Lambda, sample_std, ample size
    """

    a = lmda * sample_std / np.sqrt(sample_size)
    return a
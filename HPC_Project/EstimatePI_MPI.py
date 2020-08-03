# Program to estimate the value of pi using Monte Carlo Simulations
# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import time
from mpi4py import MPI
# Obtaining the OpenMPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()
size = nProcs
# Setting the print options
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
iterations = 0
if rank == 0:
    # Getting the input of number of iterations
    print("Enter the number of iterations: ")
    iterations = int(input())
    start_time = time.time()
iterations = comm.bcast(iterations, root=0)
# Setting the iteration of block = iterations / nProcs
iterations_block = iterations / nProcs
# The random process is seeded with the rank of the process, so that no
# two process perform the same task for the same data
np.random.seed(rank)
# Generating random points from a uniform distribution
x = np.random.uniform(-0.5, 0.5, iterations_block)
y = np.random.uniform(-0.5, 0.5, iterations_block)
dist = np.empty(iterations_block)
# Calculating the distance of point from the centre
dist[:] = x[:]**2 + y[:]**2
dist = np.sqrt(dist)
# Boolean array to map whether a point is inside or outside the inscribed
# circle
boundness = np.empty(iterations_block)
boundness[:] = (dist[:] <= 0.5)
# Defining buffer variables for gathering
x_gather = None
y_gather = None
boundness_gather = None
if rank == 0:
    x_gather = np.empty(iterations)
    y_gather = np.empty(iterations)
    boundness_gather = np.empty(iterations)
# Gathering the point from all the processes
comm.Gather(x, x_gather, root=0)
comm.Gather(y, y_gather, root=0)
comm.Gather(boundness, boundness_gather, root=0)
if rank == 0:
    # Estimate pi by the expression estimate of pi = 4 * (Number of points
    # inside circle)/(Total number of points)
    estimate = 4.00 * sum(boundness_gather) / (iterations)
    end_time = time.time()
    print("Estimated value of PI = {}".format(estimate))
    print("Error = {}".format(np.fabs(estimate - np.pi)))
    print("Total time taken for execution - {} seconds".format(end_time - start_time))
    # Plotting the points and the inscribed circle to have a better
    # understanding
    dataframe = pd.DataFrame(
        {'X': x_gather, 'Y': y_gather, 'Bound': boundness_gather})
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    centreCircle = plt.Circle((0, 0), 0.5, color="red", fill=False)
    ax.add_patch(centreCircle)
    ax.set_aspect('equal', adjustable='box')
    sns.scatterplot(data=dataframe, x='X', y='Y', hue='Bound', alpha=0.4)
    R = 0.5
    theta = np.linspace(0, 2 * np.pi, 400)
    X_circle = R * np.cos(theta)
    Y_circle = R * np.sin(theta)
    plt.show()

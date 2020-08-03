# Program to calculate the price of path independent options using Monte Carlo Simulations
# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import math
from mpl_toolkits.mplot3d import axes3d
import scipy.stats as si
import time
from mpi4py import MPI
import seaborn as sns
import matplotlib.pyplot as plt
# Obtaining the OpenMPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nProcs = comm.Get_size()
size = nProcs
# Setting the print options
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
# Defining a class for the option trade which consits of stock price, strike price, option type,...
# risk free rate, volatility and time to maturity of the given option


class OptionTrade:

    def __init__(self, stock_price, strike_price, option_type, risk_free_rate, volatility, time_to_maturity):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.option_type = option_type
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_maturity = time_to_maturity / 365.00

# Defining a class setting for the Monte Carlo Simulation setting and this
# consists of time to maturity and iterations


class Settings:

    def __init__(self, time, iterations, time_step=365):
        self.time_step = time_step / 365.00
        self.time = time / 365.00
        self.iterations = iterations
        self.size_time_step = int(time / (time_step))
# Declaring the Monte Carlo Simulator


class MCSimulator:

    def __init__(self, trade, settings):
        self.trade = trade
        self.settings = settings
        self.predicted_spots = None

    def Simulate(self):
        # This part of the code does the simulation process
        # Obtaining the required data from the option trade and settings
        time_step = self.settings.time_step
        iterations = self.settings.iterations
        size_time_step = self.settings.size_time_step
        stock_price = self.trade.stock_price
        strike_price = self.trade.strike_price
        rate = self.trade.risk_free_rate
        option_type = self.trade.option_type
        vol = self.trade.volatility
        time_to_maturity = self.trade.time_to_maturity
        # The payoff and predicted spot are initialized to zero
        # Paths are stored for later access if needed
        # Average of path and Geometric average of path are calculated
        payoff = np.zeros(iterations)
        paths = np.zeros((iterations, size_time_step))
        predicted_spots = np.zeros(iterations)
        gm_average_path = np.zeros(iterations)
        average_path = np.zeros(iterations)
        # The random process is seeded with the rank of the process, so that no
        # two process perform the same task for the same data
        np.random.seed(rank)
        for path in range(iterations):
            # Generating a random variable array from a N(0,1)
            random_var = np.random.normal(0, 1, size_time_step)
            # Setting the inital path values to be zero
            path_value = np.zeros(size_time_step)
            path_value[0] = stock_price
            # Time marching to obtain S(T) from S(t), using the relationship between S(t+1) and S(t)
            for time in range(1, size_time_step):
                displacement = (rate - 0.5 * (vol)**2) * time_step + \
                    vol * random_var[time] * np.sqrt(time_step)
                path_value[time] = path_value[time - 1] * np.exp(displacement)
            paths[path, :] = path_value[:]
            predicted_spots[path] = path_value[-1]
            average_path[path] = np.average(path_value)
            # Computing the geometric average for calculating the payoff and the option price
            gm_average_path[path] = scipy.stats.gmean(path_value)
        # Depending on the option type the payoff is calculated as follows:
        if option_type == "CALL":
            payoff = np.maximum(
                gm_average_path[:] - strike_price, np.zeros(iterations))
        else:
            payoff = np.maximum(
            strike_price - gm_average_path[:], np.zeros(iterations))
        # The computed values are added to the MCS object and will be
        # accessed later while gathering from all the process
        self.paths = paths
        self.payoff = np.exp(-rate * time_to_maturity) * payoff
        self.predicted_spots = np.exp(-rate *
                                      time_to_maturity) * predicted_spots

# Function to plot the trajectories of the simulated spots versus the time axis

def predicted_spot_plt(MCS):
    paths = MCS.paths
    size_time_step = MCS.settings.size_time_step
    selected_paths = paths[:10, :]
    # First 10 paths are selected and plotted with time 
    time_frame = np.arange(size_time_step)
    plt.figure()
    for i in range(10):
        plt.plot(time_frame, selected_paths[i, :], '-*')
    plt.xlabel("Time Step")
    plt.ylabel("Estimated Stock price")
    plt.title("10 Simulated paths")
    plt.show()
# Declaring the option type and settings variables
strike_price = None
stock_price = None
volatility = None
option_type = None
rate = None
T_expiry = None
iterations = None
optionTrade = None
settings = None
if rank == 0:
    # All the values can be changes with the help of input function and no
    # further modification in code is needed
    strike_price = 100
    stock_price = 95
    volatility = 0.35
    option_type = "CALL"
    rate = 0.075
    T_expiry = 180
    # Getting the input for number of iterations and Time to maturity
    print("Enter the number of iterations: ")
    iterations = int(input())
    print("Enter time to expiry:")
    T_expiry = int(input())
    iterations_block = iterations / nProcs
    # Each of the process will be computing the MCS for iterations/nProcs and
    # later everything will be gathered
    start_time = time.time()
    optionTrade = OptionTrade(
        strike_price, stock_price, option_type, rate, volatility, T_expiry)
    settings = Settings(T_expiry, iterations_block, 1)
# Broadcasting all the variables
strike_price = comm.bcast(strike_price, root=0)
stock_price = comm.bcast(stock_price, root=0)
volatility = comm.bcast(volatility, root=0)
option_type = comm.bcast(option_type, root=0)
rate = comm.bcast(rate, root=0)
T_expiry = comm.bcast(T_expiry, root=0)
iterations = comm.bcast(iterations, root=0)
optionTrade = comm.bcast(optionTrade, root=0)
settings = comm.bcast(settings, root=0)
# Initiating the Monte Carlo Simulator for number of iterations =
# iterations/nProcs for each of the process
MCS = MCSimulator(optionTrade, settings)
val = MCS.Simulate()
# Payoff and predicted spots of the Monte Carlo simulation are sorted back
# in variables payoff and predicted_spots
payoff = MCS.payoff
predicted_spots = MCS.predicted_spots
# Buffers to gather the payoff and predicted spots from all the processes
payoff_combined = None
predicted_spots_combined = None
if rank == 0:
    # Size of the combined array = iterations
    payoff_combined = np.zeros(iterations)
    predicted_spots_combined = np.zeros(iterations)
# Gathering the payoff and predicted spots from all the processes
comm.Gather(payoff, payoff_combined, root=0)
comm.Gather(predicted_spots, predicted_spots_combined, root=0)
if rank == 0:
    # Computing the option price in root
    option_price = np.average(payoff_combined)
    end_time = time.time()
    predicted_spot_plt(MCS)
    # Plotting the histogram of predicted spots to understand the distribution
    # of the predicted spots
    plt.hist(predicted_spots_combined, bins=100)
    plt.xlabel("Predicted Spot Value")
    plt.ylabel("Count")
    plt.title("Distribution of predicted values")
    plt.show()
    # Output the simulation result - price of the given option
    print("Monte-Carlo Simulation of the given option = {} $".format(option_price))
    if T_expiry == 30:
        print("Closed form reference price = {}$".format(5.730))
    var_payoff = np.var(payoff_combined)
    print("Variance of the simulation = {}".format(var_payoff))
    confidence_interval_bound_left = option_price - \
        (1.96) * (np.sqrt(var_payoff / iterations))
    confidence_interval_bound_right = option_price + \
        (1.96) * (np.sqrt(var_payoff / iterations))
    confidence_interval = 2 * (1.96) * (np.sqrt(var_payoff / iterations))
    print("95% Confidence interval of the simulated option = {} $ to {} $".format(
        confidence_interval_bound_left, confidence_interval_bound_right))
    print("Size of confidence interval = {}$".format(confidence_interval))
    print("The total time taken for execution = {} seconds".format(
        end_time - start_time))

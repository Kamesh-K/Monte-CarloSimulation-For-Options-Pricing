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
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
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
        self.time = time / 365.00
        self.iterations = iterations
# Defining one of the closed form expression available for european options with path independency
# Black Scholes Merton Model for Option Pricing


class BSModel:

    def __init__(self, trade):
        self.trade = trade

    def pricer(self):
        trade = self.trade
        T = trade.time_to_maturity
        sigma = trade.volatility
        S = trade.stock_price
        r = trade.risk_free_rate
        K = trade.strike_price
        option_type = trade.option_type
        d1 = (np.log(S / K) + (r + (0.5 * sigma**2) * T)) / (sigma * (T**0.5))
        d2 = d1 - sigma * (T**(0.5))
        if option_type == "CALL":
            execute_val = S * norm.cdf(d1)
            future_val = K * math.exp(-1.0 * r * T) * norm.cdf(d2)
            current_val = execute_val - future_val
            return current_val
        else:
            execute_val = S * (norm.cdf(d1) - 1)
            future_val = K * math.exp(-1.0 * r * T) * (norm.cdf(d2) - 1)
            current_val = execute_val - future_val
            return current_val
# Declaring the Monte Carlo Simulator


class MCSimulator:

    def __init__(self, trade, settings):
        self.trade = trade
        self.settings = settings

    def Simulate(self):
        # This part of the code does the simulation process
        # Obtaining the required data from the option trade and settings
        # objects
        iterations = self.settings.iterations
        stock_price = self.trade.stock_price
        strike_price = self.trade.strike_price
        rate = self.trade.risk_free_rate
        option_type = self.trade.option_type
        vol = self.trade.volatility
        time_to_maturity = self.trade.time_to_maturity
        # The payoff and predicted spot are initialized to zero, with the size
        # of iterations
        payoff = np.zeros(iterations)
        predicted_spots = np.zeros((iterations))
        # The random process is seeded with the rank of the process, so that no
        # two process perform the same task for the same data
        np.random.seed(rank)
        for path in range(iterations):
            # Generating a random variable from a N(0,1)
            random_var = np.random.normal(0, 1)
            # Calculating the exponent value of displacement
            displacement = (rate - 0.5 * (vol)**2) * time_to_maturity + \
                vol * random_var * np.sqrt(time_to_maturity)
            # From the expression, we know that the S(T) = S(t) *
            # exp(displacement) with time step as time to maturity
            path_value = stock_price * np.exp(displacement)
            predicted_spots[path] = path_value
            # Storing the predicted spots for later reference and payoff calculation
        # Depending on the option type the payoff is calculated as follows:
        if option_type == "CALL":
            payoff = np.maximum(
                predicted_spots[:] - strike_price, np.zeros(iterations))
        else:
            payoff = np.maximum(
                strike_price - predicted_spots[:], np.zeros(iterations))
            # The computed values are added to the MCS object and will be
            # accessed later while gathering from all the process
        self.payoff = np.exp(-rate * time_to_maturity) * payoff
        self.predicted_spots = np.exp(-rate *
                                      time_to_maturity) * predicted_spots

# Function to plot the trajectories of the simulated spots versus the time axis


def predicted_spot_plt(predicted_spots, stock_price):
    x = []
    y = []
    # The predicted spots are converted to line plots using the following code
    for i in range(len(predicted_spots)):
        x.append(0)
        x.append(1)
        y.append(stock_price)
        y.append(predicted_spots[i])
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Time Step")
    plt.ylabel("Stock price")
    plt.title("Simulated prices of stock")
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
    strike_price = 25
    stock_price = 20
    volatility = 0.35
    option_type = "CALL"
    rate = 0.055
    T_expiry = 182
    # Getting the input for number of iterations
    print("Enter the number of iterations: ")
    iterations = int(input())
    # Each of the process will be computing the MCS for iterations/nProcs and
    # later everything will be gathered
    iterations_block = iterations / nProcs
    start_time = time.time()
    optionTrade = OptionTrade(
        strike_price, stock_price, option_type, rate, volatility, T_expiry)
    settings = Settings(T_expiry, iterations_block, T_expiry)
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
    if iterations <= 20000:
        # If the iterations are less, we can visualize and see the predicted
        # spots vs Time
        predicted_spot_plt(predicted_spots_combined, stock_price)
    # Plotting the histogram of predicted spots to understand the distribution
    # of the predicted spots
    plt.hist(predicted_spots_combined, bins=100)
    plt.xlabel("Predicted Spot Value")
    plt.ylabel("Count")
    plt.title("Distribution of predicted values")
    plt.show()
    # Output the simulation result - price of the given option
    print("Monte-Carlo Simulation of the given option = {} $".format(option_price))
    print("Closed form reference price = {}$".format(5.9842))
    var_payoff = np.var(payoff_combined)
    print("Variance of the simulation = {}".format(var_payoff))
    confidence_interval_bound_left = option_price - \
        (1.96) * (np.sqrt(var_payoff / iterations))
    confidence_interval_bound_right = option_price + \
        (1.96) * (np.sqrt(var_payoff / iterations))
    confidence_interval = 2 * (1.96) * (np.sqrt(var_payoff / iterations))
    print("Black Scholes Model price = {}$".format(
        BSModel(optionTrade).pricer()))
    print("95% Confidence interval of the simulated option = {} $ to {} $".format(
        confidence_interval_bound_left, confidence_interval_bound_right))
    print("Size of confidence interval = {}$".format(confidence_interval))
    print("The total time taken for execution = {} seconds".format(
        end_time - start_time))

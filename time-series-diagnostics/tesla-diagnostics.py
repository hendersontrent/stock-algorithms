#---------------------------------
# This script sets out to run
# some high level time series
# diagnostics on TSLA stock data
#---------------------------------

#-------------------------------------
# Author: Trent Henderson, 29 May 2020
#-------------------------------------

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import yfinance as yf

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tspype import tsplotter
from tspype import stationary_calculator
from tspype import decomposer

#%%
#--------------------COLLECT DATA-----------------------
# Load TSLA stock data

tsla = yf.Ticker("TSLA")
data = tsla.history(period = "max")

data.reset_index(level = 0, inplace = True)
data['Date'] = pd.to_datetime(data['Date'])
data

#---------------------APPLY FUNCTIONS-------------------

#%%
# Overall
tsplotter(data['Date'], data['Close'], data)

#%%
# Stationarity
stationary_calculator(data['Close'], 0.05)

#%%
# Decomposition
decomposer(data, data['Date'], data['Close'], periods = 365)

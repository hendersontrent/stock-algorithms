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

#%%
#-----------------------DEFINE REUSABLE FUNCTIONS-----------------

# Build a function that returns a time series plot, density of y values, ACF and PACF

def tsplotter(x, y, data, figsize = (12,8)):
    sns.set(style = "darkgrid")
    fig = plt.figure(figsize = figsize)
    layout = (2,2)
    data_ax = plt.subplot2grid(layout, (0,0))
    dens_ax = plt.subplot2grid(layout, (0,1))
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    
    sns.lineplot(x = x, y = y, 
                 data = data, ax = data_ax)
    sns.distplot(y, ax = dens_ax)
    plot_acf(y, ax = acf_ax)
    plot_pacf(y, ax = pacf_ax)
    sns.despine()
    plt.tight_layout()
    return data_ax, dens_ax, acf_ax, pacf_ax

# Write a reusable stationarity calculator function

def stationary_calculator(data, sig_threshold):
    
    the_test = adfuller(data)
    
    adf_statistic = the_test[0]
    adf_pvalue = the_test[1]
    
    if adf_pvalue > sig_threshold:
        print("Your data is non-stationary and has a time-dependent structure.")
        print("Test: Augmented Dickey-Fuller")
        print('ADF Statistic: %f' % adf_statistic)
        print('ADF p-value: %f' % adf_pvalue)
    else:
        print("Your data is stationary.")
        print("Test: Augmented Dickey-Fuller")
        print('ADF Statistic: %f' % adf_statistic)
        print('ADF p-value: %f' % adf_pvalue)
        
# Write a reusable decomposition function

def decomposer(data, x, y, periods, figsize = (12,8)):
        decomposition = seasonal_decompose(y, period = periods)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
    
        sns.set(style = "darkgrid")
        fig = plt.figure(figsize = figsize)
        layout = (2,2)
        orig_ax = plt.subplot2grid(layout, (0,0))
        trend_ax = plt.subplot2grid(layout, (0,1))
        seas_ax = plt.subplot2grid(layout, (1,0))
        resid_ax = plt.subplot2grid(layout, (1,1))
    
        sns.lineplot(x = x, y = y, 
                     data = data, ax = orig_ax)
        trend.plot(ax = trend_ax, title = 'Trend')
        seasonal.plot(ax = seas_ax, title = 'Seasonal')
        residual.plot(ax = resid_ax, title = 'Residual')
        plt.tight_layout()
        return orig_ax, trend_ax, seas_ax, resid_ax

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

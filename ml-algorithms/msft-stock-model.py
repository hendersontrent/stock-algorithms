#---------------------------------
# This script sets out to build a
# machine learning model on
# MSFT stock data
#---------------------------------

#-------------------------------------
# Author: Trent Henderson, 30 May 2020
#-------------------------------------

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import yfinance as yf
import math
import time
import sklearn
import matplotlib.ticker as mtick
sns.set()

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import newaxis

#%%
#-----------------PRE PROCESSING---------------------

# Load MSFT stock data

msft = yf.Ticker("MSFT")
data = msft.history(period = "max")

data.reset_index(level = 0, inplace = True)
data['Date'] = pd.to_datetime(data['Date'])
data

stock_prices = data['Close']
stock_prices = stock_prices.values.reshape(len(stock_prices), 1)
stock_prices.shape

# Plot data to visualise

plt.plot(stock_prices)
plt.title("Reshaped MSFT stock prices")
plt.show()

#%%
# Scale data ready for algorithm

scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices = scaler.fit_transform(stock_prices)

# Split into train and test data

train_size = int(len(stock_prices) * 0.80)
test_size = len(stock_prices) - train_size
train, test = stock_prices[0:train_size,:], stock_prices[train_size:len(stock_prices),:]
print(len(train), len(test))

# Convert function to convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Use function to reshape data into X=t and Y=t+1 - this frames 
# data as a SUPERVISED LEARNING PROBLEM
# with inputs and outputs side-by-side

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#%%
# Determine optimal batch size using highest common factor

def computeHCF(x,y):
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i
            
    return hcf

the_batch = computeHCF(trainX.shape[0], testX.shape[0])

#%%
#-----------------MODEL DEVELOPMENT---------------------

# Create and fit the LSTM network

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back), dropout = 0.5))
model.add(Dense(1))
opt = Adam(learning_rate = 0.1)
model.compile(loss='mean_squared_error', optimizer= opt)
early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1) # Stops after no improvement across 2 epochs
model.fit(trainX, trainY, epochs = 100, batch_size = the_batch, verbose = 2, callbacks = [early_stop])

#%%
# Visualise loss

history = model.fit(trainX, trainY, epochs = 100, batch_size = the_batch, verbose = 2, callbacks = [early_stop],
                    validation_data = (testX, testY))
plt.plot(history.history['loss'], label = "Train")
plt.plot(history.history['val_loss'], label = "Validation")
plt.title('Model train vs validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc = 'upper right')
plt.show()

#%%
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#%%
#-----------------PREDICTIONS---------------------------

# Reshape Y data

train_shape = trainY.shape
trainYr = trainY.reshape((train_shape[0]))

test_shape = testY.shape
testYr = testY.reshape((test_shape[0]))

# Invert predictions

trainPredict = scaler.inverse_transform(trainPredict)
trainYr = scaler.inverse_transform([trainYr])
testPredict = scaler.inverse_transform(testPredict)
testYr = scaler.inverse_transform([testYr])

# Calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainYr[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testYr[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#%%
#-----------------PLOTTING OUTPUTS---------------------

# Shift train predictions for plotting

trainPredictPlot = np.empty_like(stock_prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting

testPredictPlot = np.empty_like(stock_prices)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(stock_prices)-1, :] = testPredict

# Plot baseline and predictions

plt.figure(figsize = (14,8))
plt.plot(scaler.inverse_transform(stock_prices))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(["Actual data", "Algorithm train", "Algorithm test"])
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.title("LSTM algorithm predictions vs actual data")
plt.annotate('Train RMSE: %.2f' % (trainScore), xy = (-350,164))
plt.annotate('Test RMSE: %.2f' % (testScore), xy = (-350,157))

fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
plt.gca().yaxis.set_major_formatter(tick) # Add dollar sign tick marks

plt.show()

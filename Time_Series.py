# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:06:47 2021

@author: user
"""
## TIME_SERIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ADF (Augmented Dicky fuller test)
from statsmodels.tsa.stattools import adfuller,acf,pacf

# ARIMA
from statsmodels.tsa.arima_model import ARIMA


import statsmodels.api as sm

#read the data
path="C:/Users/user/Desktop/mrf.csv"

stock=pd.read_csv(path)
stock
stock.shape
stock.columns
stock.head(10)

# take column 'close price' to forecast
mystock=stock[['Close Price']]
mystock

# check if the data is stationary using ADF test
def checkStationarity(data):
    # return value of adfuller()
    # i) test statistic
    # ii) pvalue
    # iii) lags
    # iv) total observation
    
    pvalue = adfuller(data)[1]
    if pvalue > 0.05:
        msg = 'p-value = {}. Data not stationary'.format(pvalue)
    else:
        msg = 'p-value = {}. Data is stationary'.format(pvalue)
        
    return(msg)
        
# function call to test if the input data is stationary

ret = checkStationarity(mystock);ret

# since the data is not stationarity, it has to be made stationary
# procedure: take a diffrence of data
diff_mystock = mystock-mystock.shift()

dd = pd.DataFrame({'actual':mystock['Close Price'], 'diffrenced':diff_mystock['Close Price']})    
dd
    
3056.45-2811.60

# drop the null values first
print("before droppping NA. Count = ", len(diff_mystock))
diff_mystock.dropna(inplace=True)
print("After dropping NA. count = ", len(diff_mystock))

# check for stationarity of diffrenced data
ret = checkStationarity(diff_mystock);ret

# plot the actual and stationary data for analysis
plt.subplot(121)
plt.plot(mystock,color='red')
plt.title('Actual Close Price')

plt.subplot(122)
plt.plot(diff_mystock,color='blue')
plt.title('Diffrenced Close Price')

# plot the PACF and ACF graphs  (correlograms) to determine the p and q values

lags_pacf = pacf(diff_mystock,nlags=20)
lags_acf = acf(diff_mystock,nlags=20)

# plot PACF 
plt.subplot(121)
plt.plot(lags_pacf)
plt.axhline(y=0, linestyle='-',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(diff_mystock)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff_mystock)),linestyle='--',color='gray')
plt.title('PACF')

# plot ACF
plt.subplot(122)
plt.plot(lags_acf)
plt.axhline(y=0, linestyle='-',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(diff_mystock)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff_mystock)),linestyle='--',color='gray')
plt.title('ACF')

# initialise the values for p,d,q
p=0;q=0;d=0

# ARIMA model
# model 1 -> ARIMA(0,0,0) model
m1 = ARIMA(diff_mystock,order=(p,d,q)).fit(disp=0)

m1.summary()

# plot the residuals
plt.hist(m1.resid)
plt.title("ARIMA residuals")

# Ljungbox test for residuals independence
# H0: residuals are independentely disributed
# H1: residuals are not independentely disributed
pvalue = sm.stats.acorr_ljungbox(m1.resid,lags=[1])[1]
if pvalue > 0.05:
    print("FTR H0. residuals are independentely disributed")
else:
    print("Reject H0.residuals are not independentely disributed")



# Forecast for the next "12" months

p1 = m1.forecast(steps=12)
p1

# forecasted values are in the diffrenced format
# they have to be converted into the original form

predictions = p1[0]
len(predictions)
predictions


# model 2 -> ARIMA(1,0,0) model
# model 3 -> ARIMA(0,0,1) model

# the best model is one that has the least AIC score
mystock
diff_mystock

####################

ret1 = checkStationarity(diff_mystock);ret1


dd1 = pd.DataFrame({'actual':diff_mystock['Close Price'], 'diffrenced':diff_mystock['Close Price']})    
dd1






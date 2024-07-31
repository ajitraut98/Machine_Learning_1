# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:32:32 2021

@author: user
"""
# ml_exam2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#ADF(augmented dickey fuller test)
from statsmodels.tsa.stattools import adfuller,acf,pacf

#Arima
from statsmodels.tsa.arima_model import ARIMA

#for lJung box
import statsmodels.api as sm

path= "C:/Users/user/Desktop/MonthWiseMarketArrivals_Clean.csv"

onion=pd.read_csv(path)
onion.columns
print(onion)


#1)
ss =onion[onion.market=='MUMBAI'].index
print(ss)
onion.iloc[6654:6800,6]

#2)
mdp = onion[['priceMod']] 

#mdp = mdp.astype('float64')

def stationry(data):
       
    pvalue = adfuller(data)[1]
    if pvalue > 0.05:
        msg = 'p-value = {}, Data is not stationary'.format(pvalue)
    else:
        msg = 'p-value = {}, Data is stationary'.format(pvalue)
    return(msg)

# function call to test if the input stationary , it has to be made stationary
ret = stationry(mdp); ret

p=0; d=0; q=0

m1 = ARIMA(mdp, order=(p,d,q)).fit(disp=0)

p1 = m1.forecast
p1

#3)

lags_pacf = pacf(mdp,nlags=20)
lags_acf = acf(mdp,nlags=20)


# PACF (p) value
plt.subplot(121)
plt.plot(lags_pacf)
plt.axhline(y=0, linestyle='-',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(mdp)),linestyle='--',color='red')
plt.axhline(y=1.96/np.sqrt(len(mdp)),linestyle='--',color='red')
plt.title('PACF')

# ACF (q) value
plt.subplot(122)
plt.plot(lags_acf)
plt.axhline(y=0, linestyle='-',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(mdp)),linestyle='--',color='red')
plt.axhline(y=1.96/np.sqrt(len(mdp)),linestyle='--',color='red')
plt.title('ACF')


























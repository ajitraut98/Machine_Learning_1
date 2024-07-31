# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:00:32 2021

@author: user
"""

#decision tree with regression
#dataset:energy_cooling_load




import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#read the data
path="C:/Users/user/Desktop/energy_cooling_load.csv"

energy=pd.read_csv(path)
energy

# perform th EDA


#split the data into train and test
train,test=train_test_split(energy,test_size=0.3)

train.columns

trainx = train.drop('cold_load',axis=1)
trainy = train['cold_load']

testx = test.drop('cold_load',axis=1)
testy = test['cold_load']


print('trainx={},triany={},testx={},testy={}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))

#build the decisionTree regression model
m1 = DecisionTreeRegressor(criterion='mse').fit(trainx,trainy)


#predict on the test data
p1 = m1.predict(testx)

#Store the actual and predicted values in a dataframe for analysis
df = pd.DataFrame({'actual':testy,'predicted':p1})

print(df)



#build an OlS model on the same dataset and check the differences
m2= sm.OLS(trainy,trainx).fit()
p2= m2.predict(testx)

#Store the actual and predicted values in a dataframe for analysis

df = pd.DataFrame({'actual':testy,'p_DT':p1,
                     'p_OLS':p2})

df

#mse of both the models
mse_dt=round(mean_squared_error(testy,p1),3)
mse_ols = round(mean_squared_error(testy,p2),3)
print("MSE \n DT={},OLS={}".format(mse_dt,mse_ols))
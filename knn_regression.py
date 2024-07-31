# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:42:07 2021

@author: user
"""

## KNN REGRESSION

# import libraries

import pandas as pd
import numpy as np


# standardize the data
from sklearn import neighbors,preprocessing

from sklearn.model_selection import train_test_split,cross_val_score

# knn algorithms

from sklearn.metrics import mean_squared_error



# plot the graph to identify the best k
import matplotlib.pyplot as plt
import seaborn as sns

#read the data
path="C:/Users/user/Desktop/cars.csv"

cars=pd.read_csv(path)
cars
cars.shape

cars.head(10)

# remove unwanted features
cars = cars.drop(['origin','name'],axis=1)
cars

# standardize the data
cars_std = cars.copy()
minmax = preprocessing.MinMaxScaler()
scaledvals = minmax.fit_transform(cars_std)
cars_std.iloc[:,:] = scaledvals
cars_std.head(10)
cars_std.mpg = cars.mpg
cars_std = cars_std.sample(frac=1)


# split the data into train and test 
trainx,testx,trainy,testy = train_test_split(cars_std.drop('mpg',axis=1),cars_std['mpg'],test_size=0.30)

print(trainx.shape,trainy.shape,testx.shape,testy.shape)

# find the best k by cross-validation
mse_cv = []
nn = range(3,12)
print(list(nn))
for k in nn:
    # build model with neighbours 
    model = neighbors.KNeighborsRegressor(n_neighbors=k).fit(trainx,trainy)
    pred=model.predict(testx)
    mse=mean_squared_error(testy,pred).mean()
    mse_cv.append(round(mse,3))
    
print(mse_cv)

# take the index of the smallest MSE
optk = nn[mse_cv.index(min(mse_cv))];optk

# build model with k=5
m1=neighbors.KNeighborsRegressor(n_neighbors=optk).fit(trainx,trainy)
p1=m1.predict(testx)


# store actual and predicted data for comparison
res=pd.DataFrame({'actualmpg':testy,'predmpg':p1})

# visualise the predictions
sns.regplot(testy,p1,ci=False,marker="o",color="red")
plt.title('actual vs predicted mpg')

# plt.title('err',loc='right')

## Assignment
# 1) use the actual dataset  to build the 2nd model and compare the result

trainx.head(20)
trainx.index

trainx2 = cars.loc[trainx.index,]
trainx2.head(10)
trainx.head(10)

# this is not required, since Y is the same
'''
trainy2=cars.loc[trainx.index,'type']
trainy.head(5)
trainy2.head(5)
'''

testx2 = cars.loc[testx.index,]
testx2.head(10)
testx.head(10)


# find the best k by cross-validation
mse_cv1 = []
nn = range(3,12)
print(list(nn))
for k in nn:
    # build model with neighbours 
    model = neighbors.KNeighborsRegressor(n_neighbors=k).fit(trainx2,trainy)
    pred=model.predict(testx2)
    mse=mean_squared_error(testy,pred).mean()
    mse_cv.append(round(mse,3))
    
print(mse_cv)

# take the index of the smallest MSE
optk1 = nn[mse_cv1.index(min(mse_cv1))];optk1

# build model with k=5
m2=neighbors.KNeighborsRegressor(n_neighbors=optk1).fit(trainx2,trainy)
p2=m2.predict(testx2)


# store actual and predicted data for comparison
res1=pd.DataFrame({'actualmpg':testy,'predmpg':p2})

# visualise the predictions
sns.regplot(testy,p2,ci=False,marker="o",color="blue")
plt.title('actual vs predicted mpg')

m1_mse = mean_squared_error(testy,p1)
m1_mse
m2_mse = mean_squared_error(testy,p2)
m2_mse


# 2) use Linear Regression, Decision Tree, Random Forest on the same train/test data and compare the performances of the models

# LINEAR REGRESSION
ols_model = sm.OLS(trainy,trainx).fit()
ols_pred = ols_model.predict(testx)
ols_mse = mean_squared_error(testy, ols_pred)
ols_mse

# DECISION TREE
dt_model = DecisionTreeRegressor(criterion="mse").fit(trainx,trainy)
dt_pred = dt_model.predict(testx)
dt_mse = mean_squared_error(testy, dt_pred)
dt_mse

# RANDOM FOREST
rd_model = RandomForestRegressor().fit(trainx,trainy)
rd_pred = rd_model.predict(testx)
rd_mse = mean_squared_error(testy, rd_pred)
rd_mse

r = pd.DataFrame({'1st':m1_mse,'2nd':m2_mse,'OLS mse':ols_mse,'Decisiontree mse':dt_mse,'Randomforest mse':rd_mse})

res_mse = pd.DataFrame({'1st mse':m1_mse,})

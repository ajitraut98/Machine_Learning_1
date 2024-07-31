# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:23:18 2021

@author: user
"""

# SVM Regression
# dataset: energy_cooling_load

# import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# read the data
path= "C:/Users/user/Desktop/energy_cooling_load.csv"
energy = pd.read_csv(path)
energy.columns


# standardize the dataset
energy_std = energy.copy()
minmax=preprocessing.MinMaxScaler()
energy_std.iloc[:,:] = minmax.fit_transform(energy_std.iloc[:,:])
energy_std['cold_load'] = energy['cold_load']
energy_std

# split the data
trainx,testx,trainy,testy=train_test_split(energy_std.drop('cold_load',axis=1),
                                           energy_std['cold_load'],
                                           test_size=0.25)

print(trainx.shape,trainy.shape,testx.shape,testy.shape)


kernels=['linear', 'rbf', 'sigmoid', 'poly']

# get the R-square for each kernel
for k in kernels:
    model = svm.SVR(kernel=k).fit(trainx,trainy)
    rsq = model.score(testx,testy)
    print('Kernel = ',k, " RSquare = ", rsq)


# placeholders
m_mse=[]
model=[]
preds=[]

# SVM regression function call
def svmRegression(ker,trainx,trainy,testx,testy,bestc=1,bestg='scale'):
    model = svm.SVR(kernel=ker,C=bestc,gamma=bestg).fit(trainx,trainy)
    pred = model.predict(testx)
    mse = round(mean_squared_error(testy,pred),2)
    return(pred,mse)

def showPlot(actual,predicted,ker,e):
    # plot the actual vs predicted Y (cold_load)
    ax1=sns.distplot(actual,hist=False,color='r',label='Actual Y')
    sns.distplot(predicted,hist=False,color='b',label='Predicted Y',ax=ax1)
    # plt.title('Kernel = ' + ker + ' MSE = '+ str(mse))
    plt.title('Actual vs Predicted Data for ' + ker + ' Model. MSE = ' + str(e))
    return(1)


# build all the regression models for each Kernel
for k in kernels: 
    model.append(k)
    pred,e=svmRegression(k,trainx,trainy,testx,testy)
    preds.append(pred)
    m_mse.append(e)



# plot the actual vs predicted graph for the selected Kernel
ker = "poly"
ndx = kernels.index(ker)
showPlot(testy,preds[ndx],ker,m_mse[ndx])


model
m_mse
preds


pd.DataFrame({'model':model,'MSE':m_mse})


# Assignments
# 1) how to print the graph in 1 page


































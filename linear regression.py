# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:19:06 2021

@author: user
"""
########## CHAP-- LINEAR REGRESSION

# load thre liabraries
import pandas as pd
import numpy as np
import math
import pylab
dir(pd)
# scikit library for linear regression

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import seaborn as sns
import matplotlib.pyplot as plt

# read the data
path= "C:/Users/user/Desktop/energy_cooling_load.csv"

energy = pd.read_csv(path)

#EDA

# dimension
energy.shape

# data types
energy.dtypes

# print the first and last 3 records
energy.head(3)
energy.tail(3)

# get all columns names
energy.columns

# summarise the dataset
desc=energy.describe()
desc

# NULL check
energy.isnull().sum()

# to '0' check
energy[energy==0].count()


cols=list(energy.columns)
cols.remove('cold_load')

energy.boxplot('rel_comp',vert=False)

# check for outliers
row=3; col=3; pos=1
fig=plt.figure()

for c in cols:
    fig.add_subplot(row,col,pos)
    energy.boxplot(column=c,vert=False)
    pos+=1

# histogram / distrubution plot

row=3; col=3; pos=1
fig=plt.figure()

for c in cols:
    fig.add_subplot(row,col,pos)
    sns.distplot(energy[c])
    pos+=1

# correlation check  (check colinearity)
total=len(cols)
cor=energy.iloc[:,0:total-1].corr()
cor

# take only the lower tringle to plot the heatmap
cor=np.tril(cor)
cor

# plot the heatmap
sns.heatmap(cor,xticklabels=cols,yticklabels=cols,vmin=-1,vmax=1,annot=True,square=True)



# split the data into train and test
train,test = train_test_split(energy, test_size=0.3)

train.shape
test.shape

# split the data further into trainx/y and testx/y
trainx = train.drop('cold_load',axis=1)
trainy=train['cold_load']
trainx.shape
trainy.shape

# for test
testx = test.drop('cold_load',axis=1)
testy=test['cold_load']
testx.shape
testy.shape

# build the model - OLS (ordinary least square)

# add a constant term to the trainx and testx
trainx=sm.add_constant(trainx)
testx=sm.add_constant(testx)

trainx.head()
testx.head()

ml=sm.OLS(trainy,trainx).fit()

# summarise the model
ml.summary()

# check the assumptions
#i)mean of error is 0

# predict on the train data and check for residuals

pred1=ml.predict(trainx)
err1=trainy-pred1                   # to check error/residuals
np.mean(err1)

#i) residuals have a constant variance (homoscedasticity)
# lowess -> locally weighted scatterplot smoothing

sns.set(style='whitegrid')
sns.residplot(err1,trainy,lowess=True,color='b')

# hypothesis test to check for heteroscedasticity
# white test

from statsmodels.stats.diagnostic import het_white

hwtest = het_white(ml.resid,ml.model.exog)
print(hwtest)

pvalue=hwtest[1]
print('pvalue of whites test = ', pvalue)

if pvalue <0.05:
    print('model is heteroscedasticity')
else:
    print('model is homoscedasticity')

#iii) error have a normal distrubution
sns.distplot(err1)

# number of rows > columns
energy.shape

 # 5) no outliers
 # checked in EDA using boxplot
    
###################

# actual predictions - on the testing data
pred1 = ml.predict(testx)

# create a dataframe  to store the actual and predicted values
results = pd.DataFrame({'actual_cold_load':testy,'pred_cold_load':np.round(pred1,2)})

print(results)

# individual error and squared error
results['err'] = results['actual_cold_load']-results['pred_cold_load']
results['sqerr']=results['err']**2

# get the SSE model 1
sse1=np.sum(results.sqerr)
mse1 = sse1/len(testy)
print("MSE of model 1=",mse1)



### 14/01/2021

### since model has heteroscedasticity, convert Yinto boxcox transformation and build the next model

#  transform y ('cold_load) into boxcoxY

bc1=spstats.boxcox(energy.cold_load)
print(bc1)

energy.cold_load[0]
bc1[0][0]

# add the boxcox transformed Y to dataset
energy['cold_load_bct'] = bc1[0]
energy
# store the lamda value of boxcox transformation
lamda = bc1[1]
print('box cox transformation lambda =',lamda)

# split the data into train and test to build model 2
train2,test2=train_test_split(energy,test_size=0.3)
train2.columns
test2.columns



# drop the old Y-variable (cold_load)
train2 = train2.drop('cold_load',axis=1)
test2 = test2.drop('cold_load',axis=1)

# split train/test into trainx/y and testx/y
trainx2 = train2.drop('cold_load_bct',axis=1)
trainy2=train2['cold_load_bct']

# for test
testx2 = test2.drop('cold_load_bct',axis=1)
testy2=test2['cold_load_bct']

trainx2.shape
trainy2.shape

testx2.shape
testy2.shape

trainx2.head(2)
trainy2[0:3]

# build model2 on the boxcox transformed y
# add the constant term to build the eq y=a+bnxn
trainx2=sm.add_constant(trainx2)
testx2=sm.add_constant(testx2)



# build OLS model
m2=sm.OLS(trainy2,trainx2).fit()


# hypothesis test to check for heteroscedasticity
# white test

#from statsmodels.stats.diagnostic import het_white

hwtest2 = het_white(m2.resid,m2.model.exog)

pvalue2=hwtest2[1]
print('pvalue of whites test model 2 = ', pvalue2)

if pvalue2 <0.05:
    print('model is heteroscedasticity')
else:
    print('model is homoscedasticity')

# predictions
p2=m2.predict(testx2)
print(p2[0:10])

# convert the predicted values to actual format
acty2 = np.exp(np.log(testy2*lamda+1)/lamda)
acty2

predy2 = np.round(np.exp(np.log(p2*lamda+1)/lamda),2)
predy2

# store the actual y and predicted y in a dataframe
r2=pd.DataFrame({'actual': acty2,'pred': predy2})
r2.head(20)

from sklearn.metrics import mean_squared_error as MSE
mse2 = MSE(r2.actual, r2.pred)
print('MSE of model 2 =',mse2)


#### CLASS ASSIGNMENT:
# perform feature selection on the first model (M1)
# remove features that are not significant
# rebuild and create model m1_1
# compare M1 and M1_1

energy1=energy.drop('cold_load_bct')
col=list(energy.columns)
col
col.remove('cold_load')
col
col.remove('cold_load_bct')
col

# correlation check  (check colinearity)
total=len(col)
total
cor1=energy.iloc[:,0:total].corr()
cor1

# take only the lower tringle to plot the heatmap
cor1=np.tril(cor1)
cor1

# plot the heatmap
sns.heatmap(cor1,xticklabels=col,yticklabels=col,vmin=-1,vmax=1,annot=True,square=True)

col.remove('rel_comp')
col
# split the data into train and test
train,test = train_test_split(energy, test_size=0.3)

train.shape
test.shape

# split the data further into trainx/y and testx/y
trainx = train.drop('cold_load',axis=1)
trainy=train['cold_load']
trainx.shape
trainy.shape

# for test
testx = test.drop('cold_load',axis=1)
testy=test['cold_load']
testx.shape
testy.shape


##########################################

# VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()

vif['factor'] = [variance_inflation_factor(trainx.values,i) for i in range(trainx.shape[1])]

vif['features'] = trainx.columns

print(vif)

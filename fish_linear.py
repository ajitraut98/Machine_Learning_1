# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:36:27 2021

@author: user
"""

### PROJECT_2 ON LINEAR REGRESSION

# load the liabraries

import pandas as pd
import numpy as np
import math
import pylab

# scikit library for linear regression

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import seaborn as sns
import matplotlib.pyplot as plt

# read the data
path= "C:/Users/user/Desktop/Linear Regression using Python/Fish_dataset.csv"

fish = pd.read_csv(path)

#EDA
# dimension
fish.shape

# data types
fish.dtypes

# print the first and last 3 records
fish.head(3)
fish.tail(3)

# get all columns names
fish.columns

# summarise the dataset
desc=fish.describe()
desc

# NULL check
fish.isnull().sum()

# to '0' check
fish[fish==0].count()

fish.columns


# Q1
s = dict(fish.Species.value_counts())
x = list(s.keys())
y = list(s.values())
plt.bar(x,y)
plt.xlabel("Species")
plt.ylabel("Species Count")


# Q2
cols = list(fish.select_dtypes(include=['int32','int64','float32','float64']).columns.values)
cols.remove('Weight')
row = 3; col=3; pos=1
fig = plt.figure()
for c in cols:
    fig.add_subplot(row,col,pos)
    fish.boxplot(column=c,vert=False)
    pos+=1

# boxplot(x, vert=False, showfliers=False, whis = 1.75)

fish = fish.reset_index()
ind = dict(fish.Length1[fish.Length1 >= 56])
indList = list(ind.keys())
fish = fish.drop(indList, axis=0)

fish = fish.drop('index',axis=1)
cor = fish[cols].corr()
cor = np.tril(cor)
sns.heatmap(cor,xticklabels=cols,yticklabels=cols,vmin=-1,vmax=1,annot=True)

fish = fish.drop('Species',axis=1)

train,test = train_test_split(fish, test_size=0.3)

trainx = train.drop('Weight',axis=1)
trainy = train['Weight']

testx = test.drop('Weight',axis=1)
testy = test['Weight']

trainx = sm.add_constant(trainx)
testx = sm.add_constant(testx)

# Q3 
Y = -412.00138929908917 + (113.47040106237036) (Length1) + (-67.95383462644037) (Length2) + (-26.334417922424628) (Length3) + (33.70419458477723) (Height) + (61.739230185687695) (Width)
model = sm.OLS(trainy,trainx).fit()

coef = list(model.params)
print(f'Y = {coef[0]} + ({coef[1]}) ({cols[0]}) + ({coef[2]}) ({cols[1]}) + ({coef[3]}) ({cols[2]}) + ({coef[4]}) ({cols[3]}) + ({coef[5]}) ({cols[4]})')

prediction = model.predict(trainx)
standardError = trainy - prediction
np.mean(standardError)

prediction = model.predict(testx)
results = pd.DataFrame({'Weight':testy,'Predicted Weight':prediction})

results['err'] = results['Weight'] - results['Predicted Weight']
results['sqerr'] = results['err']**2

sse = np.sum(results['sqerr'])

# Q4 mse = 7649.037268575104
mse = sse/len(testy)
mse







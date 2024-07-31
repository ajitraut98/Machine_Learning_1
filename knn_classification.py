# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:01:54 2021

@author: user
"""

## KNN 

# import lib

import pandas as pd
import numpy as np


# standardize the data
from sklearn import preprocessing

from sklearn.model_selection import train_test_split,cross_val_score

# knn algorithms

from sklearn import neighbors

# evaluation
from sklearn.metrics import confusion_matrix,classification_report

# plot the graph to identify the best k
import matplotlib.pyplot as plt
import seaborn as sns

#read the data
path="C:/Users/user/Desktop/wheat.csv"

wheat=pd.read_csv(path)
wheat
wheat.shape


# get the distrubution of the Y variable
wheat['type'].value_counts()

# basic EDA
# null check
wheat.isnull().sum()

# check correlation only for features
cols = list(wheat.columns)
cols
cols.remove('type')

cor = wheat[cols].corr()
cor = np.tril(cor,k=0)
cor

# heatmap to check correlation
sns.heatmap(cor,xticklabels=cols,yticklabels=cols,vmin=-1,vmax=1,square=False,annot=True)

# standardize the dataset

# 1)make a copy of the original data
wheat_std = wheat.copy()

# 2) standardize the copy (minmax) - only the features (x) have to be transformed
minmax = preprocessing.MinMaxScaler()
trdata = minmax.fit_transform(wheat[cols])
wheat_std[cols] = trdata

wheat_std.head(3)
wheat.head(3)

# shuffle the dataset (sice it is grouped by the wheat type)
wheat_std = wheat_std.sample(frac=1)
wheat_std.head(10)

# split the data into trainx/y,testx/y

trainx,testx,trainy,testy = train_test_split(wheat_std.drop('type',axis=1),wheat_std['type'],test_size=0.25)

print('{},{},{},{}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))
wheat_std.shape

# perform cross-validation to determine the best K
nn = range(3,12,2)
print(list(nn))

# store the accuracy of every model created below
cv_accuracy = []

# build model for every value of NN and store its accuracy 
# at the end of the loop, get the maximum accuracy from the list
# this maps to the corresponding value of nearest neighbour in NN

for k in nn:
    # build model with neighbours - K
    model = neighbors.KNeighborsClassifier(n_neighbors=k)
    
    # do a 5-fold cross-validation
    acc = cross_val_score(model,trainx,trainy,cv=5,scoring='accuracy')
    
    # append the mean of accuracy scores to the cv_accuracy
    cv_accuracy.append(acc.mean())

# print the cv_accuracy and get the maximum value
print(cv_accuracy)
max(cv_accuracy)


optk = nn[cv_accuracy.index(max(cv_accuracy))]
print('best neighbour value=',optk)


# additionlly, you can also plot the graph to determine the best K
plt.plot(nn,cv_accuracy,color='red')
plt.xlabel('neighbours')
plt.ylabel('accuracy')
plt.title('cross validation method to determine k')
plt.show()

# build the knn model and predict
m1=neighbors.KNeighborsClassifier(n_neighbors=optk).fit(trainx,trainy)
m1.get_params()

# predict
p1 = m1.predict(testx)
p1
# confusion matrix
confusion_matrix(testy,p1)

testy.value_counts()

# classification report
print(classification_report(testy,p1))
####################################################################################


## assignment
# using the same record ID's of the train and test, build the 2nd model  using the actual data and compare the results

trainx.head(20)
trainx.index

trainx2 = wheat.loc[trainx.index,]
trainx2.head(10)
trainx.head(10)

# this is not required, since Y is the same
'''
trainy2=wheat.loc[trainx.index,'type']
trainy.head(5)
trainy2.head(5)
'''

testx2 = wheat.loc[testx.index,]
testx2.head(10)
testx.head(10)

# build the model with trainx2,trainy,testx2,testy

# calculate the optk for the actual data, since the previous optk value was calculated for the scaled data



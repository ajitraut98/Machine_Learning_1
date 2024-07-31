# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:50:53 2021

@author: user
"""

# LOGISTIC REGRESSION

# (used for binary classification)

#import liab
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# read the path

path = "C:/Users/user/Desktop/bankchurn1.csv"

churn = pd.read_csv(path)

# display the data
churn.head(4)

# drop the features 'custid' and 'surname' cause it is not significant

churn = churn.drop(['custid','surname'],axis=1)
churn

# check the schema
churn.columns

# get the count of y
churn.churn.value_counts()

# plot the count of y-variable
sns.countplot(x='churn',data=churn)

# check the datatypes
churn.dtypes

# split the numeric and factor columns
nc=churn.select_dtypes(exclude='object').columns.values
nc
fc=churn.select_dtypes(include='object').columns.values
fc

# EDA on factor variables
# checking the levels of each factor variable
for c in fc:
    print('factor variable =',c)
    print(churn[c].unique())
    print('\n')


# convert factor to dummies(numbers)

pd.get_dummies(churn.country)


pd.get_dummies(churn.country,drop_first=True)

# make a copy of the original dataset
churn_d = churn.copy()


# convert the factor columns to dummy variables

for f in fc:
    dummy = pd.get_dummies(churn_d[f],drop_first=True,prefix=f)
    churn_d = churn_d.join(dummy)
    
# check the columns of the churn_d dataset
churn_d.columns
churn.columns

# drop the original factor variable
churn_d = churn_d.drop(['country','gender'],axis=1)

churn_d.columns

# check data
churn_d.loc[0]

## EDA FOR NUMERIC (assignment)

# split the data into train and test
train,test = train_test_split(churn_d, test_size=0.3)

train.shape
test.shape
print("train={},test={}".format(train.shape,test.shape))

## split train further into trainx/y 
trainx = train.drop('churn',axis=1)
trainy = train['churn']
print("trainx={},trainy={}".format(trainx.shape,trainy.shape))


# split test further into testx/y 
testx = test.drop('churn',axis=1)
testy = test['churn']
print("testx={},testy={}".format(testx.shape,testy.shape))



# build the base model (logistic regression)
m1 = sm.Logit(trainy,trainx).fit()

# summarise the models
m1.summary()

# prediction
p1 = m1.predict(testx)
p1

# p1 returns the the predictions (as probabilities)
# convert the probabilities into classes 0 and 1 based on the initial
# cutoff(0.5)

# ratio of y in the test data
testy.value_counts()

len(p1[p1<0.5])

len(p1[p1>0.5])


# convert probabilities into classes
# p<0.5 --> 0
# p>0.5 --> 1

# create a copy of the predictions
pred1y = p1.copy()

pred1y[pred1y < 0.5] = 0
pred1y[pred1y > 0.5] = 1

p1[0:5]
pred1y[0:5]


# evalute the model
# i) draw the confusion matrix
# <actual_y, predicted_y>
confusion_matrix(testy,pred1y)

testy.value_counts()
pred1y.value_counts()

# print the classification report
print(classification_report(testy,pred1y))

# AUC for the above model
from sklearn import metrics
fpr,tpr, threshold = metrics.roc_curve(testy,pred1y)

# ROC
roc_auc = metrics.auc(fpr,tpr) # area under the curve value
plt.title("ROC for model 1. AUC = " + str(round(roc_auc,2)))
plt.plot(fpr,tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# reduce the cutoff and check the predictions
pred2y = p1.copy()
pred2y[pred2y < 0.25] = 0
pred2y[pred2y > 0.25] = 1

confusion_matrix(testy,pred2y)

print(classification_report(testy,pred2y))



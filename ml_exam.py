# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:49:42 2021

@author: user
"""
# ml_exam

#import liab

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, neighbors, svm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sts
from sklearn.metrics import confusion_matrix,classification_report


import seaborn as sns
path="C:/Users/user/Desktop/bank.csv"
bank = pd.read_csv(path,sep=';')

#take first five records
bank.head(5)

# shuffling the data
bank.sample(frac=1).head()
'''
#1. Question 1. What does the primary analysis of several categorical features reveal?

 Ans=
    1. from this data we can analyse that there are total 11 categorical features.
    2. In some categorical features there is a 'unknown' value inside them. maybe after checking performance of this dataset we can impute this values.
    3. feature name 'Duration' can highly affect the output. after the call with customer it will became unknown. this feature is only considered for benchmark ourpose. so if it is showing high correlation so have to remove it from dataset.
    4. feature names like Education, Job.  we can merge some values with each other to avoid from creating more dummy variables.
   
#2. Question 2. Perform the following Exploratory Data Analysis tasks:
    1. Missing Value Analysis
    2. Label Encoding wherever required
    3. Selecting important features based on Random Forest
    4. Standardize the data using the anyone of the scalers provided by sklearn
'''
bank.shape

# here we get no. of columns,rows,null values,data-type,shape
bank.info()

# describe the dataset
bank.describe()

# doing Label encoding here on Y variable
le = LabelEncoder()
y_new = le.fit_transform(bank['y'])
bank['y_new']= y_new
bank = bank.drop('y',axis=1)
bank['y']= y_new
bank = bank.drop('y_new',axis=1)
bank.head()

# age, job, marital, default, housing, loan, compain, education, poutcome are the important features which we get from procedure after that we remove them here
bank = bank.drop(['housing','contact','month','day_of_week','duration','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis=1)

# EDA
# creating list of columns
cols = bank.columns
print(len(cols))
cols

# check values count in each column
for c in cols:
    print(c)
    print(bank[c].value_counts())
    print('\n')

# check values in each column
for c in cols:
    print(c)
    print(bank[c].unique())
    print('\n')

# remove the columns who has high % of singularity
cols=list(bank.columns)
cols.remove('y')
for c in cols:
    singularity_value = max(list(bank[c].value_counts()))
    if( singularity_value/bank.shape[0] >= 0.90):
        print('Here drop column: ',c)
        bank=bank.drop(c,axis=1)
bank.shape

# checking the nulls
bank.isnull().sum()
#checking zeroes
bank[bank==0].count()
# # split the numeric and factor columns seperately for data fixing/analysis
numc = bank.select_dtypes(exclude=['object']).columns.values
print(numc)

factc = bank.select_dtypes(include=['object']).columns.values
print(factc)

# Barplot for Y-variable
sns.countplot(x='y',data=bank, palette='hls')
plt.show()

# Correlation
cor = bank[numc].corr()
cor

# Heatmap to see correlation between features
cor = np.tril(cor,k=1)
plt.subplots(figsize=(20,15))
sns.heatmap(cor,cmap="coolwarm",annot=True,xticklabels=numc,yticklabels=numc,vmax=1,vmin=-1,square=False)
# here we can see that there is high correlation between some features(euribor3m,emp.var.rate,nr.employed), so have to drop them

# Creating Dummy Variables
for c in factc:
    dummy=pd.get_dummies(bank[c],drop_first=True,prefix=c)
    bank=bank.join(dummy)

# now remove/drop previous categorical columns
bank = bank.drop(factc, axis=1)

# standardise the dataset
bank_std = bank.copy()
minmax = preprocessing.MinMaxScaler()
bank_std.iloc[:,:] = minmax.fit_transform(bank_std.iloc[:,:])

# set y-variable to original format
bank_std['y'] = bank['y']

cols_std = list(bank_std.columns)

#split data into train & test
train,test = train_test_split(bank_std,test_size=0.3)

# split train data into trainx/trainy & test data into testx/tesy
trainx = train.drop(['y'],axis=1)
trainy = train['y']
testx = test.drop(['y'],axis=1)
testy = test['y']

print('trainx : ',trainx.shape)
print('trainy : ',trainy.shape)
print('testx : ',testx.shape)
print('testy : ',testy.shape)

#corr checkfor allfeatures
cor2 = bank_std.corr()
cor2
cor2 = np.tril(cor2,k=1)
cor2
plt.subplots(figsize=(50,30))
sns.heatmap(cor2,cmap="coolwarm",annot=False,xticklabels=bank_std.columns,
            yticklabels=bank_std.columns,vmax=1,vmin=-1,square=False)

# build the Random Forest Classification model and select important features
m1 = RandomForestClassifier().fit(trainx,trainy)
print(m1)  

# predictions
p1 = m1.predict(testx)
p1
# accuracy of the predictons
print(accuracy_score(testy,p1))
# accuracy =0.8744840980820587

# confusion matrix
confusion_matrix(testy,p1)

testy.value_counts()

# clasification report
print(classification_report(testy,p1))

# important features
imp_f = m1.feature_importances_
indices = np.argsort(imp_f)
imp_f

plt.title("important features")
plt.barh(range(len(indices)),imp_f[indices],color='g',align='center',height=0.8)
plt.yticks(range(len(indices)),[cols_std[i] for i in indices])
plt.xlabel('important feature in percentage')
plt.show()
'''
here we get some important feature, so based on that we can select this particular features and build the models. age, job, marital, default, housing, loan, compain, education, poutcome,
now go to upper line and remove remaining features
'''
'''
 Q3. Build the following Supervised Learning models:
    1. Logistic Regression
    2. AdaBoost
    3.naive bayes
    4. KNN
    5. SVM
'''
#1) logistic regression
m2 = sm.Logit(trainy,trainx).fit()
# summerise the model
print(m2.summary())
# predict
p2=m2.predict(testx)
# convert probabilities to classes (0 & 1)
p2_pred=p2.copy()
#using a cut-off, determine the classses
#take initial cut-off as 0.5 as basic
p2_pred[p2_pred<0.5]=0
p2_pred[p2_pred>0.5]=1
# distribution of predicted classes(to know proportion of classes)
p2_pred.value_counts()
# confusion matrix
print(confusion_matrix(testy,p2_pred))
# classification report
print(classification_report(testy,p2_pred))



'''
# Q4. Tabulate the performance metrics of all the above models
and tell which model performs better in predicting if the
client will subscribe to term deposit or not
'''
















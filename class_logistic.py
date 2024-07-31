# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:27:33 2021

@author: user
"""

### class assesment ON LOGISTIC REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

#read data

path= "C:/Users/user/Desktop/Logistic Regression using Python/xAPI-Edu-Data.csv"

edu=pd.read_csv(path)
edu

edu.shape

edu.isnull().sum()

edu[edu==0].count()

edu.columns
edu=edu.drop(['NationalITy','PlaceofBirth','Topic','Semester'],axis=1)

edu.Class[(edu.Class=='M')] = 'L'
edu.Class.value_counts()

# Q1
columns_list = list(edu.columns)
for i in columns_list:
    countList = list(edu[i].value_counts())
    if (round(countList[0]/480, 2) >= 0.85):
        print(i)
        edu = edu.drop(i, axis=1)

edu_d = edu.copy() 

objectCols = list(edu.select_dtypes(include = ['object']).columns.values)
objectCols.remove('Class')

for i in objectCols:
    dummy = pd.get_dummies(edu_d[i],drop_first=True,prefix=i)
    edu_d = edu_d.join(dummy)

edu_d = edu_d.drop(objectCols, axis=1)

le = LabelEncoder()
edu_d['target'] = pd.Series(le.fit_transform(edu_d['Class']))
edu_d = edu_d.drop('Class',axis=1)

train, test = train_test_split(edu_d, test_size=0.3)

trainx = train.drop('target', axis=1)
trainy = train['target']

testx = test.drop('target', axis=1)
testy = test['target']

cor = trainx.corr()
model = sm.Logit(trainy, trainx).fit()

predicted = model.predict(testx)

predicted[predicted < 0.5]=0
predicted[predicted > 0.5]=1

confusion_matrix(testy, predicted)

testy.value_counts()
predicted.value_counts()

print(classification_report(testy, predicted))

fpr, tpr, threshold = metrics.roc_curve(testy, predicted)

roc_auc = metrics.auc(fpr, tpr) # area under the curve value
plt.title("ROC for model 1. AUC = "+str(round(roc_auc,2)))
plt.plot(fpr, tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


###################################   OR  ################



import pandas as pd
import numpy as np
import math
import pylab

#scikit library for logistic regression
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
import scipy.stats as spstats

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report

#read the data

path= "C:/Users/user/Desktop/Logistic Regression using Python/xAPI-Edu-Data.csv"
edu=pd.read_csv(path)

edu.columns

edu.shape

edu.head(5)
edu.tail(5)

edu.describe()

#checking Nulls
edu.isnull().sum()

#checking zeroes
edu[edu==0].count()

'''
raisedhands               9
VisITedResources          7
AnnouncementsView         10
'''

# singularity check
edu.columns
cols=list(edu.columns)
print(cols)

for c in cols:
    sequence=list(edu[c].value_counts())
    if (round(sequence[0]/480,2)>=0.85):
        edu=edu.drop(c,axis=1)
        
# No singularity in the columns
edu.shape

edu.Class.unique()
edu.Class.value_counts()

# for c in edu:
   
edu=edu.drop(['NationalITy', 'PlaceofBirth','Topic', 'Semester'],axis=1)  
edu.shape
edu.columns

# 3) count plot
edu.dtypes
edu.columns.value_counts()
sns.countplot(x='Class',data=edu,palette='hls')
plt.title("Class-wise Order Distribution")


# Converting Y variable into Binary class
edu.Class[(edu.Class=='L') | (edu.Class=='M') ] = 'M'

edu.Class.unique()
edu.Class.value_counts()


#2) scatter plot

class1=edu.Class
gender=edu.gender
grade=edu.GradeID
plt.scatter(class1,grade)
plt.xlabel("Class")
plt.ylabel("grade")
plt.title("Scatter plot-Class-Gender")

# for numeric data
#corelation check
cols=list(edu.columns)
total = len(cols)
cor = edu.iloc[:,0:total-1].corr()
cor

# take only the lower triangle tp plot the heatmap
cor = np.tril(cor)
cor

#########################################################

numc=edu.select_dtypes(exclude=['object']).columns.values
factc=list(edu.select_dtypes(include=['object']).columns.values)
factc
numc
'''
logg.Class[(logg.Class=='M')]=1
logg.Class[(logg.Class=='H')]=0
pd.series(logg_d)

data
'''

for c in factc:
    print('factor variable=',c)
    print(edu[c].unique())
    print('\n')


# remove y from list
factc.remove('Class')
factc

#Convert the variables into dummy variables
#make a copy of the original var
edu_d=edu.copy()  #Dummy variable dataset


for f in factc:
    dummy=pd.get_dummies(edu_d[f],drop_first=True,prefix=f)
    edu_d=edu_d.join(dummy)
    edu_d=edu_d.drop(f,axis=1)   # Removing  original Y-variable

#check the

# convert Y-variable into y-variable
edu_d['target']=1
edu_d['target'][edu_d['Class']=='H']=0
edu_d['target']=pd.Series(edu_d['target'].astype('int32'))
edu_d=edu_d.drop('Class',axis=1)


# to get random selection of dataset,it is better to shuffle
edu_d=edu_d.sample(frac=1)

# check the columns ofthe edu_d dataset
edu_d.columns


# split the data into train and test
# train and test further into trainx/y and test x/y
train,test=train_test_split(edu_d, test_size=0.3)
print("train={},test={}".format(train.shape,test.shape))

#split train further into trainx/y
trainx= train.drop('target',axis=1)
trainy= train['target']
print("trainx={}, trainy={}".format(trainx.shape,trainy.shape))

#split test further into test x/y
testx =test.drop('target',axis=1)
testy=test['target']
print('testx={}, testy={}'.format(testx.shape,testy.shape))


# Build the base model(logistic regession)
ml=sm.Logit(trainy,trainx).fit()
ml.summary()

# prediction
p1=ml.predict(testx)
p1[0:20]

# p1 returns the predictions (as probabilities)
# convert the probabilities into classes 0 and 1 based on the initial
# cutoff

#ratio of y in the test data
testy.value_counts()

#check count
len(p1[p1<0.5])
len(p1[p1>0.5])

#convert probabilities into classes
#p<0.5 -->0
#p>0..5-->1

#create copy of the predictions
pred1y = p1.copy()

# cutoff and check the predictions
pred1y[pred1y< 0.5]=0
pred1y[pred1y> 0.5]=1

pred1y[0:5]

# evaluate the model
# 1) draw the confusion matrix
# <actual_y,predicted y>
confusion_matrix(testy,pred1y)

testy.value_counts()

pred1y.value_counts()

#print the classification report
print(classification_report(testy,pred1y))

#(for best prediction model both positive  and negative should be high )

# AUC for the abve model
from sklearn import metrics
fpr,tpr,threshold=metrics.roc_curve(testy,pred1y)

# ROC
roc_auc=metrics.auc(fpr,tpr) #area under the curve
plt.title("ROC for model 1. AUC =" + str(round(roc_auc,2)))
plt.plot(fpr,tpr,'b')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# Reduce the cutoff and check the predictions
pred2y= p1.copy()
pred2y[pred2y<0.25]=0
pred2y[pred2y>0.25]=1

confusion_matrix(testy,pred2y)

print(classification_report(testy,pred2y))



























################################################################


'''
Perform the following tasks:
Marks
Q.1
Visualize just the categorical features individually to see what options are included and how each option fares when it comes to count(how many times it appears) and see what can be deduce from that?
[10]
Q.2
Look at some categorical features in relation to each other, to see what insights could be possibly read?
[10]
Q.3
Visualize categorical variables with numerical variables and give conclusions?
[10]
Q.4
From the above result, what are the factors that leads to get low grades of the students?
[20]
Q.5
Build classification model and present it's classification report?
[20]
'''


















































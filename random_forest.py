# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:16:29 2021

@author: user
"""

####### RANDOM FOREST

# random forest 
# dataset - ctg



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

#read the data
path="C:/Users/user/Desktop/ctg/ctg.csv"

ctg=pd.read_csv(path)
ctg


ctg.columns
ctg.head()

# check the distrubution of the y variable
ctg.NSP.value_counts()

# check datatypes
ctg.dtypes

# perform EDA



# there should be no correlation
cols=list(ctg.columns)
cols.remove('NSP')
cols
# correlation matrix

cor = ctg[cols].corr()
cor = np.tril(cor)
cor

# plot the correlation heatmap

sns.heatmap(cor,xticklabels=cols,yticklabels=cols,vmin=-1,vmax=1,annot=True,square=False)
 
sns.set(font_scale=0.6)

# from the heatmap, it can be seen that some features are haaving correlation
# need to remove the correlated variables

# split the data into train,test

trainx,testx,trainy,testy = train_test_split(ctg.drop('NSP',axis=1),ctg['NSP'],test_size=0.3)

print('trainx={},trainy={},testx={},testy={}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))


# build the random Forest Classification model - without hyper-parameter tuning

m1 = RandomForestClassifier().fit(trainx,trainy)
print(m1)
m1.get_params()

# predict
p1 = m1.predict(testx)
p1

# accuracy score
accuracy_score(testy,p1)

# confusion matrix
confusion_matrix(testy,p1)

# classification Report #(recall value obtain from confusion_matrix)
print(classification_report(testy,p1))

# important features
impfeatures = m1.feature_importances_
impfeatures
trainx.columns     # (coresponding values of impfeatures)

len(impfeatures)
len(trainx.columns)

# plot the important features
# it will rank the score on the base on impfeature score
indices = np.argsort(impfeatures)
indices


import matplotlib.pyplot as plt
plt.title('variable importance')
plt.barh(range(len(indices)),impfeatures[indices],color='g',align='center')
plt.yticks(range(len(indices)),[cols[i]for i in indices])
plt.xlabel('relative importance')
plt.show()


############## COMAPARE MODEL WITH DECISION TREE

# random forest regression models
m3 = RandomForestRegressor().fit(trainx,trainy)
p3 = m3.predict(testx)
p3

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


# store the actual and predicted values for both the models dataframe for analysis

df=pd.DataFrame({'actual':testy,
                 'p_OLS':round(p2,2),
                 'p_DT':np.round(p1,2),
                 'p_RF':np.round(p3,2)})


df

#mse of all models
mse_dt=round(mean_squared_error(testy,p1),3)
mse_ols = round(mean_squared_error(testy,p2),3)
mse_rf = round(mean_squared_error(testy,p3),3)
print('Mean squared Errors Comparison\n\tOLS={},\n\tDT={}\n\trf={}'.format(mse_ols,mse_dt,mse_rf))



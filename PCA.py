# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:20:45 2021

@author: user
"""

# PCA (principal component analysis)
# dataset: oil spill

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# can use any scaling technique

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

import matplotlib.pyplot as plt

# read the file
path= "C:/Users/user/Desktop/oil.csv"
dataset=pd.read_csv(path,header=None)

# drop the first column
dataset = dataset.drop(0,axis=1)

# change column names to C1,C2.....
totalcols = len(dataset.columns)
cols = ["C" + str(i) for i in range(1,totalcols+1)]
dataset.columns = cols
dataset

# rename the last column to 'target', since it is the y-variable
dataset=dataset.rename(columns={'C49':'target'})


# split the data into train/test
trainx,testx,trainy,testy=train_test_split(dataset.drop('target',axis=1),
                                           dataset['target'],
                                           test_size=0.3)

print(trainx.shape,trainy.shape,testx.shape,testy.shape)


# apply standard scaling on the trainx and testx only
# y-variables will be as it is
sc=StandardScaler()
tr_trainx = sc.fit_transform(trainx)
tr_testx = sc.fit_transform(testx)

# apply PCA on the scaled data (trainx,testx)
# since, we do not know how many components are required to get the max variance, we will define components as 'None'
# from the results, we can decide on the number of components

pca = PCA(n_components = None)

pca_trainx = pca.fit_transform(tr_trainx)
pca_testx = pca.fit_transform(tr_testx)

# scree plot
explained_variance = pca.explained_variance_ratio_

x=list(range(1,len(explained_variance)+1))
plt.bar(x,explained_variance,color='red')
plt.title('Principal Components')
plt.xlabel('PC')
plt.ylabel('% of Variation')

# based on the plot, we can see that the first 3 PC's contribute roughly 60% variance. Lets try with 3 components
pca = PCA(n_components=3)
pca_trainx = pca.fit_transform(tr_trainx)
pca_testx = pca.fit_transform(tr_testx)
pca.explained_variance_ratio_

# convert the pca data into Pandas dataframes
pca_trainx = pd.DataFrame(pca_trainx)
pca_testx = pd.DataFrame(pca_testx)

## end of PCA .......................

# from here, we can build any classification / regression model (as per the problem statement)

# build a decision tree model on the PCA transformed dataset
m1 = DecisionTreeClassifier().fit(pca_trainx,trainy)
p1 = m1.predict(pca_testx)

# confusion matrix
df=pd.DataFrame({'actual':testy,'predicted':p1})
pd.crosstab(df.actual,df.predicted,margins=True)

# classification report
print(classification_report(testy,p1))
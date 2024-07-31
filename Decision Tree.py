# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:20:40 2021

@author: user
"""
#########  DECISION TREEE ##########################################

# load thre liabraries
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

# plot the decision tree
from sklearn import tree
from IPython.display import Image
import pydotplus
from subprocess import check_call

import numpy as np

# feature selection
from sklearn.feature_selection import RFE



# read the data
path= "C:/Users/user/Downloads/ecoli.csv"

ecoli = pd.read_csv(path)
ecoli
#EDA
# dimension
ecoli.shape

# data types
ecoli.dtypes

# print the first and last 3 records
ecoli.head(3)
ecoli.tail(3)

# to check null
ecoli.info()

# get all columns names
ecoli.columns

# remove unwanted column
ecoli=ecoli.drop('sequence_name',axis=1)
ecoli.columns

# summarise the dataset
desc=ecoli.describe()
desc

# check for singularity
ecoli.chg.value_counts()
ecoli.lip.value_counts()

# since chg and lip are having singularities, they can be removed from the dataset

# first model build with all features
# subsequent model remove singularities and other insignificant features

# distrubution of Y-variables
ecoli.lsp.value_counts()

# shuffle the dataset
ecoli = ecoli.sample(frac=1)   # how much % we have to shuffla 100
ecoli.head

# perform the EDA

# split the data into train and test
train,test = train_test_split(ecoli,test_size=0.3)

print('train={}, test={}'.format(train.shape,test.shape))

train.shape
test.shape

## split train further into trainx/y 
trainx = train.drop('lsp',axis=1)
trainy = train['lsp']
print("trainx={},trainy={}".format(trainx.shape,trainy.shape))


# split test further into testx/y 
testx = test.drop('lsp',axis=1)
testy = test['lsp']
print("testx={},testy={}".format(testx.shape,testy.shape))


# ! pip install pydotplus
# ! conda install python-graphviz
# after installation,include the 'graphviz'
# path in the PATH for it to work
# graphviz is under \anaconda3\library\bin

# BUILD THE MODEL
# 2 types decision model
# i) entropy
# ii) gini

# 1) ENTROPY MODEL

m1_entropy = DecisionTreeClassifier(criterion="entropy").fit(trainx,trainy)

print(m1_entropy)


m1_entropy

trainx
trainy
dir(m1_entropy)


# to know parameter
help(DecisionTreeClassifier)

# plot the decision tree
features = list(ecoli.columns)
features.remove('lsp')
print(features)

classes = ecoli.lsp.unique()
print(classes)

########################3
import pydotplus
from import stringIO

# plot the decision tree
tree.export_graphviz(m1_entropy,'tree1.dot',filled=True,
                     rounded=True,
                     special_characters=True,
                     feature_names=features,
                     class_names=classes)

check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
Image(filename='tree1.png')





########################################

# model2 : entropy with hyper-parameter tuning
# p1-> depth of the tree
# p2-> minimum sample in the leaf

m2_entropy=DecisionTreeClassifier(criterion='entropy',
                                  max_depth=3,
                                  min_samples_leaf=2).fit(trainx,trainy)

# predictions
p2_entropy = m2_entropy.predict(testx)

# accuracy score
accuracy_score(testy,p2_entropy)

# confusion matrix
confusion_matrix(testy,p2_entropy)


##########################################################################

# ASSIGNMENT
# create the next model on criteria = 'gini'
# m3 -> without tuning
# m4 -> with tuning










################################################################################

######## pruning the decisionn tree ########################

# to prune the tree (of any DT model) based on the complexity parameter value

path=m1_entropy.cost_complexity_pruning_path(trainx,trainy)
ccp_alphas = path.ccp_alphas

# ccp_alphas store the cost complexity pruning values

# for every ccp_alpha value,find the best ccp_alpha value that can give the best results

results = []
for cp in ccp_alphas:
    model = DecisionTreeClassifier(ccp_alpha=cp).fit(trainx,trainy)
    results.append(model)

results
# get the accuracy scores for the training and test data for every cp

train_scores=[r.score(trainx,trainy) for r in results]
test_scores=[r.score(testx,testy) for r in results]

# plot the 2 models and check the diffrence between the results

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs alpha for training and testing")
ax.plot(ccp_alphas,train_scores,marker="o",label='train',drawstyle='steps-post')
ax.plot(ccp_alphas,test_scores,marker="o",label='test',drawstyle='steps-post')
ax.legend()
plt.show()


# based on the chart, the cost complexity parameter(ccp) value is slightly more than 0
ccp = 0.009
model1 = DecisionTreeClassifier(ccp_alpha=ccp).fit(trainx,trainy)
pred1 = model1.predict(testx)

confusion_matrix(testy,pred1)
print(classification_report(testy,pred1))




# feature selection
# Recursive feature elimination (RFE)
cols=list(testx.columns)
rfe = RFE(m1_entropy, len(testx.columns)).fit(testx,testy)
support = rfe.support_
ranking = rfe.ranking_


# create the dataframe to store the featura importance
df_rfe = pd.DataFrame({"columns":cols,"support":support,"ranking":ranking})


# sort and print the best features by rank 
print(df_rfe.sort_values('ranking'))


################################














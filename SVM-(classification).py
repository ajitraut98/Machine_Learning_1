# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:11:29 2021

@author: user
"""
# SVM classification
# dataset: diabetes

# import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# read the input data
path="C:/Users/user/Desktop/diab.csv"
diab=pd.read_csv(path)

diab.shape
diab.columns
# EDA (impute columns that have nulls/0)

# remove feature 'class_val'
diab=diab.drop('class_val',axis=1)
diab.head()

# standardize the dataset
diab_std = diab.copy()

minmax=preprocessing.MinMaxScaler()
diab_std.iloc[:,:]=minmax.fit_transform(diab_std.iloc[:,:])

# reset the Y-variable to the original value
diab_std['class'] = diab['class']

# split dataset (75%-25%)
trainx,testx,trainy,testy=train_test_split(diab_std.drop('class',axis=1),
                                           diab_std['class'],
                                           test_size=0.25)
print(trainx.shape,trainy.shape,testx.shape,testy.shape)


# build the different models based on kernels
# kernels = ['linear', 'LinearSVC', 'poly', 'rbf','sigmoid']
#             <-- only C -------->   <---- C and Gamma --->

# do cross-validation to determine the best C
lov_c = range(1,11)
list(lov_c)

# store the cross-validation scores for every C
cv_scores=[]

for c in lov_c:
    model=svm.SVC(kernel="linear",C=c)
    err=cross_val_score(model,trainx,trainy,cv=5,scoring='accuracy')
    cv_scores.append(round(err.mean(),3))
    
print(cv_scores)    

# best C is determined by finding the maximum CV accuracy
bestc = lov_c[cv_scores.index(max(cv_scores))]    
print("best C = ", bestc)    


# model 1: kernel = 'linear'
m1=svm.SVC(kernel='linear',C=bestc).fit(trainx,trainy)
p1=m1.predict(testx)

# to get a detailed confusion matrix (using crosstab)
df=pd.DataFrame({'actual':testy, 'predicted':p1})
pd.crosstab(df.actual,df.predicted,margins=True)

confusion_matrix(testy,p1)

# classification report
print(classification_report(testy,p1))

def buildModelEvalParams(cr,ker,acc):
    cr=cr.drop(['accuracy','macro avg','weighted avg'],axis=1)
    cr=cr.drop(['support'],axis=0)
    cr=cr.T
    cr['model'] = ker
    cr['accuracy']= acc
    cr['class'] = cr.index
    
    return(cr)

# create the model evaluation parameters
cr1=pd.DataFrame(classification_report(testy,p1,output_dict=True))
cr1=buildModelEvalParams(cr1,'linear',accuracy_score(testy,p1))
cr1

# model 2: linearSVC
m2=svm.LinearSVC(C=bestc).fit(trainx,trainy)
p2=m2.predict(testx)

# confusion matrix
df=pd.DataFrame({'actual':testy,'predicted':p2})
pd.crosstab(df.actual,df.predicted,margins=True)

# classification report
print(classification_report(testy,p2))

cr2=pd.DataFrame(classification_report(testy,p2,output_dict=True))
cr2=buildModelEvalParams(cr2,'linearSVC',accuracy_score(testy,p2))
cr2

# from here, the 3 kernels need 2 parameters: C and Gamma
# lov_c: already configured
# 10 equally spaced values between 0.01 and 1
lov_g = np.linspace(0.01,1,10)
lov_g

# to store the accuracy for every C_G combination
cv_scores=[]

# for every iteration, store the C_G combination
cg_details=[]

for c in lov_c:
    for g in lov_g:
        model = svm.SVC(kernel='rbf',C=c,gamma=g)
        scores = cross_val_score(model,trainx,trainy,cv=5,scoring='accuracy')
        cv_scores.append(round(scores.mean(),2))
        cg_details.append(str(c)+":"+str(g))

print(cg_details)
        
# find the index of the max accuracy that will give the best C_G combination
cg = cg_details[cv_scores.index(max(cv_scores))]
bestc = int(cg.split(":")[0])
bestg = float(cg.split(":")[1])

print("bestC=",bestc,"bestG=",bestg)

# model 3 - RBF
m3=svm.SVC(kernel='rbf',C=bestc,gamma=bestg).fit(trainx,trainy)
p3=m3.predict(testx)

# confusion matrix
df=pd.DataFrame({'actual':testy,'predicted':p3})
pd.crosstab(df.actual,df.predicted,margins=True)

# classification report
cr3=pd.DataFrame(classification_report(testy,p3,output_dict=True))
cr3=buildModelEvalParams(cr3,'RBF',accuracy_score(testy,p3))
cr3

# model 4: kernel = poly
m4=svm.SVC(kernel='poly',C=bestc,gamma=bestg).fit(trainx,trainy)
p4=m4.predict(testx)
cr4=pd.DataFrame(classification_report(testy,p4,output_dict=True))
cr4=buildModelEvalParams(cr4,'polynomial',accuracy_score(testy,p4))
cr4

# model 5: kernel = sigmoid
m5=svm.SVC(kernel='sigmoid',C=bestc,gamma=bestg).fit(trainx,trainy)
p5=m5.predict(testx)
cr5=pd.DataFrame(classification_report(testy,p5,output_dict=True))
cr5=buildModelEvalParams(cr5,'sigmoid',accuracy_score(testy,p5))
cr5

# store the results of all models in a dataframe for evaluation
res = pd.DataFrame()
res = res.append([cr1,cr2,cr3,cr4,cr5],ignore_index=True)
res


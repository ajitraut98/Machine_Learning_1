# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:57:49 2021

@author: user
"""
### PROJECT ON LOGISTIC REGRESSION

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#read data

path= "C:/Users/user/Desktop/Vaccine Usage Prediction/Dataset/h1n1_vaccine_prediction.csv"

h1n1=pd.read_csv(path)
h1n1

h1n1.columns.shape
h1n1.columns

# TO CHECK NULL

nulls=h1n1.isnull().sum()

h1n1.h1n1_worry.isnull().sum()

h1n1.h1n1_worry[h1n1.h1n1_worry.isnull()]=0

h1n1.h1n1_worry.isnull().sum()

h1n1.h1n1_awareness.value_counts()

h1n1.h1n1_awareness[h1n1.h1n1_awareness.isnull()]=0

h1n1.h1n1_awareness.value_counts()

h1n1.antiviral_medication[h1n1.antiviral_medication.isnull()]=0

h1n1.contact_avoidance[h1n1.contact_avoidance.isnull()]=0

h1n1.bought_face_mask[h1n1.bought_face_mask.isnull()]=0

h1n1.wash_hands_frequently[h1n1.wash_hands_frequently.isnull()]=0  

h1n1.avoid_large_gatherings.value_counts()
h1n1.avoid_large_gatherings[h1n1.avoid_large_gatherings.isnull()]=0
h1n1.reduced_outside_home_cont[h1n1.reduced_outside_home_cont.isnull()]=0
h1n1.avoid_touch_face[h1n1.avoid_touch_face.isnull()]=0
h1n1.avoid_touch_face.isnull().sum()
h1n1.dr_recc_h1n1_vacc[h1n1.dr_recc_h1n1_vacc.isnull()]=0
h1n1.dr_recc_seasonal_vacc[h1n1.dr_recc_seasonal_vacc.isnull()]=0
h1n1.chronic_medic_condition[h1n1.chronic_medic_condition.isnull()]=0
h1n1.cont_child_undr_6_mnths[h1n1.cont_child_undr_6_mnths.isnull()]=0
h1n1.is_health_worker[h1n1.is_health_worker.isnull()]=0
h1n1.has_health_insur[h1n1.has_health_insur.isnull()]=0
h1n1.is_h1n1_vacc_effective.mode()
h1n1.is_h1n1_vacc_effective[h1n1.is_h1n1_vacc_effective.isnull()]=4
h1n1.is_h1n1_vacc_effective.isnull().sum()
h1n1.is_h1n1_risky[h1n1.is_h1n1_risky.isnull()]=2
h1n1.sick_from_h1n1_vacc[h1n1.sick_from_h1n1_vacc.isnull()]=2
h1n1.is_seas_vacc_effective[h1n1.is_seas_vacc_effective.isnull()]=4
h1n1.is_seas_risky[h1n1.is_seas_risky.isnull()]=2
h1n1.sick_from_seas_vacc[h1n1.sick_from_seas_vacc.isnull()]=1
h1n1.qualification[h1n1.qualification.isnull()]='College Graduate'
h1n1.qualification.isnull().sum()
h1n1.income_level.mode()
h1n1.income_level[h1n1.income_level.isnull()]='<= $75,000, Above Poverty'
h1n1.income_level.isnull().sum()
h1n1.marital_status.mode()
h1n1.marital_status[h1n1.marital_status.isnull()]='Married'
h1n1.housing_status[h1n1.housing_status.isnull()]='Own'
h1n1.employment[h1n1.employment.isnull()]='Employed'
h1n1.no_of_adults[h1n1.no_of_adults.isnull()]= 1
h1n1.no_of_children[h1n1.no_of_children.isnull()]=0

#convert data into list to remove the (u id & h1n1 vaccine)
list1=list(h1n1.columns)
list1
list1.remove('unique_id')
list1.remove('h1n1_vaccine')
list1
h1n1.columns.shape

#for loop for the singularity testing
for i in list1:
    sequence=list(h1n1[i].value_counts())
    if(round(sequence[0]/26707,2) >=0.85):
        h1n1=h1n1.drop(i,axis=1)

#distribute data into 2 list one numc and func
numc=h1n1.select_dtypes(exclude='object').columns.values
func=h1n1.select_dtypes(include='object').columns.values
print(func)

h1n1.qualification.value_counts()
h1n1.shape
h1n1.h1n1_vaccine.value_counts()


#from numc remove the unwanted y and uid columns
numc=list(numc)
numc.remove('unique_id')
numc.remove('h1n1_vaccine')

# check correlation
cor = h1n1[numc].corr()
cor
sns.heatmap(cor,xticklabels=numc,yticklabels=numc,
            vmin=-1,vmax=1,
            square=True)

#removing the upper triangle
cor=np.tril(cor)

h1n1.columns
# remove the highly positive correlated and highly negative co related
# columns
h1n1=h1n1.drop(['unique_id','h1n1_worry','is_seas_vacc_effective','is_seas_risky','sick_from_seas_vacc',
           'reduced_outside_home_cont','dr_recc_h1n1_vacc','avoid_touch_face'],axis=1)
h1n1.columns
numc
func

#create copy of the dataframe
h1n1_d = h1n1.copy()
h1n1_d

#put dummy in factor features
pd.get_dummies

#inserting dummy variable in factor columns and joining then to the
#copy of data
for f in func:
    dummy = pd.get_dummies(h1n1_d[f],drop_first=True,prefix=f)
    h1n1_d = h1n1_d.join(dummy)

h1n1_d.columns.shape
h1n1_d.dtypes

#drop the original factor columns
h1n1_d = h1n1_d.drop(['age_bracket', 'qualification', 'race' ,'sex' ,'income_level',
                 'marital_status' ,'housing_status', 'employment' ,'census_msa'],axis=1)

h1n1_d.columns

h1n1_d['age_bracket_35 - 44 Years'].value_counts()
h1n1_d.loc[0]



#split the data into train and test
#furthur into trainx trainy ,testx,trainx
train,test=train_test_split(h1n1_d,test_size=0.3)

trainx=train.drop('h1n1_vaccine',axis=1)
trainy=train['h1n1_vaccine']
print('trainx={}, trainy={}'.format(trainx.shape,trainy.shape))

testx = test.drop('h1n1_vaccine',axis=1)
testy= test['h1n1_vaccine']
print('testx={},testy={}'.format(testx.shape,testy.shape))

#build the base model (logistic regression)
m1 = sm.Logit(trainy,trainx).fit()

#summarise the model
m1.summary()

#predection
p1=m1.predict(testx)
p1

# convert predictions to classes

pred1=p1.copy()
testy.value_counts()


pred1[pred1<0.32] = 0
pred1[pred1>0.32] = 1
confusion_matrix(testy,pred1)
print(classification_report(testy,pred1))

#making model 2 as the h1n1 awareness and income are in significant
#remove insignificant data from h1n1_d
 
h1n1_d2=h1n1_d.copy()
h1n1_d2
h1n1_d2=h1n1_d2.drop(['h1n1_awareness','income_level_> $75,000','income_level_Below Poverty'],axis=1)
h1n1_d2
h1n1_d2.columns.shape

train,test=train_test_split(h1n1_d2,test_size=0.3)

trainx=train.drop('h1n1_vaccine',axis=1)
trainy=train['h1n1_vaccine']
testx=test.drop('h1n1_vaccine',axis=1)
testy=test['h1n1_vaccine']
print('trainx={},trainy={},testx{},testy{}'.format(trainx.shape,trainy.shape,testx.shape,testy.shape))

m2 = sm.Logit(trainy,trainx).fit()
print(m2)
m2.summary()

p2=m2.predict(testx)
p2
pred2=p2.copy()
testy.value_counts()


#reduce the cut off and check the predication till positive comes near
#to 1.0

pred2=p2.copy()
pred2[pred2<0.23] = 0
pred2[pred2>0.23] = 1
confusion_matrix(testy,pred2)
print(classification_report(testy,pred2))



 
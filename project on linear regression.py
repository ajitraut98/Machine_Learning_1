# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:39:42 2021

@author: user
"""
### PROJECT ON LINEAR REGRESSION

# load thre liabraries
import pandas as pd
import numpy as np
import math
import pylab
dir(pd)
# scikit library for linear regression

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split
import scipy.stats as spstats
import seaborn as sns
import matplotlib.pyplot as plt

# read the data
path= "C:/Users/user/Desktop/Dataset/HousePrices.csv"

houseprice = pd.read_csv(path)

#EDA
# dimension
houseprice.shape

# data types
houseprice.dtypes

# print the first and last 3 records
houseprice.head(3)
houseprice.tail(3)

# get all columns names
houseprice.columns

# summarise the dataset
desc=houseprice.describe()
desc
# NULL check
api=houseprice.isnull().sum()
api
houseprice.GarageArea.isnull().sum()
houseprice.GarageType.isnull().sum()
houseprice.GarageCars.isnull().sum()
houseprice.GarageQual.isnull().sum()
houseprice.GarageQual.isnull().sum()

# setting the garage related columns from NULL to "No"
# since there is no garage for the given rows
cols = ['GarageType','GarageFinish','GarageQual','GarageQual','GarageYrBlt']

for c in cols:
    houseprice[c][houseprice[c].isnull()] = "no"

# to make lotfrontage "0"
houseprice.LotFrontage[houseprice.LotFrontage.isnull()] = 0

# split the dataset into numeric and factor columns

numc=houseprice.select_dtypes(include=['int32','int64']).columns.values
numc=houseprice.select_dtypes(include=['object']).columns.values
numc

numc=houseprice.select_dtypes(include=['float32','float64']).columns.values
numc

# to check null values in Alley
ap=houseprice.Alley.isnull().sum()
ap

# columns from NULL to "No" for Alley
houseprice.Alley[houseprice.Alley.isnull()] = 'no'

# MasVnrType
houseprice.MasVnrType.isnull().sum()

# columns from NULL to "No" for MasVnrType
houseprice.MasVnrType[houseprice.MasVnrType.isnull()] = 'no'

# MasVnrArea
houseprice.MasVnrArea.isnull().sum()

# TO CHECK "0" IN MasVnrArea
houseprice.MasVnrArea[houseprice.MasVnrArea==0].count()

houseprice.MasVnrArea[houseprice.MasVnrArea.isnull()] = 0

## BsmtQual convert NA TO NO
houseprice.BsmtQual.isnull().sum()
houseprice.BsmtQual[houseprice.BsmtQual.isnull()] = 'no'

# BsmtCond convert NA TO NO
houseprice.BsmtCond.isnull().sum()
houseprice.BsmtCond[houseprice.BsmtCond.isnull()] = 'no'

# BsmtExposure
houseprice.BsmtExposure.isnull().sum()
houseprice.BsmtExposure[houseprice.BsmtExposure.isnull()] = 'no'

# BsmtFinType1
houseprice.BsmtFinType1.isnull().sum()
houseprice.BsmtFinType1[houseprice.BsmtFinType1.isnull()] = 'no'

# FireplaceQu
houseprice.FireplaceQu.isnull().sum()
houseprice.FireplaceQu[houseprice.FireplaceQu.isnull()] = "NO"

# PoolQC
houseprice.PoolQC.isnull().sum()
houseprice.PoolQC[houseprice.PoolQC.isnull()] = "NO"

# Fence
houseprice.Fence.isnull().sum()
houseprice.Fence[houseprice.Fence.isnull()] = "No Fence"

# MiscFeature
houseprice.MiscFeature.isnull().sum()
houseprice.MiscFeature[houseprice.MiscFeature.isnull()] = "NO"

# Electrical
houseprice.Electrical.isnull().sum()
houseprice.Electrical[houseprice.Electrical.isnull()] = "Mix"



# final null check
nullcheck = houseprice.isnull().sum()
nullcheck

#singularity

houseprice.columns
cols=list(houseprice.columns)
print(cols)

#remove one column
houseprice = houseprice.drop(['Id'],axis=1)
houseprice

#select rows and columns
houseprice.iloc[0:50,70:75]

houseprice.Zone_Class.value_counts()

#remove multiple column

houseprice = houseprice.drop(['Alley','LandContour','Road_Type','Utilities','Condition1','Condition2','Dwelling_Type','RoofMatl','ExterCond','BsmtCond','BsmtFinType2','BsmtFinSF2','Heating','CentralAir','LowQualFinSF','KitchenAbvGr','GarageQual','GarageCond','PavedDrive','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','MiscFeature','MiscVal','SaleType'],axis=1)


houseprice.shape

len(houseprice)      # len shows only total no. of rows
houseprice.columns

# check for outliers

row = 3; col=3; pos=1
fig = plt.figure()
for c in cols:
    fig.add_subplot(row,col,pos)
    houseprice.boxplot(column=c,vert=False)
    pos+=1

## split the data into numeric and factors columns

num = list(houseprice.select_dtypes(include=['int32','int64','float32','float64']).columns.values)
num
len(num)

num.remove('Property_Sale_Price')
num
len(num)


# ANOVA Test


# method 1
# we do ANOVA here, coz to check given factor is significant or not
# ANOVA for Zone_Class  (do not run just for understanding )
m1 = ols('Property_Sale_Price ~ Zone_Class', data=houseprice).fit()
s = sm.stats.anova_lm(m1, typ=2)

# predefine pcritical value 
pcritical = 0.05 

# pcalculated value
pvalue = s["PR(>F)"][0]

# 
if pvalue < pcritical :
    print("H0 reject")
    print("It is significant i.e variation are available for Zone_Class")
    print("this column not drop")
else:
    print("Fail To reject H0")
    print("It is not significant i.e variation are not available for Zone_Class")
    print("this column drop")
    housePrices = houseprice.drop('Zone_Class', axis=1)
OR
## ANOVA
colsObject = list(houseprice.select_dtypes(include=['object']).columns.values)
colsObject
for i in colsObject:
    model = ols('Property_Sale_Price ~ '+i+' ', data=houseprice).fit()
    anovaValue = sm.stats.anova_lm(model, typ=2)
    pcritical = 0.05
    pvalue = anovaValue["PR(>F)"][0]
    if pvalue > pcritical:
        houseprice = houseprice.drop(i,axis=1)

houseprice.shape


# correlation check  (check colinearity)
total= houseprice[num].corr()
total



# take only the lower tringle to plot the heatmap
cor1=np.tril(total)
cor1

# plot the heatmap   # here annot=false means graph only with colour notations
sns.heatmap(cor1,xticklabels=total,yticklabels=total,vmin=-1,vmax=1,annot=False,square=True)

houseprice.columns

# Drop the highly corelated columns from the dataset

houseprice=houseprice.drop(['BsmtFinSF1'],axis=1)
houseprice=houseprice.drop(['BsmtFullBath'],axis=1)
houseprice=houseprice.drop(['TotalBsmtSF'],axis=1)
houseprice=houseprice.drop(['TotRmsAbvGrd'],axis=1)
houseprice=houseprice.drop(['GarageArea'],axis=1)
houseprice=houseprice.drop(['2ndFlrSF'],axis=1)
houseprice=houseprice.drop(['GrLivArea'],axis=1)

houseprice.shape

#EDA on factor variables
#Checking the levels of each factor variable

fc1=houseprice.select_dtypes(include='object')
fc1.columns
for c in fc1:
    print('factor variable=',c)
    print(fc1[c].unique())
    print('\n')

# generate dummies for the
pd.get_dummies
houseprice.columns
houseprice_2=houseprice.copy()
houseprice_2.shape
houseprice_2.columns

for f in fc1:
    dummy=pd.get_dummies(houseprice_2[f],drop_first=True,prefix=f)
    houseprice_2=houseprice_2.join(dummy)

#check the columns of the houseprice_2 dataset
houseprice_2.columns

# check the columns of the houseprice dataset
houseprice.columns

#drop the original factor variable
houseprice_2= houseprice_2.drop(['Zone_Class', 'Property_Shape', 'LotConfig', 'Neighborhood',
       'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
       'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
       'GarageType', 'GarageYrBlt', 'GarageFinish', 'Fence', 'SaleCondition'],axis=1)

houseprice_2.columns
houseprice_2.head(5)
houseprice_2.shape

#split the data into train and test  data
train,test=train_test_split(houseprice_2,test_size=0.3)

train.shape
test.shape

#split the data further into trainx/y and testx/y
#train split
trainx=train.drop('Property_Sale_Price',axis=1) #Y-variable drop fron trainx
trainy=train['Property_Sale_Price']# Add this y -variable into train y

trainx.shape
trainy.shape

#test split
testx=test.drop('Property_Sale_Price',axis=1)
testy=test['Property_Sale_Price']
testx.shape
testy.shape

trainx.head()
trainy.head()

#build the model -ols (ordinary least square)
#add a constant term to the trainx and testx
trainx=sm.add_constant(trainx)
testx=sm.add_constant(testx)




#machinelearning model build
ml=sm.OLS(trainy,trainx).fit()
ml

trainx.dtypes

#summarise the model
ml.summary()

# check the assumptions
# mean of error is 0
# predict on the train data and check for residuals

pred1= ml.predict(trainx)
err1=trainy-pred1 #(actual- prediction)

np.mean(err1)

#residuals have a constant variance (homoscedasticity)
sns.set(style='whitegrid')
sns.residplot(err1,trainy,lowess=True,color='b')

#hypothesis test to check for Heteroscedasticity
#whites test
'''
from statsmodels.stats.diagnostic import het_white
hwtest= het_white(ml.resid, ml.model.exog)
print(hwtest)


pvalue=hwtest[1]
print('pvalueof Whites Test=',pvalue)

if pvalue<0.05:
    print('model is heteroscedastic')
else:
    print('model is homoscedastic')
'''  
# 3)errors have normal distribtion
sns.distplot(err1)


pred1=ml.predict(testx)
pred1

## create a dataframe to store the actual and predicted values
results = pd.DataFrame({'Actual_property_sale_price':testy,'pred_property_sale_price':np.round(pred1,2)})

print(results)

results['err']= results['Actual_property_sale _price']-results['pred_property_sale_price']
results['sqerr']=results['err']**2


# get the SSE of model 1

sse1=np.sum(results.sqerr)
mse1=sse1/len(testy)
print("SSE of model 1 =",sse1)
print("MSE of Model 1=",mse1)
######## since the model has heteroscedasticity, convert Y into BoxCox transformation and built the next model

#transform Y into boxcox Y
bc1= spstats.boxcox(houseprice_2.Property_Sale_Price)
print(bc1)

houseprice_2.Property_Sale_Price[0]    # to check
bc1[0][0]

# add the boxcox transformed Y to dataset
houseprice_2['Property_Sale_Price_bct']=bc1[0]

#store the  lambda value of boxcox into actual Y format
lamda= bc1[1]
print('boxcox transformation lambda= ',lamda)

######################################################################

#split the data into train and test to build model 2
train2,test2 =train_test_split(houseprice_2,test_size=0.3) # 70% and 30% split data

train2.columns
test2.columns

#remove old Y-variable (cold_load)
train2=train2.drop('Property_Sale_Price',axis=1) #here axis1 means columnwise
test2=test2.drop('Property_Sale_Price',axis=1)


#split the data further into trainx/y and testx/y
#train split
trainx2=train2.drop('Property_Sale_Price_bct',axis=1)
trainy2=train2['Property_Sale_Price_bct']
# dimensions
trainx2.shape
trainy2.shape

#test split
testx2=test2.drop('Property_Sale_Price_bct',axis=1)
testy2=test2['Property_Sale_Price_bct']
# dimensions
testx2.shape
testy2.shape

trainx2.head(2)
trainy2[0:3]


#build model 2 on the Boxcox transformed Y

#add the constant term to build the eq (y=a+bx*n)

#build the model -ols(ordinary least square)
#add aconstant term to the trainx and testx
trainx2=sm.add_constant(trainx2)
testx2=sm.add_constant(testx2)

trainx2.head(2)

ml2=sm.OLS(trainy2,trainx2).fit()

#summarise the model
ml2.summary()

#Hypothesis test to check for Heteroscedasticity
'''
#whites test

#from statsmodels.stats.diagnostic import het_white
hwtest2= het_white(ml2.resid, ml2.model.exog)
print(hwtest)


pvalue2=hwtest2[1]
print('pvalueof Whites Test=',pvalue2)

if pvalue2<0.05:
    print('model is heteroscedastic')
else:
    print('model is homoscedastic')
'''
   
#prediction
p2=ml2.predict(testx2)    
print(p2[0:10])


#convert the pridicted values into actual format
acty2= np.exp(np.log(testy2*lamda+1)/lamda)
predy2=np.round(np.exp(np.log(p2*lamda+1)/lamda),2)

# store the actual Y and predicted Y in a dataframe
r2=pd.DataFrame({'Actual':acty2,'pred':predy2})


r2.head(20)

from sklearn.metrics import mean_squared_error as MSE
mse2= MSE(r2.Actual, r2.pred)
print('MSE OF MODEL 2 =',mse2)


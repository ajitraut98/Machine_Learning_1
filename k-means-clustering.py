# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:38:07 2021

@author: user
"""
# K-MEANS CLUSTERING
# dataset- mall customers


# import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 

import matplotlib.pyplot as plt

# read the data
path= "C:/Users/user/Desktop/mall.csv"
mall = pd.read_csv(path)
mall.columns

# rename the columns

mall=mall.rename(columns={'Annual Income (k$)':'income','Spending Score (1-100)':'score'})
mall

# take columns 'income' and 'score' for clustering
X = np.array(mall[['income','score']].values)
X
# perform the cross-validation to determine the best K for K-Means

# to store all the WCSS (within cluster sum of aquares) for each cluster
wcss=[]

# list of clusters from 1-10
lc = range(1,11)
lc
for c in lc:
    model=KMeans(n_clusters=c, init="k-means++",max_iter=50).fit(X)
    
    # WCSS
    wcss.append(model.inertia_)
    
# plot the graph
plt.plot(lc,wcss,color='r',marker='o')
plt.title('Clusters vs WCSS')
plt.xlabel('Clusters')
plt.ylabel('WCSS')    

# based on the graph, the best value for K is 5
# build model with K=5 and associate the data points to the respective cluster
# plot the clusters

optK = 5

# model and finding clusters
clusters = KMeans(n_clusters=optK,init='k-means++',max_iter=50).fit_predict(X)

print(clusters)

# update the dataframe with the clusters
mall['cluster'] = clusters

# visualise all the clusters

plt.scatter(mall.income[mall.cluster==0],mall.score[mall.cluster==0],
            s=100,c='violet',label='Cluster 1')

plt.scatter(mall.income[mall.cluster==1],mall.score[mall.cluster==1],
            s=100,c='blue',label='Cluster 2')

plt.scatter(mall.income[mall.cluster==2],mall.score[mall.cluster==2],
            s=100,c='green',label='Cluster 3')

plt.scatter(mall.income[mall.cluster==3],mall.score[mall.cluster==3],
            s=100,c='black',label='Cluster 4')

plt.scatter(mall.income[mall.cluster==4],mall.score[mall.cluster==4],
            s=100,c='red',label='Cluster 5')

plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()
plt.show()


# get the records cluster-wise

# cluster 1
mall[mall.cluster==0] 
mall[mall.cluster==1]
mall[mall.cluster==2]

















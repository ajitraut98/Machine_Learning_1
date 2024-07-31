# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:15:20 2021

@author: user
"""

# Hierarchical / Agglomerative clustering

# dataset: Mall 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dendrogram
import scipy.cluster.hierarchy as sch

# hierarchical clustering library
from sklearn.cluster import AgglomerativeClustering

# read the file
path= "C:/Users/user/Desktop/mall.csv"

dataset=pd.read_csv(path)

dataset

# rename the columns
dataset=dataset.rename(columns={'Annual Income (k$)':'income', 'Spending Score (1-100)':'score'})

# for clustering, take 'income' and 'score' as input columns
X = dataset[['income','score']].values
X

# plot the dendrogram
# method='ward' -> minimises variance within clusters
dendro = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# based on the dendrogram, the best clusters is between 4-6

# lets start with clusters = 6

# create the clusters for the dataset
clusters=AgglomerativeClustering(n_clusters=4,
        affinity='euclidean',linkage='ward',
        ).fit_predict(X)

np.unique(clusters)

dataset = dataset.drop('cluster',axis=1)
# copy the clusters to the dataset
dataset['cluster'] = clusters 

print(dataset)


# visualise the clusters
plt.scatter(dataset.income[dataset.cluster==0],dataset.score[dataset.cluster==0],s=40,c='violet',label='C1')

plt.scatter(dataset.income[dataset.cluster==1],dataset.score[dataset.cluster==1],s=40,c='blue',label='C2')

plt.scatter(dataset.income[dataset.cluster==2],dataset.score[dataset.cluster==2],s=40,c='green',label='C3')

plt.scatter(dataset.income[dataset.cluster==3],dataset.score[dataset.cluster==3],s=40,c='yellow',label='C4')

# plt.scatter(dataset.income[dataset.cluster==4],dataset.score[dataset.cluster==4],s=40,c='orange',label='C5')

#plt.scatter(dataset.income[dataset.cluster==5],dataset.score[dataset.cluster==5],s=40,c='red',label='C6')

plt.title('Customer Clustering')
plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()

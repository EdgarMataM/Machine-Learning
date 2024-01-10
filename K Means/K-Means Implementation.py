# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:12:49 2021

@author: Edgar Mata
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import make_blobs

#Generating 300 points and saving them in X; Y saves the labels.
X,Y = make_blobs(n_samples=300,centers=3,cluster_std=1,random_state=3)
#Taking 3 random centroids.
centers = np.array([X[0],X[4],X[10]])
colour = ['green','blue','red'] #List for providing colour to clusters.

# Plotting the 300 points.
plt.scatter(X[:,0], X[:,1], s=50, c=np.take(colour,Y))
# Plotting initial centers.
plt.scatter(centers[:,0], centers[:,1], s=100, c='black')


# This function gets the distance from each randmon initial point to a cluster and saves it in a matrix.
def DistanceMatrix(centers,X):
    dist = []
    distM = []
    for i in range(len(X)):
        for j in range(3):
            dist.append(math.dist(X[i],centers[j])) # Geting the distance from i-point to j-cluster.
        distM.append(dist)
        dist = []
    binary=np.array(distM) # Converting into numpy array for easier manipulation.
    return binary

# This function returns a matrix which indicates the cluster of each point, i.e., gives the labels.
def Membership(distances):
    membership = []
    gamma = []
    minimum = []
    for i in range(len(distances)):
        minimum.append(min(distances[i])) # Taking the shortest distance.
    for j in range(len(distances)):
        for k in range(3):
            if minimum[j] == distances[j][k]:
                membership.append(1)
            else:
                membership.append(0)
        gamma.append(membership)
        membership = []
    return np.array(gamma)


# This function updates the clusters and returns new ones.
def Update(gamma,X):
    n = 0
    mu = []
    sums = np.array([0,0])
    cluster = []
    for i in range(3):
        for j in range(len(X)):
            if gamma[j][i] == 1:
                n = n + 1
        for k in range(len(X)):
            mu.append(gamma[k][i]*X[k])
        muf=np.array(mu)
        for l in range(len(X)):
            sums = sums + muf[l]
        average = sums/n
        cluster.append(average)
        # Re-starting values for next iteration for the following cluster i.
        n = 0
        mu = []
        muf = np.array([0,0])
        sums = np.array([0,0])
    return np.array(cluster)


# We use the k-means with 20 iterations.
for w in range(20):
    # Geting the distance from each point to each cluster.
    distances = DistanceMatrix(centers,X)
    # We get the 'gamma' matrix, which determines in which cluster each point belong.
    gamma = Membership(distances)
    # Updating the clusters centers.
    centers = Update(gamma,X)
    
#Grafica los nuevos centroides actualizados.
plt.scatter(centers[:,0], centers[:,1], s=150, c = 'yellow')
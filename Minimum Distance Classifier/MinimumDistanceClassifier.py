# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:33:27 2021

@author: Edgar Mata
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_blobs

#Generating 200 points with 3 centers and standard deviation 0.5
colour = ['green','blue','red'] #List for colouring the points.
samp,cent = 200,3
training_data,labels = make_blobs(n_samples=samp,centers=cent,cluster_std=0.5,random_state=2)
plt.scatter(training_data[:,0],training_data[:,1], c=np.take(colour,labels))
#c=np.take(,) takes the 'colour' list and associates the points and their respective label.

#Creating 10 new test points with the same parameters as the 200 points.
test_samp = 10
test_data,label_testdata=make_blobs(n_samples=test_samp,centers=cent,cluster_std=0.5,random_state=2)

#This function finds the average vectors using the labels for each of the 200 points.
def CENTERS(training_data,labels):
    group = []
    aver2 = []
    n=0
    sums = np.array([0,0])
    for i in range(cent):
        for j in range(samp):
            if labels[j] == i:
                group.append(1)
            else:
                group.append(0)
        #To this point, 'group' has identified the vectors which belong to the i-th cluster.
        groupof = np.array(group)
        #Counts the number of vectors (1's) in 'groupf'.
        for k in range(samp):
            if groupof[k]==1:
                n=n+1
        #Take the vectors average.
        for l in range(samp):
            sums = sums + training_data[l]*groupof[l] #Sums all the vectors of the group.
        aver = sums/n
        ###### This is only for one center ######
        aver2.append(aver) #Saving the average vectors in 'aver2'.
        group = [] #Emptying the variables for next iteration.
        sums = np.array([0,0])
        groupof = np.array([0,0])
        n = 0
    return np.array(aver2)
        
#Getting the centers, or average vectors, with the previously created function.
centers = CENTERS(training_data,labels)
#Plotting the average vectors of each group.
plt.scatter(centers[:,0],centers[:,1],c='black',s=150)
plt.scatter(test_data[:,0],test_data[:,1])
#Plotting the 10 new points to notice how many are in each group.
plt.scatter(test_data[:,0],test_data[:,1], c='yellow')
plt.grid()
plt.title('Minimum Distance Classifier')
        

#This functions calculates the distances from a set of points (test_data) to another one (the centers).
#Returns distances matrix.
def distance(test_data,centers):
    dist = [] #Creating a list for saving the distances.
    distM = [] #Creating a list that will be used as matrix for 'dist'.
    for i in range(test_samp):
        for j in range(cent):
            dist.append(math.dist(test_data[i],centers[j]))
        distM.append(dist)
        dist = []
    return np.array(distM)

#Creating the matrix of distances.
distance_matrix = distance(test_data,centers)


#This function assigns the labels for the test points (test_data).
def testLabels(distance_matrix):
    minim_vect = []
    label_vect = []
    for i in range(test_samp):
            minim = min(distance_matrix[i]) #Takes the minimum value for each row of the distance_matrix.
            minim_vect.append(minim) #Fills a list with the minimum distances for each point.
    for j in range(test_samp):
        for k in range(cent):
            if minim_vect[j] == distance_matrix[j][k]:
                label_vect.append(k) #Saving the group in which each point in label_vect belongs to.
    return np.array(label_vect)

predicted_labels  = testLabels(distance_matrix)

#Saveing predicted_labels as an extra column in distance_matrix.
distance_matrix = np.insert(distance_matrix,distance_matrix.shape[1],predicted_labels,1)

"""
This functions counts the number of correct and wrong predictions for the obtained labels
with the minimum distance classifier vs the obtained labels with make_blobs().
"""
def correctP(predicted_labels,label_testdata):
    correct = 0
    error = 0
    for i in range(test_samp):
        if predicted_labels[i]==label_testdata[i]:
            correct = correct+1
        else:
            error = error+1
    return correct,error

CorrectPredictions,Error=correctP(predicted_labels,label_testdata)

print('Correct Predictions: ',CorrectPredictions,'\nErrors: ',Error,'\nDistance_matrix: \n',distance_matrix)


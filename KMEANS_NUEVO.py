# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:12:49 2021

@author: Edgar Mata
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import make_blobs

#Genera 300 puntos y almacena en X; en Y almacena las etiquetas de cada punto.
X,Y = make_blobs(n_samples=300,centers=3,cluster_std=1,random_state=3)
#Se toman tres centros aleatorios.
centros = np.array([X[0],X[4],X[10]])
color = ['green','blue','red'] #Vector para dar color a los puntos.

#Se grafican los 300 puntos.
plt.scatter(X[:,0], X[:,1], s=50, c=np.take(color,Y))
#Grafica los centroides iniciales.
plt.scatter(centros[:,0], centros[:,1], s=100, c='black')


#Esta función obtiene la distancia de cada punto a cada cluster y guarda en una matriz.
def MatrizDistancia(centros,X):
    dist = []
    distM = []
    for i in range(300):
        for j in range(3):
            dist.append(math.dist(X[i],centros[j])) #Toma la distancia del i-esimo punto con el j-esimo centro.
        distM.append(dist)
        dist = []
        #distM.clear()
    binario=np.array(distM) #Convierte en arreglo matricial la lista.
    return binario


#Se obtienen las distancias de cada punto a cada cluster.
distancias = MatrizDistancia(centros,X)


#Esta función devuelve una matriz que indica a qué grupo pertenece cada punto.
def Membresia(distancias):
    membresia = []
    gamma = []
    minimo = []
    for i in range(300):
        minimo.append(min(distancias[i])) #Toma la distancia mínima para cada punto.
    for j in range(300):
        for k in range(3):
            if minimo[j] == distancias[j][k]:
                membresia.append(1)
            else:
                membresia.append(0)
        gamma.append(membresia)
        membresia = []
    return np.array(gamma)

gamma = Membresia(distancias)

#Esta funcion actualiza los centroides y devuelve los nuevos.
def Actualiza(gamma,X):
#for w in range(1):
    n = 0
    mu = []
    suma = np.array([0,0])
    centroide = []
    for i in range(3):
        for j in range(300):
            if gamma[j][i] == 1:
                n = n + 1
        for k in range(300):
            mu.append(gamma[k][i]*X[k])
        muf=np.array(mu)
        for l in range(300):
            suma = suma + muf[l]
        promedio = suma/n
        centroide.append(promedio)
        #Se reinician valores para itereación de siguiente cluster (i).
        n = 0
        mu = []
        muf = np.array([0,0])
        suma = np.array([0,0])
    return np.array(centroide)


for w in range(20):
    #Se obtienen las distancias de cada punto a cada cluster.
    distancias = MatrizDistancia(centros,X)
    #Se obtiene la matriz gamma que determina para cada vector a qué cluster pertenecen.
    gamma = Membresia(distancias)
    #Actualiza los centroides.
    centros = Actualiza(gamma,X)
    
#Grafica los nuevos centroides actualizados.
plt.scatter(centros[:,0], centros[:,1], s=150, c = 'yellow')

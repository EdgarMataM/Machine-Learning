# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:46:42 2021

@author: Edgar Mata
"""

"""
NOTE: In order to plotting the decision boundaries, the code provided in the following
link was used:
https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib

For using cross-validation the code provided in the following link was used:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    
"""
### Note for the plottings of the Feedforward Neural Network (FFNN) ###

"""
3 hidden layers were used in the neural network.
A random seed with value 11 was also used to obtain specific values in every 
iteration, when the type XOR data was obtained.
"""

### Note for the decision boundaries figure and the accuracies table. ###
"""
The decision boundaries figure can be seen as a matrix, where the columns represent
the different classifiers and the rows are each of the 4 datasets. For instance, the
element M_24 of the figure represents the decision boundary of the Perceptron neural 
network implementation (not the one provided by Sklearn) applied to the second dataset
of concenctric circles.

Analogously, for the accuracies table, the arrangement is similar, so that the
element C_24 of the table represents the average accuracy of the Perceptron
neural network (not the one provided by Sklearn) applied to the second dataset,
i.e., each element of the table corresponds to each element of the decision 
boundaries table.
"""

#Libraries to use.
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn import metrics #Method for creating the confusion matrix.
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import seaborn as sns; sns.set()

""""
Genterating the type XOR dataset. The distribution was made using the code
provided in class from the notebook:
https://colab.research.google.com/drive/1i9CEIJKX4pp3ib4kBMMROI4ugVSFPtCy#scrollTo=WpRD1T-_2JUp
"""

#Initializing the seed to set a one-time randomness.
np.random.seed(11)

#Creating the type XOR dataset.

#Mean and standard deviation for first class.
mu_x1, sigma_x1 = 0, 0.1

#Constants which will differentiate first class.
x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

#First distribution.
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'type': 0})

d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'type': 1})

d3 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'type': 0})

d4 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'type': 1})

#Converting numpy arrays into pandas dataframe.
d1 = d1.to_numpy()
d2 = d2.to_numpy()
d3 = d3.to_numpy()
d4 = d4.to_numpy()
xf4_1 = []
xf4_2 = []
x_etiq4 = []
d =[d1,d2,d3,d4]

for i in range(len(d)):
    for j in range(len(d1)):
        xf4_1.append(d[i][j,0]) #Guarda todos los puntos de primer componente del dataset en la lista.
        xf4_2.append(d[i][j,1])
        x_etiq4.append(int(d[i][j,2]))
        
#Convierte a arreglos de numpy los datos anteriores.
xf4 = np.transpose(np.array([xf4_1,xf4_2]))
x_etiq4 = np.array(x_etiq4)

xf4p = pd.DataFrame(xf4)
x_etiq4p = pd.DataFrame(x_etiq4)



#Se abre como lectura el archivo dataset_classifier1.csv y guardan los puntos.
with open('dataset_classifiers1.csv','r') as archivo:
    #Listas auxiliares para manipular los datos de los archivos.
    puntos1 = []
    aux = []
    aux2= []
    aux_etiq = []
    reader = csv.reader(archivo, delimiter = ',')
    for row in reader:
        puntos1.append(row)
    for i in range(500):
        i=i+1
        for j in range(2):
            j=j+1
            aux.append(float(puntos1[i][j]))
        aux_etiq.append(int(puntos1[i][3]))
        aux2.append(aux)
        aux = []
    x_etiq = np.array(aux_etiq) #Guarda en arreglo las etiquetas del dataset.
    xf = np.array(aux2) #Guarda en arreglo los puntos del dataset.


#Se abre como lectura el archivo dataset_classifier2.csv y guardan los puntos.
with open('dataset_classifiers2.csv','r') as archivo:
    puntos2 = []
    aux = []
    aux2= []
    aux_etiq2 = []
    reader = csv.reader(archivo, delimiter = ',')
    for row in reader:
        puntos2.append(row)
    for i in range(350):
        i=i+1
        for j in range(2):
            j=j+1
            aux.append(float(puntos2[i][j]))
        aux_etiq2.append(int(puntos2[i][3]))
        aux2.append(aux)
        aux = []
    x_etiq2 = np.array(aux_etiq2) #Guarda en arreglo las etiquetas del dataset.
    xf2 = np.array(aux2) #Guarda en arreglo los puntos del dataset.


#Se abre como lectura el archivo dataset_classifier3.csv y guardan los puntos.
with open('dataset_classifiers3.csv','r') as archivo:
    puntos3 = []
    aux = []
    aux2= []
    aux_etiq3 = []
    reader = csv.reader(archivo, delimiter = ',')
    for row in reader:
        puntos3.append(row)
    for i in range(10000):
        for j in range(2):
            aux.append(float(puntos3[i][j]))
        aux_etiq3.append(int(puntos3[i][2]))
        aux2.append(aux)
        aux = []
    x_etiq3 = np.array(aux_etiq3) #Guarda en arreglo las etiquetas del dataset.
    xf3 = np.array(aux2) #Guarda en arreglo los puntos del dataset.
    
    
    
"""
Genera dataframes de pandas para los datasets originales (Servirán para el perceptrón multicapa).
"""
xfp = pd.DataFrame(xf)
x_etiqp = pd.DataFrame(x_etiq)

xf2p = pd.DataFrame(xf2)
x_etiq2p = pd.DataFrame(x_etiq2)

xf3p = pd.DataFrame(xf3)
x_etiq3p = pd.DataFrame(x_etiq3)


"""
CLASIFICADOR DE MÍNIMA DISTANCIA DESCOMPUESTO POR FUNCIONES.
"""
#Esta función encuentra los vectores promedio con base a las etiquetas de cada uno de los 200 puntos.
def CENTROS(training_data,labels):
    grupo = []
    promedio2 = []
    n=0
    suma = np.array([0,0])
    for i in range(2):
        for j in range(len(training_data)):
            if labels[j] == i:
                grupo.append(1)
            else:
                grupo.append(0)
        #Hasta aquí, 'grupo' tiene identificado los vectores pertenecientes al i-esimo cluster.
        grupof = np.array(grupo)
        #Cuenta el número de vectores (1's) en 'grupof'.
        for k in range(len(training_data)):
            if grupof[k]==1:
                n=n+1
        #Toma promedio de vectores.
        for l in range(len(training_data)):
            suma = suma + training_data[l]*grupof[l] #Suma todos los vectores del grupo.
        promedio = suma/n
        promedio2.append(promedio) #Guarda los vectores promedio en 'promedio2'.
        grupo = [] #Vacía variables para siguiente iteración.
        suma = np.array([0,0])
        grupof = np.array([0,0])
        n = 0
    return np.array(promedio2)
        
#Función que calcula las distancias de cjto de puntos (test_data) a otro (los centros). 
#Regresa matriz de distancias.
def distancia(test_data,centros):
    dist = [] #Se crea una lista para guardar las distancias.
    distM = [] #Se crea lista que servira como matriz para 'dist'.
    for i in range(len(test_data)):
        for j in range(2):
            dist.append(math.dist(test_data[i],centros[j]))
        distM.append(dist)
        dist = []
    return np.array(distM)

#Función que asigna las etiquetas de los puntos de prueba (test_data).
def etiquetasPrueba(distance_matrix):
    minim_vect = []
    label_vect = []
    for i in range(len(distance_matrix)):
            minim = min(distance_matrix[i]) #Toma el mínimo para cada renglón de distance_matrix.
            minim_vect.append(minim) #Llena una lista de distancias mínimas para cada punto.
    for j in range(len(distance_matrix)):
        for k in range(2):
            if minim_vect[j] == distance_matrix[j][k]:
                label_vect.append(k) #Guarda el grupo al que pertenece cada punto en label_vect.
    return np.array(label_vect)

"""
Fin de funciones para clasificador de mínima distancia.
"""

"""
Red neuronal McCulloch-Pitts.
"""

#Esta función es la función de activación a usar: el escalón unitario.
def step_fun(z):
    if z>=0:
        return 1
    else:
        return 0
    
#Esta función sirve para hacer el entrenamiento de la red. Recibe las épocas, pesos iniciales
#datos y etiquetas del dataset de entrenamiento.
def entrena_percep(epochs,w0,w1,w2,x1,x2,y):
    eta = 0.1 #Factor de ...
    yn = [] #Lista para hacer la diferencia delta wi.
    for i in range(epochs):
        for j in range(len(x1)):
            z = 1*w0 + x1[j]*w1 + x2[j]*w2 
            yn.append(step_fun(z)) #Guarda la etiqueta predicha en la lista pare diferencia delta wi.
            update = eta*(y[j]-yn[j])
            #Actualiza los pesos.
            w0 = w0 + update
            w1 = w1 + update*x1[j]
            w2 = w2 + update*x2[j]    
    return w0,w1,w2 #Regresa los pesos actualizados.

#Esta función termina de entrenar al perceptrón.
def perceptron(X,y): #Recibe los datos de entrenamiento del dataset y las etiquetas.
    x1 = X[:,0] #Toma todos los elementos de la primera columna del dataset.
    x2 = X[:,1] #Toma todos los elementos de la segunda columna.

    W0,W1,W2 = entrena_percep(1,1,0.1,0.1,x1,x2,y)
    return W0,W1,W2 #Devuelve los pesos actualizados; los ya entrenados.

#Esta función usa los pesos actualizados para realizar la predicción sobre un conjunto de prueba.
def prediction_percep(W0,W1,W2,X):
    x1 = X[:,0] #Toma todos los elementos de la primera columna del dataset.
    x2 = X[:,1] #Todos los elementos de la segunda columna.
    y_pred = [] #Lista para devolver las etiquetas predichas.
    for i in range(len(x1)):
        zn = 1*W0+x1[i]*W1+x2[i]*W2 #Ya con los pesos actualizados, evalua con func. de act.
        y_pred.append(step_fun(zn))
    return np.array(y_pred)

"""
RED NEURONAL MULTICAPA TOMADA DEL SCRIPT:
https://colab.research.google.com/drive/1i9CEIJKX4pp3ib4kBMMROI4ugVSFPtCy#scrollTo=tBN3GMGZAH_r
"""
#Función de activación sigmoide.
def sigmoid(s):
    # Activation function
    return 1 / (1 + np.exp(-s))
#Derivada de func. de act.
def sigmoid_prime(s):
    # Derivative of the sigmoid
    return sigmoid(s) * (1 - sigmoid(s))

class FFNN(object):

    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        # Adding 1 as it will be our bias
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size
        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0

        # The whole weight matrix, from the inputs till the hidden layer
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        # The final set of weights from the hidden layer till the output layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        # Forward propagation through our network
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.z1 = np.dot(X, self.w1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = sigmoid(self.z1)  # activation function
        self.z3 = np.dot(self.z2, self.w2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = sigmoid(self.z3)  # final activation function
        return o

    def backward(self, X, y, output, step):
        # Backward propagation of the errors
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.o_error = y - output  # error in output
        self.o_delta = self.o_error * sigmoid_prime(output) * step  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(
            self.w2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2) * step  # applying derivative of sigmoid to z2 error

        self.w1 += X.T.dot(self.z2_delta)  # adjusting first of weights
        self.w2 += self.z2.T.dot(self.o_delta)  # adjusting second set of weights

    def predict(self, X):
        return forward(self, X)

    def fit(self, X, y, epochs=10, step=0.05):
        for epoch in range(epochs):
            X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
            output = self.forward(X)
            self.backward(X, y, output, step)
            
#Esta función sirve para definir la clase binaria tras obtener el valor de la salida en el perceptrón multicapa.
def binarioFFNN(y):
   prediccion = []
   for i in range(len(y)):
       threshold = 0.5
       if y[i] < threshold:
           prediccion.append(0)
       else:
           prediccion.append(1)
   return np.array(prediccion)

def binarioFFNN3(y):
    prediccion = []
    for i in range(len(y)):
        threshold = 0.5
        if y[i] <= threshold:
           prediccion.append(0)
        else:
           prediccion.append(1)
    return np.array(prediccion)
  

"""
EN ESTE PUNTO DEL CÓDIGO SE ENTRENAN LOS CLASIFICADORES Y SE GRAFICAN PARA CADA DATASET.
"""


"""
Estos valores serán usados para graficar las regiones de decisión.
"""

color = ['g','b'] #Este vector servirá para dar color a los puntos.
h = 0.09
x_min, x_max = xf[:, 0].min() - 1, xf[:, 0].max() + 1
y_min, y_max = xf[:, 1].min() - 1, xf[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
"""
"""


"""
*** PRIMER DATASET ***
"""
### Se entrena el conjunto de datos para el KNN usando la biblioteca de sklearn.
knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(xf,x_etiq) #xf corresponde a los puntos del dataset 1; x_etiq son sus etiquetas.

#Se hace el gráfico de fronteras con el código extraído del enlace mostrado al inicio.
A1 = knn1.predict(np.c_[xx.ravel(), yy.ravel()])
Z1 = A1.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,1).title.set_text('KNN.') #Se usa subplot para colocar varios gráficos en una ventana.
plt.pcolormesh(xx, yy, Z1, cmap=cmap_light)
plt.scatter(xf[:, 0], xf[:, 1], c=np.take(color,x_etiq), cmap=cmap_bold,edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el conjunto de datos para la SVM usando la biblioteca de sklearn.
maquina = SVC(kernel = 'rbf', C = 10, gamma=0.1)
maquina.fit(xf,x_etiq)

#Se hace el gráfico de fronteras.
A2 = maquina.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = A2.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,2).title.set_text('SVM.')
plt.pcolormesh(xx, yy, Z2, cmap=cmap_light)
plt.scatter(xf[:, 0], xf[:, 1], c=np.take(color,x_etiq), cmap=cmap_bold,edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el clasificador de mínima distancia tomando como puntos de prueba los obtenidos con np.meshgrid.
#Calcula los centros para el primer dataset.
centros = CENTROS(xf,x_etiq)
#Genera la matriz de distancias.
distance_matrix = distancia(np.c_[xx.ravel(),yy.ravel()],centros)
#Regresa etiquetas que serán usadas para graficar las fronteras de decisión.
predicted_labels  = etiquetasPrueba(distance_matrix)

#Se hace el gráfico de fronteras.
A3 = predicted_labels
Z3 = A3.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,3).title.set_text('CMD.')
plt.pcolormesh(xx, yy, Z3, cmap=cmap_light)
plt.scatter(xf[:, 0], xf[:, 1], c=np.take(color,x_etiq), cmap=cmap_bold,edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el la red neuronal tomando como puntos de prueba los obtenidos con np.meshgrid.
W0,W1,W2 = perceptron(xf, x_etiq) #Obtiene los pesos actualizados tras entrenar perceptrón.
A10 = prediction_percep(W0,W1,W2,np.c_[xx.ravel(), yy.ravel()]) #Obtiene etiquetas de datos de prueba con np.meshgrid.
#Se hace el gráfico de fronteras.
Z10 = A10.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,4).title.set_text('Perceptrón.')
plt.pcolormesh(xx, yy, Z10, cmap=cmap_light)
plt.scatter(xf[:, 0], xf[:, 1], c=np.take(color,x_etiq), cmap=cmap_bold,edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el conjunto de datos para el perceptron usando la biblioteca de sklearn.
percep = Perceptron(tol=1e-3, random_state=0)
percep.fit(xf,x_etiq)

#Se hace el gráfico de fronteras.
A13 = percep.predict(np.c_[xx.ravel(), yy.ravel()])
Z13 = A13.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,5).title.set_text('Perc. (sk)')
plt.pcolormesh(xx, yy, Z13, cmap=cmap_light)
plt.scatter(xf[:, 0], xf[:, 1], c=np.take(color,x_etiq), cmap=cmap_bold,edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el perceptrón de multicapas tomando como puntos de prueba los obtenidos con np.meshgrid.

pmc = FFNN()
pmc.fit(xfp,x_etiqp,500,0.05)
test_ffnn = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
pred_y = test_ffnn.apply(pmc.forward,axis=1)
A16 = binarioFFNN(pred_y)
#Se hace el gráfico de fronteras.
Z16 = A16.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,6).title.set_text('FFNN.')
plt.pcolormesh(xx, yy, Z16, cmap=cmap_light)
plt.scatter(xf[:, 0], xf[:, 1], c=np.take(color,x_etiq), cmap=cmap_bold,edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')



"""
*** SEGUNDO DATASET ***
"""

### Se entrena el conjunto de datos para el KNN usando la biblioteca de sklearn.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xf2,x_etiq2) #xf2 corresponde a los puntos del dataset 2; x_etiq2 son sus etiquetas.

#Se hace el gráfico de fronteras.
A4 = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z4 = A4.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,7)
plt.pcolormesh(xx, yy, Z4, cmap=cmap_light)
plt.scatter(xf2[:, 0], xf2[:, 1], c=np.take(color,x_etiq2), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el segundo conjunto de datos para la SVM usando la biblioteca de sklearn.
maquina = SVC(kernel = 'rbf', C = 10, gamma=0.1)
maquina.fit(xf2,x_etiq2)

#Se hace el gráfico de fronteras.
A5 = maquina.predict(np.c_[xx.ravel(), yy.ravel()])
Z5 = A5.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,8)
plt.pcolormesh(xx, yy, Z5, cmap=cmap_light)
plt.scatter(xf2[:, 0], xf2[:, 1], c=np.take(color,x_etiq2), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el clasificador de mínima distancia con los puntos de prueba obtenidos con np.meshgrid.
#Calcula los centros para el segundo dataset.
centros2 = CENTROS(xf2,x_etiq2)
#Genera la matriz de distancias.
distance_matrix2 = distancia(np.c_[xx.ravel(),yy.ravel()],centros2)
#Regresa etiquetas que serán usadas para graficar las fronteras de decisión.
predicted_labels2  = etiquetasPrueba(distance_matrix2)

#Se hace el gráfico de fronteras.
A6 = predicted_labels2
Z6 = A6.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,9)
plt.pcolormesh(xx, yy, Z6, cmap=cmap_light)
plt.scatter(xf2[:, 0], xf2[:, 1], c=np.take(color,x_etiq2), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2,2)
plt.ylim(-2,2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el la red neuronal tomando como puntos de prueba los obtenidos con np.meshgrid.
W0,W1,W2 = perceptron(xf2, x_etiq2) #Obtiene los pesos actualizados tras entrenar perceptrón.
A11 = prediction_percep(W0,W1,W2,np.c_[xx.ravel(), yy.ravel()]) #Obtiene etiquetas de datos de prueba con np.meshgrid.
#Se hace el gráfico de fronteras.
Z11 = A11.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,10)
plt.pcolormesh(xx, yy, Z11, cmap=cmap_light)
plt.scatter(xf2[:, 0], xf2[:, 1], c=np.take(color,x_etiq2), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el conjunto de datos para el perceptron usando la biblioteca de sklearn.
percep = Perceptron(tol=1e-3, random_state=0)
percep.fit(xf2,x_etiq2)

#Se hace el gráfico de fronteras.
A14 = percep.predict(np.c_[xx.ravel(), yy.ravel()])
Z14 = A14.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,11)
plt.pcolormesh(xx, yy, Z14, cmap=cmap_light)
plt.scatter(xf2[:, 0], xf2[:, 1], c=np.take(color,x_etiq2), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')



pmc = FFNN()
pmc.fit(xf2p,x_etiq2p,500,0.05)
test_ffnn = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
pred_y2 = test_ffnn.apply(pmc.forward,axis=1)
A17 = binarioFFNN(pred_y2) 
#Se hace el gráfico de fronteras.
Z17 = A17.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,12)
plt.pcolormesh(xx, yy, Z17, cmap=cmap_light)
plt.scatter(xf2[:, 0], xf2[:, 1], c=np.take(color,x_etiq2), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')



"""
*** TERCER DATASET ***
"""

### Se entrena el conjunto de datos para el KNN usando la biblioteca de sklearn.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xf3,x_etiq3) #xf3 corresponde a los puntos del dataset 3; x_etiq3 son sus etiquetas.

#Se hace el gráfico de fronteras.
A7 = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z7 = A7.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,13)
plt.pcolormesh(xx, yy, Z7, cmap=cmap_light)
plt.scatter(xf3[:, 0], xf3[:, 1], c=np.take(color,x_etiq3), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el tercer conjunto de datos para la SVM usando la biblioteca de sklearn.
maquina = SVC(kernel = 'rbf', C = 10, gamma=0.1)
maquina.fit(xf3,x_etiq3)

#Se hace el gráfico de fronteras.
A8 = maquina.predict(np.c_[xx.ravel(), yy.ravel()])
Z8 = A8.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,14)
plt.pcolormesh(xx, yy, Z8, cmap=cmap_light)
plt.scatter(xf3[:, 0], xf3[:, 1], c=np.take(color,x_etiq3), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')

### Se entrena el clasificador de mínima distancia con los puntos de prueba obtenidos con np.meshgrid.
#Calcula los centros para el tercer dataset.
centros3 = CENTROS(xf3,x_etiq3)
#Genera la matriz de distancias.
distance_matrix3 = distancia(np.c_[xx.ravel(),yy.ravel()],centros3)
#Regresa etiquetas que serán usadas para graficar las fronteras de decisión.
predicted_labels3  = etiquetasPrueba(distance_matrix3)

#Se hace el gráfico de fronteras.
A9 = predicted_labels3
Z9 = A9.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,15)
plt.pcolormesh(xx, yy, Z9, cmap=cmap_light)
plt.scatter(xf3[:, 0], xf3[:, 1], c=np.take(color,x_etiq3), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2,3)
plt.ylim(-2,3)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el la red neuronal tomando como puntos de prueba los obtenidos con np.meshgrid.
W0,W1,W2 = perceptron(xf3, x_etiq3) #Obtiene los pesos actualizados tras entrenar perceptrón.
A12 = prediction_percep(W0,W1,W2,np.c_[xx.ravel(), yy.ravel()]) #Obtiene etiquetas de datos de prueba con np.meshgrid.
#Se hace el gráfico de fronteras.
Z12 = A12.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,16)
plt.pcolormesh(xx, yy, Z12, cmap=cmap_light)
plt.scatter(xf3[:, 0], xf3[:, 1], c=np.take(color,x_etiq3), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el conjunto de datos para el perceptron usando la biblioteca de sklearn.
percep = Perceptron(tol=1e-3, random_state=0)
percep.fit(xf3,x_etiq3)

#Se hace el gráfico de fronteras.
A15 = percep.predict(np.c_[xx.ravel(), yy.ravel()])
Z15 = A15.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,17)
plt.pcolormesh(xx, yy, Z15, cmap=cmap_light)
plt.scatter(xf3[:, 0], xf3[:, 1], c=np.take(color,x_etiq3), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower left')


pmc = FFNN()
pmc.fit(xf3p,x_etiq3p,1000,0.05)
test_ffnn = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
pred_y3 = test_ffnn.apply(pmc.forward,axis=1)
A18 = binarioFFNN3(pred_y3)
#Se hace el gráfico de fronteras.
Z18 = A18.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,18)
plt.pcolormesh(xx, yy, Z18, cmap=cmap_light)
plt.scatter(xf3[:, 0], xf3[:, 1], c=np.take(color,x_etiq3), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


"""
*** CUARTO DATASET ***
"""

### Se entrena el conjunto de datos para el KNN usando la biblioteca de sklearn.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xf4,x_etiq4) #xf4 corresponde a los puntos del dataset 4; x_etiq4 son sus etiquetas.

#Se hace el gráfico de fronteras.
A19 = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z19 = A19.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,19)
plt.pcolormesh(xx, yy, Z19, cmap=cmap_light)
plt.scatter(xf4[:, 0], xf4[:, 1], c=np.take(color,x_etiq4), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el cuarto conjunto de datos para la SVM usando la biblioteca de sklearn.
maquina = SVC(kernel = 'rbf', C = 10, gamma=0.1)
maquina.fit(xf4,x_etiq4)

#Se hace el gráfico de fronteras.
A20 = maquina.predict(np.c_[xx.ravel(), yy.ravel()])
Z20 = A20.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,20)
plt.pcolormesh(xx, yy, Z20, cmap=cmap_light)
plt.scatter(xf4[:, 0], xf4[:, 1], c=np.take(color,x_etiq4), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el clasificador de mínima distancia con los puntos de prueba obtenidos con np.meshgrid.
#Calcula los centros para el cuarto dataset.
centros4 = CENTROS(xf4,x_etiq4)
#Genera la matriz de distancias.
distance_matrix4 = distancia(np.c_[xx.ravel(),yy.ravel()],centros4)
#Regresa etiquetas que serán usadas para graficar las fronteras de decisión.
predicted_labels4  = etiquetasPrueba(distance_matrix4)

#Se hace el gráfico de fronteras.
A21 = predicted_labels4
Z21 = A21.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,21)
plt.pcolormesh(xx, yy, Z21, cmap=cmap_light)
plt.scatter(xf4[:, 0], xf4[:, 1], c=np.take(color,x_etiq4), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2,2)
plt.ylim(-2,2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el la red neuronal tomando como puntos de prueba los obtenidos con np.meshgrid.
W0,W1,W2 = perceptron(xf4, x_etiq4) #Obtiene los pesos actualizados tras entrenar perceptrón.
A22 = prediction_percep(W0,W1,W2,np.c_[xx.ravel(), yy.ravel()]) #Obtiene etiquetas de datos de prueba con np.meshgrid.
#Se hace el gráfico de fronteras.
Z22 = A22.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,22)
plt.pcolormesh(xx, yy, Z22, cmap=cmap_light)
plt.scatter(xf4[:, 0], xf4[:, 1], c=np.take(color,x_etiq4), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')


### Se entrena el conjunto de datos para el perceptron usando la biblioteca de sklearn.
percep = Perceptron(tol=1e-3, random_state=0)
percep.fit(xf4,x_etiq4)

#Se hace el gráfico de fronteras.
A23 = percep.predict(np.c_[xx.ravel(), yy.ravel()])
Z23 = A23.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,23)
plt.pcolormesh(xx, yy, Z23, cmap=cmap_light)
plt.scatter(xf4[:, 0], xf4[:, 1], c=np.take(color,x_etiq4), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower left')


pmc = FFNN()
pmc.fit(xf4p,x_etiq4p,1000,0.7)
test_ffnn = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
pred_y4 = test_ffnn.apply(pmc.forward,axis=1)
A24 = binarioFFNN(pred_y4) #Obtiene las etiquetas de clases binarias.
#Se hace el gráfico de fronteras.
Z24 = A24.reshape(xx.shape)
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933'])    
plt.subplot(4,6,24)
plt.pcolormesh(xx, yy, Z24, cmap=cmap_light)
plt.scatter(xf4[:, 0], xf4[:, 1], c=np.take(color,x_etiq4), cmap=cmap_bold,edgecolor='k')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
patch0 = mpatches.Patch(color='#FF0000', label='0')
patch1 = mpatches.Patch(color='#ff9933', label='1')
plt.legend(handles=[patch0, patch1], prop = {'size': 4}, loc = 'lower right')




"""
SE REALIZA LA VALIDACIÓN CRUZADA CON AYUDA DEL CÓDIGO OBTENIDO EN EL ENLACE ESPECIFICADO 
EN EL INICIO.
"""
kf = KFold(n_splits = 10)
kf.get_n_splits(xf)


"""
Dataset 1.
"""
#Listas para guardar accuracys por cada pliege y obtener un promedio.
kn1_prom = [] 
svm1_prom = [] 
cmd1_prom = []
pct1_prom = []
pcsk1_prom = []
ffnn1_prom = []
for train_index, test_index in kf.split(xf):
    x_train, x_test = xf[train_index], xf[test_index]
    y_train, y_test = x_etiq[train_index], x_etiq[test_index]
    
    """
    Validación cruzada para KNN.
    """
    knn1 = KNeighborsClassifier(n_neighbors=5)
    knn1.fit(x_train,y_train)  #Se entrena el knn con el dataset del 
    k1_labels = knn1.predict(x_test)
    k1_matrix = metrics.confusion_matrix(y_test,k1_labels) #Genera matriz de confusión.
    traza_k1 = np.trace(k1_matrix)
    s_k1 = np.sum(k1_matrix) #Suma todos los elementos en la matriz.
    acc_k1 = traza_k1/s_k1
    kn1_prom.append(acc_k1) #Guarda el accuracy por pliege en la lista.
    
    """
    Validación para SVM.
    """
    svm1 = SVC(kernel = 'rbf', C = 10, gamma=0.1)
    svm1.fit(x_train,y_train)
    svm1_labels = svm1.predict(x_test)
    svm1_matrix =  metrics.confusion_matrix(y_test,svm1_labels)
    traza_svm1 = np.trace(svm1_matrix)
    s_svm1 = np.sum(svm1_matrix) #Suma todos los elementos en la matriz.
    acc_svm1 = traza_svm1/s_svm1   
    svm1_prom.append(acc_svm1)  #Guarda el accuracy por pliege en la lista.
    
    """
    Validación para Clasificador de mínima distancia.
    """
    centros_cmd1 = CENTROS(x_train,y_train)
    distance_matrix_cmd1 = distancia(x_test,centros_cmd1)
    cmd1_labels  = etiquetasPrueba(distance_matrix_cmd1)
    cmd1_matrix = metrics.confusion_matrix(y_test,cmd1_labels)
    traza_cmd1 = np.trace(cmd1_matrix)
    s_cmd1 = np.sum(cmd1_matrix) #Suma todos los elementos en la matriz.
    acc_cmd1 = traza_cmd1/s_cmd1
    cmd1_prom.append(acc_cmd1)  #Guarda el accuracy por pliege en la lista.
    
    """
    Validación para red neuronal.
    """
    w0_1,w1_1,w2_1 = perceptron(x_train,y_train) #Se entrena perceptron.
    pct1_labels = prediction_percep(w0_1,w1_1,w2_1,x_test) #Predice eitquetas.
    pct1_matrix = metrics.confusion_matrix(y_test,pct1_labels)
    traza_pct1 = np.trace(pct1_matrix)
    s_pct1 = np.sum(pct1_matrix)
    acc_pct1 = traza_pct1/s_pct1
    pct1_prom.append(acc_pct1)
    
    """
    Validación para red neuronal de sklearn.
    """
    pcsk1 = Perceptron(tol=1e-3, random_state=0)
    pcsk1.fit(x_train,y_train)
    pcsk1_labels = pcsk1.predict(x_test)
    pcsk1_matrix = metrics.confusion_matrix(y_test,pcsk1_labels)
    traza_pcsk1 = np.trace(pcsk1_matrix)
    s_pcsk1 = np.sum(pcsk1_matrix)
    acc_pcsk1 = traza_pcsk1/s_pcsk1
    pcsk1_prom.append(acc_pcsk1)    
    
    """
    Validación para red neuronal de capas múltiples.
    """
    ffnn1 = FFNN()
    ffnn1.fit(pd.DataFrame(x_train),pd.DataFrame(y_train),1000,0.7)
    predicciones1 = (pd.DataFrame(x_test)).apply(ffnn1.forward,axis=1)
    ffnn1_labels = binarioFFNN(predicciones1)
    ffnn1_matrix = metrics.confusion_matrix(y_test,ffnn1_labels)
    traza_ffnn1 = np.trace(ffnn1_matrix)
    s_ffnn1 = np.sum(ffnn1_matrix)
    acc_ffnn1 = traza_ffnn1/s_ffnn1
    ffnn1_prom.append(acc_ffnn1)    
    
"""
Dataset 2.
"""
kn2_prom = [] 
svm2_prom = [] 
cmd2_prom = []
pct2_prom = []
pcsk2_prom = []
ffnn2_prom = []

for train_index, test_index in kf.split(xf2):
    x_train, x_test = xf2[train_index], xf2[test_index]
    y_train, y_test = x_etiq2[train_index], x_etiq2[test_index]
    
    """
    Validación cruzada para KNN.
    """
    knn2 = KNeighborsClassifier(n_neighbors=5)
    knn2.fit(x_train,y_train) 
    k2_labels = knn2.predict(x_test)
    k2_matrix = metrics.confusion_matrix(y_test,k2_labels) #Genera matriz de confusión.
    traza_k2 = np.trace(k2_matrix)
    s_k2 = np.sum(k2_matrix) #Suma todos los elementos en la matriz.
    acc_k2 = traza_k2/s_k2
    kn2_prom.append(acc_k2)
    
    """
    Validación para SVM.
    """
    svm2 = SVC(kernel = 'rbf', C = 10, gamma=0.1)
    svm2.fit(x_train,y_train)
    svm2_labels = svm2.predict(x_test)
    svm2_matrix =  metrics.confusion_matrix(y_test,svm2_labels)
    traza_svm2 = np.trace(svm2_matrix)
    s_svm2 = np.sum(svm2_matrix) #Suma todos los elementos en la matriz.
    acc_svm2 = traza_svm2/s_svm2    
    svm2_prom.append(acc_svm2)
    
    """
    Validación para Clasificador de mínima distancia.
    """
    centros_cmd2 = CENTROS(x_train,y_train)
    distance_matrix_cmd2 = distancia(x_test,centros_cmd2)
    cmd2_labels  = etiquetasPrueba(distance_matrix_cmd2)
    cmd2_matrix = metrics.confusion_matrix(y_test,cmd2_labels)
    traza_cmd2 = np.trace(cmd2_matrix)
    s_cmd2 = np.sum(cmd2_matrix) #Suma todos los elementos en la matriz.
    acc_cmd2 = traza_cmd2/s_cmd2     
    cmd2_prom.append(acc_cmd2)
    
    """
    Validación para red neuronal.
    """
    w0_2,w1_2,w2_2 = perceptron(x_train,y_train) #Se entrena perceptron.
    pct2_labels = prediction_percep(w0_2,w1_2,w2_2,x_test) #Predice eitquetas.
    pct2_matrix = metrics.confusion_matrix(y_test,pct2_labels)
    traza_pct2 = np.trace(pct2_matrix)
    s_pct2 = np.sum(pct2_matrix)
    acc_pct2 = traza_pct2/s_pct2
    pct2_prom.append(acc_pct2)
    
    """
    Validación para red neuronal de sklearn.
    """
    pcsk2 = Perceptron(tol=1e-3, random_state=0)
    pcsk2.fit(x_train,y_train)
    pcsk2_labels = pcsk2.predict(x_test)
    pcsk2_matrix = metrics.confusion_matrix(y_test,pcsk2_labels)
    traza_pcsk2 = np.trace(pcsk2_matrix)
    s_pcsk2 = np.sum(pcsk2_matrix)
    acc_pcsk2 = traza_pcsk2/s_pcsk2
    pcsk2_prom.append(acc_pcsk2)    

    """
    Validación para red neuronal de capas múltiples.
    """
    ffnn2 = FFNN()
    ffnn2.fit(pd.DataFrame(x_train),pd.DataFrame(y_train),1000,0.7)
    predicciones2 = (pd.DataFrame(x_test)).apply(ffnn2.forward,axis=1)
    ffnn2_labels = binarioFFNN(predicciones2)
    ffnn2_matrix = metrics.confusion_matrix(y_test,ffnn2_labels)
    traza_ffnn2 = np.trace(ffnn2_matrix)
    s_ffnn2 = np.sum(ffnn2_matrix)
    acc_ffnn2 = traza_ffnn2/s_ffnn2
    ffnn2_prom.append(acc_ffnn2)    
    
"""
Dataset 3.
"""
kn3_prom = [] 
svm3_prom = [] 
cmd3_prom = []
pct3_prom = []
pcsk3_prom = []
ffnn3_prom = []

for train_index, test_index in kf.split(xf3):
    x_train, x_test = xf3[train_index], xf3[test_index]
    y_train, y_test = x_etiq3[train_index], x_etiq3[test_index]
    
    """
    Validación cruzada para KNN.
    """
    knn3 = KNeighborsClassifier(n_neighbors=5)
    knn3.fit(x_train,y_train) 
    k3_labels = knn3.predict(x_test)
    k3_matrix = metrics.confusion_matrix(y_test,k3_labels) #Genera matriz de confusión.
    traza_k3 = np.trace(k3_matrix)
    s_k3 = np.sum(k3_matrix) #Suma todos los elementos en la matriz.
    acc_k3 = traza_k3/s_k3
    kn3_prom.append(acc_k3)
    
    """
    Validación para SVM.
    """
    svm3 = SVC(kernel = 'rbf', C = 10, gamma=0.1)
    svm3.fit(x_train,y_train)
    svm3_labels = svm3.predict(x_test)
    svm3_matrix =  metrics.confusion_matrix(y_test,svm3_labels)
    traza_svm3 = np.trace(svm3_matrix)
    s_svm3 = np.sum(svm3_matrix) #Suma todos los elementos en la matriz.
    acc_svm3 = traza_svm3/s_svm3    
    svm3_prom.append(acc_svm3)
    
    """
    Validación para Clasificador de mínima distancia.
    """
    centros_cmd3 = CENTROS(x_train,y_train)
    distance_matrix_cmd3 = distancia(x_test,centros_cmd3)
    cmd3_labels  = etiquetasPrueba(distance_matrix_cmd3)
    cmd3_matrix = metrics.confusion_matrix(y_test,cmd3_labels)
    traza_cmd3 = np.trace(cmd3_matrix)
    s_cmd3 = np.sum(cmd3_matrix) #Suma todos los elementos en la matriz.
    acc_cmd3 = traza_cmd3/s_cmd3
    cmd3_prom.append(acc_cmd3)
    
    """
    Validación para red neuronal.
    """
    w0_3,w1_3,w2_3 = perceptron(x_train,y_train) #Se entrena perceptron.
    pct3_labels = prediction_percep(w0_3,w1_3,w2_3,x_test) #Predice eitquetas.
    pct3_matrix = metrics.confusion_matrix(y_test,pct3_labels)
    traza_pct3 = np.trace(pct3_matrix)
    s_pct3 = np.sum(pct3_matrix)
    acc_pct3 = traza_pct3/s_pct3
    pct3_prom.append(acc_pct3)
    
    """
    Validación para red neuronal de sklearn.
    """
    pcsk3 = Perceptron(tol=1e-3, random_state=0)
    pcsk3.fit(x_train,y_train)
    pcsk3_labels = pcsk3.predict(x_test)
    pcsk3_matrix = metrics.confusion_matrix(y_test,pcsk3_labels)
    traza_pcsk3 = np.trace(pcsk3_matrix)
    s_pcsk3 = np.sum(pcsk3_matrix)
    acc_pcsk3 = traza_pcsk3/s_pcsk3
    pcsk3_prom.append(acc_pcsk3)    
    
    """
    Validación para red neuronal de capas múltiples.
    """
    ffnn3 = FFNN()
    ffnn3.fit(pd.DataFrame(x_train),pd.DataFrame(y_train),1000,0.7)
    predicciones3 = (pd.DataFrame(x_test)).apply(ffnn3.forward,axis=1)
    ffnn3_labels = binarioFFNN3(predicciones3)
    ffnn3_matrix = metrics.confusion_matrix(y_test,ffnn3_labels)
    traza_ffnn3 = np.trace(ffnn3_matrix)
    s_ffnn3 = np.sum(ffnn3_matrix)
    acc_ffnn3 = traza_ffnn3/s_ffnn3
    ffnn3_prom.append(acc_ffnn3)    
    
"""
Dataset 4.
"""
kn4_prom = [] 
svm4_prom = [] 
cmd4_prom = []
pct4_prom = []
pcsk4_prom = []
ffnn4_prom = []

for train_index, test_index in kf.split(xf4):
    x_train, x_test = xf4[train_index], xf4[test_index]
    y_train, y_test = x_etiq4[train_index], x_etiq4[test_index]
    
    """
    Validación cruzada para KNN.
    """
    knn4 = KNeighborsClassifier(n_neighbors=5)
    knn4.fit(x_train,y_train) 
    k4_labels = knn4.predict(x_test)
    k4_matrix = metrics.confusion_matrix(y_test,k4_labels) #Genera matriz de confusión.
    traza_k4 = np.trace(k4_matrix)
    s_k4 = np.sum(k4_matrix) #Suma todos los elementos en la matriz.
    acc_k4 = traza_k4/s_k4
    kn4_prom.append(acc_k4)
    
    """
    Validación para SVM.
    """
    svm4 = SVC(kernel = 'rbf', C = 10, gamma=0.1)
    svm4.fit(x_train,y_train)
    svm4_labels = svm4.predict(x_test)
    svm4_matrix =  metrics.confusion_matrix(y_test,svm4_labels)
    traza_svm4 = np.trace(svm4_matrix)
    s_svm4 = np.sum(svm4_matrix) #Suma todos los elementos en la matriz.
    acc_svm4 = traza_svm4/s_svm4    
    svm4_prom.append(acc_svm4)
    
    """
    Validación para Clasificador de mínima distancia.
    """
    centros_cmd4 = CENTROS(x_train,y_train)
    distance_matrix_cmd4 = distancia(x_test,centros_cmd4)
    cmd4_labels  = etiquetasPrueba(distance_matrix_cmd4)
    cmd4_matrix = metrics.confusion_matrix(y_test,cmd4_labels)
    traza_cmd4 = np.trace(cmd4_matrix)
    s_cmd4 = np.sum(cmd4_matrix) #Suma todos los elementos en la matriz.
    acc_cmd4 = traza_cmd4/s_cmd4
    cmd4_prom.append(acc_cmd4)
    
    """
    Validación para red neuronal.
    """
    w0_4,w1_4,w2_4 = perceptron(x_train,y_train) #Se entrena perceptron.
    pct4_labels = prediction_percep(w0_4,w1_4,w2_4,x_test) #Predice eitquetas.
    pct4_matrix = metrics.confusion_matrix(y_test,pct4_labels)
    traza_pct4 = np.trace(pct4_matrix)
    s_pct4 = np.sum(pct4_matrix)
    acc_pct4 = traza_pct4/s_pct4
    pct4_prom.append(acc_pct4)
    
    """
    Validación para red neuronal de sklearn.
    """
    pcsk4 = Perceptron(tol=1e-3, random_state=0)
    pcsk4.fit(x_train,y_train)
    pcsk4_labels = pcsk4.predict(x_test)
    pcsk4_matrix = metrics.confusion_matrix(y_test,pcsk4_labels)
    traza_pcsk4 = np.trace(pcsk4_matrix)
    s_pcsk4 = np.sum(pcsk4_matrix)
    acc_pcsk4 = traza_pcsk4/s_pcsk4
    pcsk4_prom.append(acc_pcsk4)    
    
    """
    Validación para red neuronal de capas múltiples.
    """
    ffnn4 = FFNN()
    ffnn4.fit(pd.DataFrame(x_train),pd.DataFrame(y_train),1000,0.7)
    predicciones4 = (pd.DataFrame(x_test)).apply(ffnn4.forward,axis=1)
    ffnn4_labels = binarioFFNN(predicciones4)
    ffnn4_matrix = metrics.confusion_matrix(y_test,ffnn4_labels)
    traza_ffnn4 = np.trace(ffnn4_matrix)
    s_ffnn4 = np.sum(ffnn4_matrix)
    acc_ffnn4 = traza_ffnn4/s_ffnn4
    ffnn4_prom.append(acc_ffnn4)    
    
# Aquí se obtiene el promedio de accuracy para cada clasificador y dataset.
d1 = [sum(kn1_prom)/len(kn1_prom),sum(kn2_prom)/len(kn2_prom),sum(kn3_prom)/len(kn3_prom),sum(kn4_prom)/len(kn4_prom)]
d2 = [sum(svm1_prom)/len(svm1_prom),sum(svm2_prom)/len(svm2_prom),sum(svm3_prom)/len(svm3_prom),sum(svm4_prom)/len(svm4_prom)]
d3 = [sum(cmd1_prom)/len(cmd1_prom),sum(cmd2_prom)/len(cmd2_prom),sum(cmd3_prom)/len(cmd3_prom),sum(cmd4_prom)/len(cmd4_prom)]
d4 = [sum(pct1_prom)/len(pct1_prom),sum(pct2_prom)/len(pct2_prom),sum(pct3_prom)/len(pct3_prom),sum(pct4_prom)/len(pct4_prom)]
d5 = [sum(pcsk1_prom)/len(pcsk1_prom),sum(pcsk2_prom)/len(pcsk2_prom),sum(pcsk3_prom)/len(pcsk3_prom),sum(pcsk4_prom)/len(pcsk4_prom)]
d6 = [sum(ffnn1_prom)/len(ffnn1_prom),sum(ffnn2_prom)/len(ffnn2_prom),sum(ffnn3_prom)/len(ffnn3_prom),sum(ffnn4_prom)/len(ffnn4_prom)]
accuracies = np.array([d1,d2,d3,d4,d5,d6])
accuracies_f = np.transpose(accuracies) #Tabla final de accuracies.
print(accuracies_f)
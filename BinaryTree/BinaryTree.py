# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:48:50 2021

@author: Edgar Mata
"""

import csv
import numpy as np

# Global lists to save values after traversing the tree.
InOr = ['In order: ']
PreOr = ['Pre order: ']
PosOr = ['Post order: ']

"""
Open the .csv file as reading.
Here, the file numbers are given to a list, then we convert the
numbers from the list to floats to place them in the 'random' array.
"""
with open('random.csv','r') as file:
    data = []
    dataf = []
    aux = []
    reader = csv.reader(file, delimiter = ',')
    for row in reader:
        data.append(row)
    for k in range(8): 
        for i in range(4):
            aux.append(float(data[k][i]))
        dataf.append(aux)
        aux = []
    
random = np.array(dataf) #The list becomes a numpy array.

#Creating the class Node.
class Node:
    def __init__(self, val=0, left=None, right=None): #Nodes constructor.
        self.val = val
        self.left = left
        self.right = right
        
    def __str__(self): #This method allows to show the node value when printing.
        return " %s " %(self.val)
        
#Creating aBinaries class.
class aBinaries:
    #Class constructor.
    def __init__(self):
        self.root = None
    
    #This method is responsible for arranging the numbers in the nodes of the tree.
    def add(self, element):
        if self.root == None: 
            self.root = element #If root is null, then root node is defined.
        else:
            parent = None
            aux2 = self.root
            while aux2 != None:
                parent = aux2
                if element.val >= aux2.val:
                    aux2 = aux2.right
                else:
                    aux2 = aux2.left
            if element.val >= parent.val:
                parent.right = element
            else:
                parent.left = element
                
    #Method for in order path.
    def inorder(self,element):
        global InOr
        if element != None:
            self.inorder(element.left)
            print(element)
            InOr.append(element.val) #Saves the value obtained in InOr.
            self.inorder(element.right)
        return np.array(InOr)    
            
    #Method for pre order path.
    def preorder(self,element):
        global PreOr
        if element != None:
            print(element)
            PreOr.append(element.val)
            self.preorder(element.left)
            self.preorder(element.right)
        return np.array(PreOr)
    
    #Method for post order path.
    def postorder(self,element):
        global PosOr
        if element != None:
            self.postorder(element.left)
            self.postorder(element.right)
            print(element)
            PosOr.append(element.val)
        return np.array(PosOr)
    
    #This method returns the tree roots to execute the through.
    def getRaiz(self):
        return self.root
    
#Creating an object from the aBinaries class; the tree.
tree = aBinaries()

#Creating nodes and adding them to the tree.
for i in range(8):
    for j in range(4):        
        node = Node(random[i][j])
        tree.add(node)
        
#Writting the .csv file using methods to traverse the tree.
with open('paths.csv', 'w', newline = '') as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(tree.inorder(tree.getRaiz()))
    writer.writerow(tree.preorder(tree.getRaiz()))
    writer.writerow(tree.postorder(tree.getRaiz()))

"""
References:
For the tree implementation (specifically for Node class), the following
website was used:
https://dev.to/abdisalan_js/4-ways-to-traverse-binary-trees-with-animations-5bi5

and the video:
https://www.youtube.com/watch?v=ph5HPNvWaCk&list=WL&index=1&t=140s
"""
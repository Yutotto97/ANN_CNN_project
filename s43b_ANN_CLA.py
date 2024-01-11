#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:30:10 2020

@author: Yuto
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.backend import clear_session

dataset = pd.read_excel('/factory_process_CLA.xlsx')

X = dataset.iloc[:, 0:41].values

X=np.delete(X,[2,3,4,5,26,27,28,29],1)  #noticed when standardised

y = dataset.iloc[:, 47].values

def Standardize(L):
    m=sum(L)/len(L)
    S=0
    for i in L:
        S+=(i-m)**2
    Sigma=np.sqrt(S/len(L))
    standard_L=[(k-m)/Sigma for k in L]
    return(standard_L)
        
def Normalize(L):
    Max=max(L)
    Min=min(L)
    norm_L=[(g-Min)/(Max-Min) for g in L]
    return(norm_L)
    
(m,n)=np.shape(X)

for x in range (n):
    X[:,x]=Standardize(X[:,x])
    X[:,x]=Normalize(X[:,x])

#y=Standardize(y)


def best_loss_indice(L):  #function which would find the best parameter for (sum of loss minimum)
    a=len(L)
    Sum_min=sum(L[0])
    indice=0
    for b in range (a):
        if sum(L[b])<Sum_min:
            indice=b
            Sum_min=sum(L[b])
    return(indice)
    
def best_acc_indice(L):  #function which would find the best parameter for accuracy (sum of loss maximum)
    a=len(L)
    Sum_max=sum(L[0])
    indice=0
    for b in range (a):
        if sum(L[b])>Sum_max:
            indice=b
            Sum_max=sum(L[b])
    return(indice)

curves=['sigmoid', 'tanh']
Neurons=[10+10*j for j in range (8)]
Hidden_layers=[h for h in range (4)]

L_curves_loss=[]
A_curves_loss=[]
L_curves_acc=[]
A_curves_acc=[]
for p in curves:  #for each activation function
    L_neurons_loss=[]
    A_neurons_loss=[]
    L_neurons_acc=[]
    A_neurons_acc=[]
    for q in Neurons:  #for different number of neurons
        L_layers=[]
        A_layers=[]
        for r in Hidden_layers:   # for different number of hidden layers
                # clear the neuron structure
                clear_session()
                # The part which automatically calculate every case asked
                classifier = Sequential()
                classifier.add(Dense(q, input_dim=n, activation = p))
                for g in range (r):
                    classifier.add(Dense(q, input_dim=q, activation = p))
                classifier.add(Dense(1, input_dim=q, activation = p))

                # Define optimisation paramater
                optimisation = SGD(0.1, 0.9)
                classifier.compile(loss="binary_crossentropy",optimizer=optimisation, metrics = ["binary_accuracy"])

                #visualize the network structure
                #classifier.summary()
                
                # Training network by stocking the result in memory every step
                my_history = classifier.fit(X, y, epochs=100, batch_size=100,verbose=0)
                
                # Keep the result in list
                L_layers.append(my_history.history['loss'])
                A_layers.append(my_history.history['binary_accuracy'])
        
        # This permit to check what is the best parameter for number of layers
        best_loss_layer=best_loss_indice(L_layers)
        best_acc_layer=best_acc_indice(A_layers)
        # The indice corresponds to i such as 1+i layers because already 1 exists
        print('\n'+'number of layers for best parameter (loss) : '+ str(1+best_loss_layer))
        print('number of layers for best parameter (accuracy) : '+ str(1+best_acc_layer)+'\n')
        
        #The most important is to have the best paramater for loss
        L_neurons_loss.append(L_layers[best_loss_layer])
        A_neurons_loss.append(A_layers[best_loss_layer])
        
        #Keep the list with best accuracy to compare in the end
        L_neurons_acc.append(L_layers[best_acc_layer])
        A_neurons_acc.append(A_layers[best_acc_layer])
        
    best_loss_neurons=best_loss_indice(L_neurons_loss)
    
    best_acc_neurons=best_acc_indice(A_neurons_acc)
    
    # The indice corresponds to j in the operation 10+10*j neurons
    print('\n'+'number of neurons for best parameter (loss) : '+ str(10+10*best_loss_neurons))
    print('number of neurons for best parameter (accuracy) : '+ str(10+10*best_acc_neurons)+'\n')
    
    #The most important is to have the best paramater for loss
    L_curves_loss.append(L_neurons_loss[best_loss_neurons])
    A_curves_loss.append(A_neurons_loss[best_loss_neurons])
    
    #Keep the list with best accuracy to compare in the end
    L_curves_acc.append(L_neurons_acc[best_acc_neurons])
    A_curves_acc.append(A_neurons_acc[best_acc_neurons])
    
    
print('\n \n' + 'Conclusion : ')

best_loss_curve=best_loss_indice(L_curves_loss)

best_acc_curve=best_acc_indice(A_curves_acc)

if best_loss_curve!=best_acc_curve or best_loss_neurons!=best_acc_neurons or best_loss_layer!=best_acc_layer:
    print('best paramater for loss : \n' + '  -' + curves[best_loss_curve]
    + '\n' + '  -' + str(10+10*best_loss_neurons) + ' neurons \n' + '  -' +  str(1+best_loss_layer) + ' hidden layers \n')
    
    print('best paramater for accuracy : \n' + '  -' + curves[best_acc_curve]
    + '\n' + '  -' + str(10+10*best_acc_neurons) + ' neurons \n' + '  -' +  str(1+best_acc_layer) + ' hidden layers \n')

else:
    print('best paramater (for loss and accuracy) : \n' + '  -' + curves[best_loss_curve]
    + '\n' + '  -' + str(10+10*best_loss_neurons) + ' neurons \n' + '  -' +  str(1+best_loss_layer) + ' hidden layers \n')
    
#        Unfortunatly my computer couldn't process matplotlib (even by creating new env...)
#        plt.figure()
#        for f in range (len(L_layers)):
#            plt.subplot(211)
#            plt.plot(L_layers[f], label="loss"+'('+str(f+1)+'layers)')
#            plt.subplot(212)
#            plt.plot(A_layers[f], label="accuracy"+'('+str(f+1)+'layers)')
#        plt.legend(loc=5)
#        plt.show()
#        plt.savefig('plot_layers_'+p+'_'+str(q)+'Neurons'+'.png')
        


#print("my_history.history['loss'] is "+ str(L))


X = dataset.iloc[:, 0:41].values
X=np.delete(X,[2,3,4,5,26,27,28,29],1)
(m,n)=np.shape(X)
for x in range (n):
    X[:,x]=Standardize(X[:,x])
    X[:,x]=Normalize(X[:,x])

Inputs= dataset.iloc[:, 41:48].values
Inputs= np.delete(Inputs,[1,2,4,5],1)  #selecting KC1 KC4 KC7

best_classifier=Sequential()
best_classifier.add(Dense(75, input_dim=n, activation = "sigmoid"))
best_classifier.add(Dense(3, input_dim=75, activation = "sigmoid"))
best_classifier.summary()
optimisation = SGD(0.1, 0.9)
best_classifier.compile(loss="binary_crossentropy",optimizer=optimisation, metrics = ["binary_accuracy"])
my_history = best_classifier.fit(X, Inputs, epochs=100, batch_size=100)










indice_delete=[0,1]+[i for i in range (6,26)]+[k for k in range (30,41)]
Accuracies=[]
for d in indice_delete:
    X = dataset.iloc[:, 0:41].values
    X=np.delete(X,[2,3,4,5,26,27,28,29,d],1)
    (m,n)=np.shape(X)
    for x in range (n):
        X[:,x]=Standardize(X[:,x])
        X[:,x]=Normalize(X[:,x])
    clear_session()
    best_classifier=Sequential()
    best_classifier.add(Dense(75, input_dim=n, activation = "sigmoid"))
    best_classifier.add(Dense(1, input_dim=75, activation = "sigmoid"))
    best_classifier.summary()
    optimisation = SGD(0.1, 0.9)
    best_classifier.compile(loss="binary_crossentropy",optimizer=optimisation, metrics = ["binary_accuracy"])
    my_history = best_classifier.fit(X, y, epochs=100, batch_size=100,verbose=0)
    Accuracies.append(my_history.history['binary_accuracy'][-1])
    
plt.plot(indice_delete,Accuracies)
plt.title("Accuracy removing parameter")
plt.show()





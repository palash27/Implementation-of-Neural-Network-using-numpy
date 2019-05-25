# -*- coding: utf-8 -*-
"""
Created on Sat May 25 21:07:12 2019

@author: ratho
"""

import numpy as np

#Input Array
X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
Y = np.array([[1],[1],[0]])

'''
Sigmoid function which is an activation function used to map
output in a range between 0-1 or -1 to 1
other activation function : tanh, ReLu
'''
def sigmoid(x):
    return 1/(1 + np.exp(-x))

'''
Derivative of sigmoid function
This is done to calculate the slope/gradient
'''
def derivatives_sigmoid(x):
    return x*(1 - x)

#variable initializations
epoch = 5000 #setting the number of training iterations
lr = 0.1 #setting the learning rate
inputlayer_neurons = X.shape[1] #No. of features in data set
hiddenlayer_neurons = 3 #number of hidden layer neurons
output_neurons = 1 #number of output layer neurons

#Initializing weights and biases
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))

#running epochs
for i in range(epoch):
    #Forward propagation
    hidden_layer_input1 = np.dot(X,wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    
    #Backward propagation
    E = Y - output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    
print(output)
    
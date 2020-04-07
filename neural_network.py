import pandas as pd
import numpy as np
from functions import *
from copy import copy

class NeuralNetwork:
    def __init__(self, layers, functions):
        '''
        Initializes network's weights and activation functions

        layers: dimensions of each layer
        functions: activation function for each layer

        self parameters
        input: input layer
        weights - array of arrays: stores the weights between each layer
        # output, y: output layer -- cancelled
        functions - list[String]: activation function on each layer
                    (if functions are less than layers, the functions list will be filled with the last function of the list)
        z - array of arrays: contains all the neuron of each hidden layer and output layer
        a - array of arrays: activation value at hidden layer
        '''
        self.layers     = layers
        self.n_layers   = len(self.layers)
        self.input      = layers[0]
        # self.y          = layers[-1] # last layer of the array
        # self.output     = np.zeros(self.y.shape)
        self.functions  = functions
        self.weights, self.z = [], []
        self.partial_deri = []
        self.gradient   = []
        self.a          = [[np.zeros(layers[0])]] # initialization activation value of the input layer
        for i in range(1, self.n_layers):
            # initialization of parameters and random weights
            self.weights.append(np.random.rand(layers[i-1],layers[i]))
            self.z.append(np.zeros(self.layers[i]))
            self.a.append(np.zeros(self.layers[i]))
            #initialization of the partial derivatives
            self.partial_deri.append(np.zeros(self.layers[i]))

        for _ in range(len(self.functions), self.n_layers-1) :
            # to fill the list of function, in case these are less than the number of layer
            self.functions.append(self.functions[-1])


    def feedforward(self):
        for i in range(0, self.n_layers-1):
            self.z[i] = np.dot(self.a[i], self.weights[i])
            vectFunc = np.vectorize(activation(self.functions[i])) # vectorization of the layer's activation function
            self.a[i+1] = vectFunc(self.z[i])

    """fit NN with the training model_selection
    a batch size can be set to obtain a batch stocastic GD"""
    def fit(input_train, output_train, epoch=0, batch_size=0, error_funct = "mean_squared_error"):
        #creating the gradient table
        self.error_funct = error_funct
        if batch_size != 0:
            batches = [input_train[x:x+batch_size] for x in range(0, len(input_train), batch_size)]
            outputs = [output_train[x:x+batch_size] for x in range(0, len(output_train), batch_size)]
        else:
            batches = [input_train]
            outputs = [output_train]
        for i in range(1, self.n_layers):
            self.gradient.append(np.zeros(layers[i-1],layers[i]))
        #The error variable is a list of the means of the errors of the single batches
        self.errors = []
        for batch, output_train in zip(batches, outputs):
            update_batch(batch,output_train)


    """for each batch the gradient table is computed and the weights are updated"""
    def update_batch(self, batch, true_out):
        n_batch = len(batch)
        #batch_error contains the errors of the inputs in the batch
        batch_error = []
        for i in range(n_batch):
            self.intput[:] = batch[i]
            feedforward()
            backpropagation(true_out[i])
            batch_error.append(error_function_getter(self.error_funct)(true_out, a[-1]))
        #this steps calculates the mean of the gradients of the batch and adds it to the weights
        self.gradient /= n_batch
        self.weights += self.gradient
        self.errors.append(np.mean(batch_error))

    def predict(self, input):
        self.intput[:] = input
        feedforward()
        return copy(a[-1])


        """the partial derivative is computed in the same way for every hidden layers
        but in a different way for the output layer that needs to use the
        derivative of the loss function"""
    def backpropagation(self, true_out):
        vectFuncDer = np.vectorize(derivative(self.function[-1]))
        error_funct_derivative = error_f_deriv_getter(self.error_funct)

        self.partial_deri[-1] = error_funct_derivative(true_out, self.a[-1])*vectFuncDer(self.z[-1])

        for i in range(slef.n_layers-3, 0,-1):
            vectFuncDer = np.vectorize(derivative(self.function[i]))
            self.partial_deri[i] = np.dot(self.weights[i+1], self.partial_deri[i+1])*vectFuncDer(self.z[i])

        #updating the gradient matrix
        for i in range(self.n_layers):
            for j in range(self.w.shape[0]):
                for k in range(self.w.shape[1]):
                    self.gradient[i][j][k] += self.partial_deri[j]  * self.a[k]

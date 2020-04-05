import pandas as pd
import numpy as np
import functions

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
        functions: activation function on each layer
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
        self.a          = [[np.zeros(layers[0])]] # initialization activation value of the input layer

        for i in range(1, self.n_layers):
            # random initialization of weights
            self.weights.append(np.random.rand(layers[i-1],layers[i]))
            self.z.append(np.zeros(self.layers[i]))
            self.a.append(np.zeros(self.layers[i]))

        for _ in range(len(self.functions), self.n_layers-1) :
            # to fill the list of function, in case these are less than the number of layer
            self.functions.append(self.functions[-1])

    def feedforward(self):
        for i in range(0, self.n_layers-1):
            self.z[i] = np.dot(self.a[i], self.weights[i])
            vectFunc = np.vectorize(self.functions[i]) # vectorization of the layer's activation function
            self.a[i+1] = vectFunc(self.z[i])

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        for i in range(l-3,-1,-1):
            d[i]=np.dot(w[i+1],d[i+1])*np.vectorize(actderiv)(z[i])

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    #def weights_derivative(self, curr_weight, current_layer ):

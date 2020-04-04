import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dimensions: List[int] = [], functions: List[str] = []):
        '''
        Initializes network's weights and activation functions

        layer_dimensions: dimensions of each layer
        functions: activation function for each layer

        self parameters

        '''
        self.input      = layer_dimensions[0]
        self.y          = layer_dimensions[-1] # last layer of the array
        self.output     = np.zeros(self.y.shape)

        self.weights    = []
        for i in range(1, len(layer_dimensions)):
            w.append(np.random.rand(layer_dimensions[i-1],layer_dimensions[i]))


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        for i in range(l-3,-1,-1):
            d[i]=np.dot(w[i+1],d[i+1])*np.vectorize(actderiv)(z[i])

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def weights_derivative(self, curr_weight, current_layer ):

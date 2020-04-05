from neural_network import *
import functions as f

'''
initializing a NN with the folloging features:
[4 input neurons] -- [3] -- [2] -- [3 output neurons]
'''
layers = [4,3,2,3]
functions = [f.relu, f.relu, f.softmax]
x = NeuralNetwork(layers=layers, functions=functions)
x.feedforward()
# Compila! funzionaerà  ¯\(o_o)/¯

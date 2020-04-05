import numpy as np
'''
Sigmoid is used in binary classification problem [-1,1]
ReLu is mainly used to perform threshold operation to each input elem where val <0 is set to 0 (better performance in generalization)
Softmax is mainly used for multi-class problem
'''
#def sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x): return (1/(1 + np.exp(-x)))

def relu(x): return np.maximum(0,x)

def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(np.exp(x))
    return exp/exp_sum

def sigmoid_derivative(x): return (sigmoid(x)*(1-sigmoid(x)))

def reluderiv(x):
  if x >= 0: return 1
  else: return 0

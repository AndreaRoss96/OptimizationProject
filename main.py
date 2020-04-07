import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
import random


def main():
    ds = pd.read_csv("wine.data")
    seed = 42
    #seed = random.randint(0,100)
    Y = ds["Class"]
    X = ds.drop("Class", axis = 1)
    i_train, i_test, o_train, o_test = train_test_split(
            X, Y, train_size=0.33, random_state = seed)
    nn = NeuralNetwork([13,5,5,3], ["relu", "relu", "relu"])
    nn.fit(i_train, o_train, batch_size=5)


if __name__ == "__main__":
    main()

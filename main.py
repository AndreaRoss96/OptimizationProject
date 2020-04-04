import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from neural_network import NeuralNetwork


def main():
    ds = pd.read_csv("wine.data")
    figure = plt.figure(figsize=(5,5))
    print(ds.head())
    Y = ds["Class"]
    X = ds.drop("Class", axis = 1)
    nn = NeuralNetwork()


if __name__ == "__main__":
    main()

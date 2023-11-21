from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import Dataset
from src.network.perceptron import Perceptron


def parser_args():
    args = ArgumentParser()
    args.add_argument("--dataset", "-d", type=str, required=True)
    args.add_argument("--input-shape", "-i", type=int, required=True)

    return args.parse_args()


def test():
    args = parser_args()
    dataset = Dataset(filename=args.dataset)
    X, Y = dataset.load_from_xls(input_shape=args.input_shape)
    model = Perceptron(input_shape=args.input_shape)
    model.load("models/weights.npy")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    weights = model.weights

    x_max = X[:, 0].max()
    x_min = X[:, 0].min()

    x = np.linspace(x_min, x_max, 100)

    y_max = X[:, 1].max()
    y_min = X[:, 1].min()

    y = np.linspace(y_min, y_max, 100)

    Xs, Ys = np.meshgrid(x, y)
    Zs = (weights[0] * Xs - weights[1] * Ys - weights[2]) / weights[3]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(Xs, Ys, Zs, alpha=0.5)

    for _x, y in zip(X, Y):
        if y > 0:
            ax.scatter(_x[0], _x[1], _x[2], color="green")
        else:
            ax.scatter(_x[0], _x[1], _x[2], color="red")

    plt.show()


if __name__ == "__main__":
    test()

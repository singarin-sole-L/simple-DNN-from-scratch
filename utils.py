import numpy as np
from matplotlib import pyplot as plt


def sigmoide(Z):
    return 1 / (1 + np.exp(-Z))


def log_loss(A, y, m, eps=10E-15):
    return (-1 / m) * np.sum(y * np.log(A + eps) + (1 - y) * np.log(1 - A + eps))


def visualize_cost_function(epochs, cost_func_values):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost Function')
    ax.set_title('Evolution of the Cost Function')
    line, = ax.plot(epochs, cost_func_values)
    plt.draw()
    plt.pause(0.1)


def calculate_accuracy(DNN, X, y, threshold=0.5):
    predictions = DNN.predict(X, threshold)
    return np.mean(predictions == y)

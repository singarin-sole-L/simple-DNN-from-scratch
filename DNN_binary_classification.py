# We implement many logisticRegression for create our DNN
# We use only sigmoide (activate function), Log Loss (cost function) and normal Gradient Descent (GD)
# Problem of vanishing gradient here (must use ReLU) and SDG if we have a lot of data
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
from utils import sigmoide, log_loss, visualize_cost_function, calculate_accuracy


class Dnn:
    def __init__(self, sequential: list):
        self.sequential = sequential
        self.nb_layers = len(sequential) - 1
        self.W, self.b = self.initialize_params()
        self.A = None

    def initialize_params(self):
        weights = [np.random.randn(self.sequential[layer + 1], self.sequential[layer])
                   for layer in range(self.nb_layers)]
        bias = [np.zeros((self.sequential[layer + 1], 1))
                for layer in range(self.nb_layers)]
        return weights, bias

    def forward(self, X):
        self.A = [X]
        for layer in range(self.nb_layers):
            Z = self.W[layer] @ self.A[layer] + self.b[layer]
            self.A.append(sigmoide(Z))
        return self.A[-1]

    def backward(self, y: np.array, m: int):
        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(b) for b in self.b]
        dZ = [np.zeros_like(self.A[k]) for k in range(1, len(self.A))]
        dZ[-1] = self.A[-1] - y
        dW[-1] = (1 / m) * dZ[-1] @ self.A[-2].T
        db[-1] = (1 / m) * np.sum(dZ[-1], axis=1)
        for layer in range(self.nb_layers - 2, -1, -1):
            dZ[layer] = self.W[layer + 1].T @ dZ[layer + 1] * self.A[layer + 1] * (1 - self.A[layer + 1])
            dW[layer] = (1 / m) * dZ[layer] @ self.A[layer].T
            db[layer] = (1 / m) * np.sum(dZ[layer], axis=1)
        return dW, db

    def GD(self, dW, db, alpha=0.01):
        for layer in range(self.nb_layers):
            self.W[layer] = self.W[layer] - alpha * dW[layer]
            self.b[layer] = self.b[layer] - alpha * db[layer]

    def train(self, X: np.ndarray, y: np.array, nb_epochs: int, patience=10, min_delta=0.001, visualization=True):
        X = self.validate_input(X, y)
        cost_func_values = []
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in tqdm(range(nb_epochs)):
            self.forward(X)
            current_loss = log_loss(self.A[-1], y, X.shape[1])
            if visualization:
                cost_func_values.append(current_loss)
                visualize_cost_function(range(len(cost_func_values)), cost_func_values)
            # Early stopping logic
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info(f'Early stopping at epoch {epoch}')
                break
            dW, db = self.backward(y, X.shape[1])
            self.GD(dW, db)
        accuracy = calculate_accuracy(self, X, y)
        logging.info(f'Final Accuracy: {accuracy}')
        if visualization:
            plt.ioff()
            plt.show()

    def predict(self, X: np.ndarray, threshold=0.5):
        X = self.validate_input(X)
        prediction = self.forward(X)
        return (prediction >= threshold).astype(int)

    def validate_input(self, X, y=None):
        if X.shape[0] != self.sequential[0]:
            raise ValueError(f"Input features must match first layer size {self.sequential[0]}")
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        if y is not None:
            # Ensure y is binary for logistic regression
            if not np.all(np.isin(y, [0, 1])):
                raise ValueError("Labels must be binary (0 or 1)")
            if y.shape[1] != X.shape[1]:
                raise ValueError("Number of labels must match number of samples")

        return X_normalized

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        self.weights = np.zeros(training_inputs.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)
import numpy as np


def null(x):
    return 0


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def hyper_tangent(x):
    return np.tanh(x)


def cosine(x):
    return np.cos(x)


def gaussian(x):
    return np.exp(-(x ** 2 / 2))


class NeuralNetwork:
    def __init__(self, x, y, layers, weights=None):
        self.input = x
        self.weights = []
        self.layers = layers
        self.activations_functions = [null, sigmoid, hyper_tangent, cosine, gaussian]
        start = True
        index = 0
        for layer in self.layers:
            if start:
                self.weights.append(np.random.rand(self.input.shape[1], layer[0]))
            else:
                self.weights.append(np.random.rand(self.layers[index - 1][0], layer[0]))
            start = False
            index += 1
        if weights is not None:
            index = 0
            x = 0
            for layer in self.layers:
                if index == 0:
                    for input_index in range(len(self.input[0])):
                        for neurons_index in range(layer[0]):
                            self.weights[index][input_index, neurons_index] = weights[x]
                            x += 1
                else:
                    for layer_index in range(self.layers[index - 1][0]):
                        for neurons_index in range(layer[0]):
                            self.weights[index][layer_index, neurons_index] = weights[x]
                            x += 1
                index += 1
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        value_layers = []
        index = 0
        for layer in self.layers:
            if len(value_layers) == 0:
                value_layers.append(layer[1](np.dot(self.input, self.weights[index])))
            else:
                value_layers.append(layer[1](np.dot(value_layers[-1], self.weights[index])))
            index += 1
        self.output = value_layers[-1]

    def get_weights_as_vector(self):
        vector = []
        for x in self.weights:
            for y in x:
                for z in y:
                    vector.append(z)
        return vector

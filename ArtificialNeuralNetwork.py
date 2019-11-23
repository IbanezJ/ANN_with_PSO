import random
import math


def null(x, deriv=False):
    return 0


def sigmoid(x, deriv=False):
    if not deriv:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(-x) / ((1.0 + math.exp(-x)) ** 2)


def hyper_tangent(x, deriv=False):
    if not deriv:
        return math.tanh(x)
    else:
        return 1 - (math.tan(x) ** 2)


def cosine(x, deriv=False):
    if not deriv:
        return math.cos(x)
    else:
        return (math.sin(x)) * -1


def gaussian(x, deriv=False):
    if not deriv:
        return math.exp(-1 * (x**2 / 2))


class ArtificialNeuralNetwork:
    def __init__(self, nb_inputs, nb_layers, nb_neurons, activation_functions, known_weights=None):
        self.nb_inputs = nb_inputs
        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.activation_functions = activation_functions
        self.weights = []
        self.layers = []
        if known_weights is None:
            for i in range(nb_layers):
                all_layer_weights = []
                for x in range(nb_inputs if i == 0 else nb_neurons[i - 1]):
                    curr_weights = []
                    for y in range(nb_neurons[i]):
                        curr_weights.append(random.random())
                    all_layer_weights.append(curr_weights)
                self.weights.append(all_layer_weights)
        else:
            iter_weights = 0
            for i in range(nb_layers):
                all_layer_weights = []
                for x in range(nb_inputs if i == 0 else nb_neurons[i - 1]):
                    curr_weights = []
                    for y in range(nb_neurons[i]):
                        curr_weights.append(known_weights[iter_weights])
                        iter_weights += 1
                    all_layer_weights.append(curr_weights)
                self.weights.append(all_layer_weights)

    def feed_forward(self, inputs):
        if len(inputs) != self.nb_inputs:
            print("Wrong number of inputs (needs " + self.nb_inputs + ").")
            return
        self.layers = []
        step = 0
        for layer_weights in self.weights:
            layer = []
            i = 0
            while i < len(layer_weights[0]):
                inputs_sum = 0.0
                y = 0
                for neuron_weights in layer_weights:
                    inputs_sum += neuron_weights[i] * (inputs[y] if len(self.layers) == 0 else self.layers[-1][y][1])
                    y += 1
                layer.append((inputs_sum, self.activation_functions[step](inputs_sum)))
                i += 1
            self.layers.append(layer)
            step += 1
        print("ann result =", self.layers[-1][0][1])

    def get_error(self, right_output):
        error = right_output - self.layers[-1][0][1]
        return error

    def get_weights_as_vector(self):
        weights_vector = []
        for weights in self.weights:
            for neuron_weights in weights:
                for element in neuron_weights:
                    weights_vector.append(element)
        return weights_vector


"""
    def back_propagation(self, outputs):
        if self.nb_neurons[-1] != len(outputs):
            print("Wrong number of outputs (needs " + self.nb_neurons[-1] + ").")
            return
        sum_errors = 0.0
        for x in range(len(outputs)):
            sum_errors += outputs[x] - self.layers[-1][x][1]
        x = -1
        print("layers =", self.layers)
        print("weights =", self.weights)
        deltas = []
        for i in range(len(self.layers)):
            delta_layer = []
            for z_and_a in self.layers[x]:
                delta_layer.append(sigmoid(z_and_a[0]) * (sum_errors if x == -1 else 2))  # à modifier
            print(self.weights[x])
            print(self.layers[x])
            x -= 1
"""

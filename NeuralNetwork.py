import numpy as np


def null(x):
    return 0


# range [0, 1]
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# range[-1, 1]
def hyper_tangent(x):
    return np.tanh(x)


# range[-1, 1]
def cosine(x):
    return np.cos(x)


# range[0, 1]
def gaussian(x):
    return np.exp(-(x ** 2 / 2))


# my artificial neural network represented by a class
class NeuralNetwork:
    """
    the class can have 4 variables:
    x: all inputs as a numpy array where each line is a list of number corresponding to input variables
    y: all outputs as a numpy array where each line is an output
    layers: a tuples array where each case is a layer. In the tuple there are the number of neurons and the activation
     function of the layer among the one defined above
    weights: (optional) a simple array to represent a vector containing all the weights you need in the network.
     If the argument is given it creates a network with these weights. If not it creates random weights
    """

    def __init__(self, x, y, layers, weights=None):
        self.input = x
        self.weights = []
        self.layers = layers
        self.activations_functions = [null, sigmoid, hyper_tangent, cosine, gaussian]
        start = True
        index = 0
        for layer in self.layers:
            # creates weights as matrix of shape (number of neurons in the current layer (or of inputs),
            # number of neurons in the next layer)
            if start:
                self.weights.append(np.random.rand(self.input.shape[1], layer[0]))
            else:
                self.weights.append(np.random.rand(self.layers[index - 1][0], layer[0]))
            start = False
            index += 1
        # if the weights' vector is given replace the weights by the ones given
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
        #  keeps the desired outputs in self.y and sets the first outputs to 0
        self.y = y
        self.output = np.zeros(self.y.shape)

    #  feed_forward function
    def feed_forward(self):
        value_layers = []
        index = 0
        # for each layer (starting by the one "closest" to the inputs), apply the activation function linked
        # to this layer
        for layer in self.layers:
            if len(value_layers) == 0:
                value_layers.append(layer[1](np.dot(self.input, self.weights[index])))
            else:
                value_layers.append(layer[1](np.dot(value_layers[-1], self.weights[index])))
            index += 1
        # self.output keeps the result given by the last layer (corresponding to the output layer)
        self.output = value_layers[-1]

    # turns the weights into a vector which ban be used in the constructor
    def get_weights_as_vector(self):
        vector = []
        for x in self.weights:
            for y in x:
                for z in y:
                    vector.append(z)
        return vector

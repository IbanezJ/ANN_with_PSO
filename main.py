import numpy as np

from PSO import PSO
from NeuralNetwork import NeuralNetwork, sigmoid

if __name__ == "__main__":

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    layers = [(4, sigmoid), (1, sigmoid)]
    ann = NeuralNetwork(X, Y, layers)

    pso = PSO(len(ann.get_weights_as_vector()), 100, 0.5, 2.0, 2.0, 1.0)

    for _ in range(100):
        pso.train(X, Y, layers)
    new_ann = NeuralNetwork(X, Y, [(4, sigmoid), (1, sigmoid)], list(pso.best))
    new_ann.feed_forward()
    print("FINAL ANN OUTPUT =")
    print(new_ann.output)

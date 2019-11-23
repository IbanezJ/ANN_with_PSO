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
    print("PSO BEST =")
    print(pso.best)
    print("FINAL ANN OUTPUT =")
    print(new_ann.output)

    """
    [ 0.04959152  0.61688732  0.96189469  0.01402569 -0.29641678 -0.9668396
    -0.35432748 -0.96222846  0.21140214  0.1106617   0.63547942 -0.22265218
    -0.70206729 -0.6996286  -0.7496279   0.77079024]
    """
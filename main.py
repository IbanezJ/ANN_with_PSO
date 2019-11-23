import numpy as np

from PSO import PSO
from NeuralNetwork import NeuralNetwork, sigmoid

if __name__ == "__main__":

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    ann = NeuralNetwork(X, y, [(4, sigmoid), (1, sigmoid)])

    pso = PSO(len(ann.get_weights_as_vector()), 50, 0.5, 2.0, 2.0, 1.0)

    while pso.fitness != 0:
        pso.train()
    new_ann = NeuralNetwork(X, y, [(4, sigmoid), (1, sigmoid)], list(pso.best))
    print("THIS =", list(pso.best))
    print("AND THIS =", new_ann.weights)
    new_ann.feed_forward()
    print(new_ann.output)
    print(pso.best)

    """
    [ 0.04959152  0.61688732  0.96189469  0.01402569 -0.29641678 -0.9668396
    -0.35432748 -0.96222846  0.21140214  0.1106617   0.63547942 -0.22265218
    -0.70206729 -0.6996286  -0.7496279   0.77079024]
    """
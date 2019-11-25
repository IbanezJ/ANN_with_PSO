import numpy as np
import matplotlib.pyplot as plt

from PSO import PSO
from NeuralNetwork import NeuralNetwork, sigmoid, hyper_tangent, cosine, gaussian


def readfiles():
    files_names = [
        ("./data/2in_xor.txt", "xor"),
        ("./data/2in_complex.txt", "complex"),
        ("./data/1in_tanh.txt", "tanh"),
        ("./data/1in_linear.txt", "linear"),
        ("./data/1in_cubic.txt", "cubic"),
        ("./data/1in_sine.txt", "sine")
    ]
    inputs = {}
    for filename in files_names:
        f = open(filename[0], "r")
        lines = f.readlines()
        inputs[filename[1]] = []
        for line in lines:
            if line[-1] == '\n':
                sep_line = line[:-1].split(" ")
            else:
                sep_line = line.split(" ")
            if len(sep_line) == 1:
                sep_line = sep_line[0].split("\t")
            x = len(sep_line) - 1
            while x >= 0:
                if sep_line[x] == '':
                    del sep_line[x]
                x -= 1
            numeric_values = [float(n) for n in sep_line]
            inputs[filename[1]].append(numeric_values)
        f.close()
    return inputs


def get_ann(inputs, layers, swarm_size, informants_size, velocity_retained, personal, social, jump_size, epochs):
    x = []
    y = []
    for inp in inputs:
        if len(inp) == 3:
            x.append([inp[0], inp[1]])
            y.append([inp[2]])
        else:
            x.append([inp[0]])
            y.append([inp[1]])
    X = np.array(x)
    Y = np.array(y)

    ann = NeuralNetwork(X, Y, layers)
    print(len(ann.get_weights_as_vector()))
    sum_fitness = 0.0
    for z in range(10):
        pso = PSO(len(ann.get_weights_as_vector()), swarm_size, informants_size, velocity_retained, personal, social, jump_size)

        all_best = []
        x_plot = []
        for _ in range(epochs):
            pso.train(X, Y, layers)
            all_best.append(pso.best_fitness_value)
            x_plot.append(_)
        fig, ax = plt.subplots()
        ax.plot(x_plot, all_best)
        plt.show()
        new_ann = NeuralNetwork(X, Y, layers, list(pso.best))
        new_ann.feed_forward()
        desired_output = [list(n) for n in list(Y)]
        ann_output = [list(n) for n in list(new_ann.output)]
        results = []
        x = 0
        while x in range(len(desired_output)):
            results.append((desired_output[x][0], ann_output[x][0]))
            x += 1
        print(results)
        sum_fitness += pso.best_fitness_value
    print(sum_fitness / 10)
    return new_ann


if __name__ == "__main__":

    """
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    """

    inputs = readfiles()
    #get_mean_error(inputs["cubic"], [(10, hyper_tangent), (1, hyper_tangent)], 100, 10, 0.5, 1.655, 1.655, 1)
    #get_mean_error(inputs["tanh"], [(10, hyper_tangent), (1, hyper_tangent)], 100, 10, 0.5, 1.655, 1.655, 1, 10)
    #get_mean_error(inputs["sine"], [(25, cosine), (1, cosine)], 100, 10, 0.5, 1.655, 1.655, 0.1, 100)
    #get_mean_error(inputs["linear"], [(4, sigmoid), (1, sigmoid)], 100, 10, 1, 0, 4, 0.1, 100)
    #get_mean_error(inputs["xor"], [(2, sigmoid), (1, sigmoid)], 100, 10, 0.5, 1.655, 1.655, 1)
    get_ann(inputs["complex"], [(2, sigmoid), (1, sigmoid)], 100, 10, 0.5, 1.655, 1.655, 1, 100)



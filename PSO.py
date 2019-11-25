import random

import numpy as np

from NeuralNetwork import NeuralNetwork, sigmoid


# class corresponding to a particle in the PSO
class Particle:
    # just needs a position and an initial velocity
    # also keeps track of the best_pos the particle was
    def __init__(self, pos, velocity):
        self.pos = pos
        self.velocity = velocity
        self.best_pos = pos
        self.best_pos_value = None
        self.informants_best = None


# my particle swarm optimisation
class PSO:
    """
    the class can have 7 arguments:
    dimension_size: the number of dimensions of particles position and velocity (the vector length)
    swarm_size: the number of particles in the swarm
    informants_size: size of a pack of particle
    alpha: proportion of velocity to be retained
    beta: proportion of personal best to be retained
    gamma: proportion of the informants' best to be retained
    epsilon: jump size of a particle
    delta: (optional) proportion of all the particles' best
    """

    def __init__(self, dimension_size, swarm_size, informants_size, alpha, beta, gamma, epsilon, delta=0):
        self.dimension_size = dimension_size
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.pop = {}
        self.informants = {}
        for i in range(informants_size):
            self.informants[i] = 9999999
        self.fitness = 99999999
        # sets the initial postions to random values and initial velocity to zero
        informant = 0
        for _ in range(swarm_size):
            initial_position = np.random.random_sample((dimension_size,)) * 2 - 1
            initial_velocity = np.random.random_sample((dimension_size,)) * 2 - 1
            self.pop[Particle(initial_position, initial_velocity)] = informant % informants_size
            informant += 1
        # best and best_fitness_values are used to keeps track of the best
        self.best = None
        self.best_fitness_value = None

    # trains the PSO given the inputs and outputs desired with the layers you want in the neural network
    def train(self, X, Y, layers, verbose=False):
        for particle in self.pop:
            # create the neural networks with the particle positions as weights and makes one feed forward to get the
            # output
            new_ann = NeuralNetwork(X, Y, layers, list(particle.pos))
            new_ann.feed_forward()
            # displays the output of the previously created network
            if verbose:
                print("CURRENT ANN OUTPUT =")
                print([list(n) for n in list(new_ann.output)])
            # displays the desired output
            if verbose:
                print("DESIRED OUTPUT =")
                print([list(n) for n in list(Y)])
            #  calculates the fitness of this particle (it's the mean squared error)
            fitness = (np.sum((Y - new_ann.output) ** 2)) / len(list(Y))
            self.fitness = fitness
            if verbose:
                print("CURRENT FITNESS (mean squared error) =")
                print(fitness)

            # saves the global best positions and the best value
            if self.best_fitness_value is None or fitness < self.best_fitness_value:
                self.best = particle.pos
                self.best_fitness_value = fitness
            # saves the particle best
            if particle.best_pos_value is None or fitness < particle.best_pos_value:
                particle.best_pos_value = fitness
                particle.best_pos = particle.pos
            # saves the informants' best
            if particle.informants_best is None or fitness < self.informants[self.pop[particle]]:
                particle.informants_best = particle.pos
                self.informants[self.pop[particle]] = fitness

        # changes velocity
        for particle in self.pop:
            b = random.random() * self.beta
            c = random.random() * self.gamma
            d = random.random() * self.delta
            particle.velocity = self.alpha * particle.velocity + b * (particle.best_pos - particle.pos) + c * (
                        self.best - particle.pos) + d * (self.best - particle.pos)

        # updates particle position
        for particle in self.pop:
            particle.pos += self.epsilon * particle.velocity

import random

import numpy as np

from NeuralNetwork import NeuralNetwork, sigmoid


class Particle:
    def __init__(self, pos, velocity):
        self.pos = pos
        self.velocity = velocity
        self.best_pos = pos
        self.best_pos_value = None


class PSO:
    def __init__(self, dimension_size, swarm_size, alpha, beta, gamma, epsilon, delta=0):
        self.dimension_size = dimension_size
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.pop = []
        self.fitness = 99999999
        for _ in range(swarm_size):
            initial_position = np.random.random_sample((dimension_size,)) * 2 - 1
            initial_velocity = np.zeros(dimension_size)
            self.pop.append(Particle(initial_position, initial_velocity))
        self.best = None
        self.best_fitness_value = None

    def train(self, X, Y, layers):
        for particle in self.pop:
            new_ann = NeuralNetwork(X, Y, layers, list(particle.pos))
            new_ann.feed_forward()
            print("CURRENT ANN OUTPUT =", new_ann.output)
            fitness = (np.sum((Y - new_ann.output) ** 2)) / 4
            self.fitness = fitness
            print("CURRENT FITNESS =", fitness)

            if self.best_fitness_value is None or fitness < self.best_fitness_value:
                self.best = particle.pos
                self.best_fitness_value = fitness
            if particle.best_pos_value is None or fitness < particle.best_pos_value:
                particle.best_pos_value = fitness
                particle.best_pos = particle.pos

        for particle in self.pop:
            b = random.random() * self.beta
            c = random.random() * self.gamma
            particle.velocity = self.alpha * particle.velocity + b * (particle.best_pos - particle.pos) + c * (self.best - particle.pos)

        for particle in self.pop:
            particle.pos += self.epsilon * particle.velocity

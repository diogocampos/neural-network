import numpy as np


class NeuralNetwork:
    def __init__(self, regularization, structure):
        self.regularization = regularization
        self.structure = tuple(structure)


    def set_weights(self, weights):
        weights = [np.array(theta) for theta in weights]

        # verifica match dos pesos fornecidos com a estrutura da rede
        for i, theta in enumerate(weights):
            num_neurons = self.structure[i + 1]
            num_inputs_per_neuron = self.structure[i] + 1
            assert theta.shape == (num_neurons, num_inputs_per_neuron)

        self.weights = weights


    def propagate(self, features):
        activations = features.T
        for theta in self.weights:
            bias = np.ones(activations.shape[1])
            a = np.vstack((bias, activations))
            z = theta.dot(a)
            activations = 1.0 / (1.0 + np.exp(-z))
        return activations.T

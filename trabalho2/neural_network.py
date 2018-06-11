import numpy as np


class NeuralNetwork:

    def __init__(self, lambda_, structure):
        # Inicializa uma rede neural.
        # - lambda_: fator de regularização
        # - structure: lista dos tamanhos de cada camada da rede

        self.lambda_ = lambda_
        self.structure = tuple(structure)


    def set_weights(self, weights):
        # Define os pesos dos neurônios da rede.
        # - weights: lista de matrizes theta (um neurônio por linha, com bias)

        weights = [np.array(theta) for theta in weights]

        # verifica match entre os pesos fornecidos e a estrutura da rede
        for i, theta in enumerate(weights):
            num_neurons = self.structure[i + 1]
            num_inputs_per_neuron = self.structure[i] + 1
            assert theta.shape == (num_neurons, num_inputs_per_neuron)

        self.weights = weights


    def propagate(self, features):
        # Calcula as ativações da rede para um conjunto de instâncias.
        # - features: matriz de atributos de instância (uma instância por linha)
        # Retorna uma matriz com as ativações (uma instância por linha).

        activations = features.T

        for theta in self.weights:
            bias = np.ones(activations.shape[1])
            a = np.vstack((bias, activations))
            z = theta.dot(a)
            activations = 1.0 / (1.0 + np.exp(-z))

        return activations.T

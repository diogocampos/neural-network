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
        # - features: matriz de atributos de instância (instâncias nas colunas)
        # Retorna uma matriz com as ativações (uma instância por coluna).

        activations = features

        for theta in self.weights:
            bias = np.ones(activations.shape[1])
            a = np.vstack((bias, activations))
            z = theta.dot(a)
            activations = 1.0 / (1.0 + np.exp(-z))

        return activations


    def total_error(self, outputs, activations):
        # Calcula o erro total (J), com regularização.
        # - outputs: matriz de saídas esperadas (instâncias nas colunas)
        # - activations: matriz de saídas preditas (instâncias nas colunas)

        assert outputs.shape == activations.shape

        y = outputs
        f = activations
        n = y.shape[1]  # número de instâncias
        error = np.sum(-y * np.log(f) - (1.0 - y) * np.log(1.0 - f)) / n

        if self.lambda_ == 0.0:
            regularization = 0.0
        else:
            # cálculo da regularização, sem os pesos de bias (primeira coluna)
            weights = [theta[:, 1:] for theta in self.weights]
            sum_squares = sum(np.sum(np.square(theta)) for theta in weights)
            regularization = sum_squares * self.lambda_ / (2.0 * n)

        return error + regularization

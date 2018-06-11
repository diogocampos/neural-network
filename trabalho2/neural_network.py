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
        # Retorna uma lista de matrizes com as ativações de cada camada.
        #   * uma matriz por camada da rede, incluindo as entradas
        #   * cada coluna tem as ativações da camada para uma instância
        #   * todas as matrizes incluem uma linha de bias, menos a última

        a = features
        activations = []

        for theta in self.weights:
            # adiciona a linha dos neurônios de bias
            bias = np.ones(a.shape[1])
            a = np.vstack((bias, a))
            activations.append(a)

            z = theta.dot(a)
            a = 1.0 / (1.0 + np.exp(-z))

        activations.append(a)
        return activations


    def backpropagate(self, expectations, activations):
        # Calcula os gradientes finais para um conjunto de instâncias.
        # - expectations: matriz de saídas esperadas (instâncias nas colunas)
        # - activations: lista de matrizes das ativações dos neurônios
        #     (mesmo formato da saída do método `propagate`)
        # Retorna uma lista de matrizes de gradientes, com o mesmo formato que
        # a lista de matrizes dos pesos da rede.

        y = expectations
        f = activations[-1]
        assert y.shape == f.shape

        n = y.shape[1]  # número de instâncias
        delta = f - y
        gradients = []

        for i in range(len(self.weights) - 1, -1, -1):
            theta = self.weights[i]
            a = activations[i]

            gradient = delta.dot(a.T)
            delta = theta.T.dot(delta) * a * (1.0 - a)
            delta = delta[1:]  # descarta a primeira linha

            if self.lambda_ == 0.0:
                regularization = 0.0
            else:
                temp = np.array(theta)
                temp[:, 0] = 0.0  # zera a primeira coluna
                regularization = self.lambda_ * temp

            gradient = (gradient + regularization) / n
            gradients.append(gradient)

        gradients.reverse()
        return gradients


    def total_error(self, expectations, predictions):
        # Calcula o erro total (J), com regularização.
        # - expectations: matriz de saídas esperadas (instâncias nas colunas)
        # - predictions: matriz de saídas preditas (instâncias nas colunas)

        y = expectations
        f = predictions
        assert y.shape == f.shape

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

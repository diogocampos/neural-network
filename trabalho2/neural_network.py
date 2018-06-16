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


    def set_random_weights(self):
        # Inicializa os pesos dos neurônios aleatoriamente.

        self.weights = []

        for i in range(len(self.structure) - 1):
            num_neurons = self.structure[i + 1]
            num_inputs_per_neuron = self.structure[i] + 1

            theta = np.random.random((num_neurons, num_inputs_per_neuron))
            self.weights.append(theta)


    def propagate(self, features):
        # Calcula as ativações da rede para um conjunto de instâncias.
        # - features: matriz de atributos de instância (instâncias nas linhas)
        # Retorna uma lista de matrizes com as ativações de cada camada.
        #   * uma matriz por camada da rede, incluindo as entradas
        #   * cada linha tem as ativações da camada para uma instância
        #   * todas as matrizes (menos a última) incluem uma coluna de bias

        a = features.T
        activations = []

        for theta in self.weights:
            # adiciona os neurônios de bias
            bias = np.ones(a.shape[1])
            a = np.vstack((bias, a))
            activations.append(a.T)

            z = theta.dot(a)
            a = 1.0 / (1.0 + np.exp(-z))

        activations.append(a.T)
        return activations


    def _backpropagate(self, expectations, activations):
        # Calcula os gradientes finais para um conjunto de instâncias.
        # - expectations: matriz de saídas esperadas (instâncias nas linhas)
        # - activations: lista de matrizes das ativações dos neurônios
        #     (mesmo formato da saída do método `propagate`)
        # Retorna uma lista de matrizes com os gradientes de cada camada,
        # com o mesmo formato que a lista de matrizes dos pesos da rede.

        y = expectations.T
        f = activations[-1].T
        assert y.shape == f.shape

        n = y.shape[1]  # número de instâncias
        delta = f - y
        gradients = []

        for i in range(len(self.weights) - 1, -1, -1):
            theta = self.weights[i]
            a = activations[i].T

            gradient = delta.dot(a.T)
            delta = theta.T.dot(delta) * a * (1.0 - a)
            delta = delta[1:]  # descarta a primeira linha (bias)

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


    def gradients(self, dataset):
        # Calcula os gradientes por backpropagation.
        # Retorna uma lista de matrizes com os gradientes de cada camada,
        # com o mesmo formato que a lista de matrizes dos pesos da rede.

        activations = self.propagate(dataset.features)
        gradients = self._backpropagate(dataset.expectations, activations)
        return gradients


    def numeric_gradients(self, dataset, epsilon=1e-6):
        # Calcula estimativas numéricas dos gradientes.
        # Retorna uma lista de matrizes com os gradientes de cada camada,
        # com o mesmo formato que a lista de matrizes dos pesos da rede.

        gradients = [np.empty(theta.shape) for theta in self.weights]

        for t, theta in enumerate(self.weights):
            for n, neuron in enumerate(theta):
                for w, weight in enumerate(neuron):

                    self.weights[t][n, w] = weight + epsilon
                    activations = self.propagate(dataset.features)
                    j2 = self.total_error(dataset.expectations, activations[-1])

                    self.weights[t][n, w] = weight - epsilon
                    activations = self.propagate(dataset.features)
                    j1 = self.total_error(dataset.expectations, activations[-1])

                    gradients[t][n, w] = (j2 - j1) / (2.0 * epsilon)
                    self.weights[t][n, w] = weight

        return gradients


    def total_error(self, expectations, predictions):
        # Calcula o erro total (J), com regularização.
        # - expectations: matriz de saídas esperadas (instâncias nas linhas)
        # - predictions: matriz de saídas preditas (instâncias nas linhas)

        y = expectations.T
        f = predictions.T
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

import numpy as np

from .dataset import join_datasets


class NeuralNetwork:

    def __init__(self, lambda_, structure):
        """
        Inicializa uma rede neural.
          - lambda_: fator de regularização
          - structure: lista dos tamanhos de cada camada da rede
        """

        self.lambda_ = lambda_
        self.structure = tuple(structure)


    def set_weights(self, weights):
        """
        Define os pesos dos neurônios da rede.
          - weights: lista de matrizes theta (um neurônio por linha, com bias)
        """

        weights = [np.array(theta) for theta in weights]

        # verifica match entre os pesos fornecidos e a estrutura da rede
        for i, theta in enumerate(weights):
            num_neurons = self.structure[i + 1]
            num_inputs_per_neuron = self.structure[i] + 1
            assert theta.shape == (num_neurons, num_inputs_per_neuron)

        self.weights = weights


    def set_random_weights(self):
        """
        Inicializa os pesos dos neurônios aleatoriamente.
        """

        self.weights = []

        for i in range(len(self.structure) - 1):
            num_neurons = self.structure[i + 1]
            num_inputs_per_neuron = self.structure[i] + 1

            size = np.sqrt(2.0 / num_inputs_per_neuron)
            random = np.random.random((num_neurons, num_inputs_per_neuron))
            theta = random * 2.0 * size - size  # valores entre -size e +size
            self.weights.append(theta)


    def train(self, batches, alpha=1e-3, beta=0.5, mindelta=1e-9, skip=100):
        """
        Treina a rede neural com um dataset.
          - batches: lista de datasets de treinamento
          - alpha: taxa de aprendizado
          - beta: fator do método do momento
          - mindelta: critério de parada (variação mínima do erro total)
          - skip: número de iterações entre cada `yield` do erro J
        Retorna um gerador que fornece o erro J após cada `skip` iterações.
        """

        self.set_random_weights()

        combined = join_datasets(batches)
        z = [0 for theta in self.weights]  # método do momento
        prev_j = -1
        counter = 0

        while True:
            for batch in batches:
                activations = self.propagate(batch.features)
                gradients = self._backpropagate(batch.expectations, activations)

                # atualiza os pesos
                for i in range(len(self.weights)):
                    z[i] = beta * z[i] + gradients[i]
                    self.weights[i] -= alpha * z[i]

            activations = self.propagate(combined.features)
            j = self.total_error(combined.expectations, activations[-1])

            if abs(j - prev_j) < mindelta: yield j; break
            prev_j = j

            if counter == 0: yield j
            counter = (counter + 1) % skip


    def propagate(self, features):
        """
        Calcula as ativações da rede para um conjunto de instâncias.
          - features: matriz de atributos de instância (instâncias nas linhas)
        Retorna uma lista de matrizes com as ativações de cada camada:
          * uma matriz por camada da rede, incluindo as entradas
          * cada linha tem as ativações da camada para uma instância
          * todas as matrizes (menos a última) incluem uma coluna de bias
        """

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
        """
        Calcula os gradientes finais para um conjunto de instâncias.
          - expectations: matriz de saídas esperadas (instâncias nas linhas)
          - activations: lista de matrizes das ativações dos neurônios
            (mesmo formato da saída do método `propagate`)
        Retorna uma lista de matrizes com os gradientes de cada camada,
        com o mesmo formato que a lista de matrizes dos pesos da rede.
        """

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
        """
        Calcula os gradientes por backpropagation.
        Retorna uma lista de matrizes com os gradientes de cada camada,
        com o mesmo formato que a lista de matrizes dos pesos da rede.
        """

        activations = self.propagate(dataset.features)
        gradients = self._backpropagate(dataset.expectations, activations)
        return gradients


    def numeric_gradients(self, dataset, epsilon=1e-6):
        """
        Calcula estimativas numéricas dos gradientes.
        Retorna uma lista de matrizes com os gradientes de cada camada,
        com o mesmo formato que a lista de matrizes dos pesos da rede.
        """

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


    def evaluate(self, dataset):
        """
        Avalia o desempenho da rede neural com instâncias de teste.
          - dataset: conjunto de instâncias de teste
        Retorna o erro total J e a média dos F1-scores.
        """

        predictions = self.propagate(dataset.features) [-1]
        j = self.total_error(dataset.expectations, predictions)
        scores = f1_scores(dataset.expectations, predictions)
        return j, np.mean(scores)


    def total_error(self, expectations, predictions):
        """
        Calcula o erro total (J), com regularização.
          - expectations: matriz de saídas esperadas (instâncias nas linhas)
          - predictions: matriz de saídas preditas (ativação da última camada)
        """

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


def f1_scores(expectations, predictions):
    """
    Calcula as F1-measures de cada classe.
      - expectations: matriz de saídas esperadas (instâncias nas linhas)
      - predictions: matriz de saídas preditas (ativação da última camada)
    Retorna um np.array de valores, um para cada classe.
    """

    assert expectations.shape == predictions.shape
    y = expectations.astype(int)
    f = classify(predictions).astype(int)

    with np.errstate(invalid='ignore'):  # ignora divisões 0/0
        true_positives = np.sum(y & f, axis=0).astype(float)
        precisions = true_positives / np.sum(f, axis=0)
        recalls = true_positives / np.sum(y, axis=0)
        scores = 2.0 * precisions * recalls / (precisions + recalls)

    scores[np.isnan(scores)] = 0.0  # corrige divisões 0/0
    return scores


def classify(predictions):
    """
    Converte as ativações de saída da rede neural em classes 0.0 ou 1.0
      - predictions: matriz de saídas preditas (ativação da última camada)
    """

    if predictions.shape[1] == 1:
        # uma coluna: classificação binária, arredonda para 0.0 ou 1.0
        f = np.round_(predictions)
    else:
        # múltiplas colunas: problema multiclasse
        # acha o índice da coluna com a maior predição de cada linha
        classes = np.argmax(predictions, axis=1)
        # em cada linha, põe 1.0 na coluna com a maior predição e 0.0 nas demais
        f = np.zeros_like(predictions)
        f[np.arange(f.shape[0]), classes] = 1.0

    return f

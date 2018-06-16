import math
import numpy as np


class Dataset:

    def __init__(self, instances, normalize=False):
        # Inicializa um dataset.
        # - instances: lista de instâncias de treinamento
        # (cada instância é um par de duas listas: atributos e saídas)

        self.features = np.array([i[0] for i in instances])
        self.expectations = np.array([i[1] for i in instances])

        if normalize:
            maximums = np.max(self.features, axis=0)
            minimums = np.min(self.features, axis=0)
            self.features = (self.features - minimums) / (maximums - minimums)


    def minibatches(self, num_batches):
        # Divide o dataset em mini-batches.
        # - num_batches: número desejado de mini-batches
        # Retorna uma lista de sub-datasets, um para cada mini-batch.

        batch_size = math.ceil(len(self.features) / float(num_batches))
        batches = []

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            batch = Dataset([])

            batch.features = self.features[start:end]
            batch.expectations = self.expectations[start:end]
            batches.append(batch)

        return batches

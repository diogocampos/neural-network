import numpy as np


class Dataset:

    def __init__(self, instances, normalize=False):
        # Inicializa um dataset.
        # - instances: lista de instâncias de treinamento
        # (cada instância é uma tupla de duas listas: atributos e saídas)

        self.features = np.array([i[0] for i in instances])
        self.expectations = np.array([i[1] for i in instances])

        if normalize:
            maximums = np.max(self.features, axis=0)
            minimums = np.min(self.features, axis=0)
            self.features = (self.features - minimums) / (maximums - minimums)

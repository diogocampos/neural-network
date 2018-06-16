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


    def __len__(self):
        return len(self.features)


    def random_folds(self, num_folds):
        # Divide o dataset em folds aleatórios estratificados.
        # - num_folds: número desejado de folds
        # Retorna uma lista de sub-datasets, um para cada fold.

        # reordena o dataset aleatoriamente
        indexes = np.random.permutation(len(self.features))
        features = self.features[indexes]
        expectations = self.expectations[indexes]

        # separa as instâncias em grupos, por classe
        classes = np.unique(expectations, axis=0)
        groups = [np.where(np.all(expectations == c, axis=1))[0] for c in classes]

        # divide grupos em `num_folds` pedaços
        groups = [np.array_split(g, num_folds) for g in groups]

        # forma folds com um pedaço de cada grupo
        folds = []

        for i in range(num_folds):
            indexes = np.concatenate([g[i] for g in groups])
            fold = Dataset([])
            fold.features = features[indexes]
            fold.expectations = expectations[indexes]
            folds.append(fold)

        return folds

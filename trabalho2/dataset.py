import numpy as np


class Dataset:

    def __init__(self, instances):
        # Inicializa um dataset.
        # - instances: lista de instâncias de treinamento
        # (cada instância é uma tupla de duas listas: atributos e saídas)

        self.features = np.array([i[0] for i in instances])
        self.outputs = np.array([i[1] for i in instances])

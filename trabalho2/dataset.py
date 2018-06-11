import numpy as np


class Dataset:

    def __init__(self, instances):
        # Inicializa um dataset.
        # - instances: lista de instâncias de treinamento
        # (cada instância é uma tupla de duas listas: atributos e saídas)

        # as matrizes são armazenadas transpostas (instâncias nas colunas)
        self.features = np.array([i[0] for i in instances]).T
        self.outputs = np.array([i[1] for i in instances]).T

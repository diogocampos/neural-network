import numpy as np


class Dataset:
    def __init__(self, instances):
        self.features = np.array([i[0] for i in instances])
        self.outputs = np.array([i[1] for i in instances])

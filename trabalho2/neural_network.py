import numpy as np


class NeuralNetwork:
    def __init__(self, regularization, structure):
        self.regularization = regularization
        self.structure = tuple(structure)

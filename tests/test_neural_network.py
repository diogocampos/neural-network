import numpy as np
import pytest

from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


def test_regularization(example):
    regularization, structure = example['network']
    network = NeuralNetwork(regularization, structure)
    assert network.regularization == regularization


def test_structure(example):
    regularization, structure = example['network']
    network = NeuralNetwork(regularization, structure)
    assert list(network.structure) == structure

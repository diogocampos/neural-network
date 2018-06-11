import numpy as np
import pytest

from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


def test_lambda_(example):
    lambda_, structure = example['network']
    network = NeuralNetwork(lambda_, structure)
    assert network.lambda_ == lambda_


def test_structure(example):
    lambda_, structure = example['network']
    network = NeuralNetwork(lambda_, structure)
    assert list(network.structure) == structure


def test_set_weights(example):
    network = NeuralNetwork(*example['network'])
    network.set_weights(example['weights'])

    weights = [theta.tolist() for theta in network.weights]
    assert weights == example['weights']


def test_propagate(example):
    network = NeuralNetwork(*example['network'])
    network.set_weights(example['weights'])

    dataset = Dataset(example['dataset'])
    activations = network.propagate(dataset.features)

    activations = np.round_(activations, decimals=5)
    assert activations.tolist() == example['activations']

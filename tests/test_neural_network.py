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
    network = init_network(example)
    weights = network.weights

    weights = [theta.tolist() for theta in weights]
    assert weights == example['weights']


def test_propagate(example):
    network, dataset = init_network_and_dataset(example)
    activations = network.propagate(dataset.features)

    activations = round_matrixes(activations, decimals=5, transpose=True)
    assert activations == example['activations']


def test_gradients(example):
    network, dataset = init_network_and_dataset(example)
    gradients = network.gradients(dataset)

    gradients = round_matrixes(gradients, decimals=5)
    assert gradients == example['gradients']


def test_numeric_gradients(example):
    network, dataset = init_network_and_dataset(example)
    numeric_gradients = network.numeric_gradients(dataset, epsilon=1e-6)

    numeric_gradients = round_matrixes(numeric_gradients, decimals=5)
    assert numeric_gradients == example['gradients']


def test_total_error(example):
    network, dataset = init_network_and_dataset(example)
    activations = network.propagate(dataset.features)
    total_error = network.total_error(dataset.expectations, activations[-1])

    assert round(total_error, 5) == example['total_error']


## Funções auxiliares

def init_network(example):
    # Inicializa uma rede neural com os valores do exemplo dado.

    network = NeuralNetwork(*example['network'])
    network.set_weights(example['weights'])
    return network


def init_network_and_dataset(example):
    # Inicializa uma rede neural e um dataset com os valores do exemplo dado.

    network = init_network(example)
    dataset = Dataset(example['dataset'])
    return network, dataset


def round_matrixes(matrixes, decimals=5, transpose=False):
    # Arredonda os valores dos elementos de uma lista de arrays NumPy.

    matrixes = [np.round_(m, decimals=decimals) for m in matrixes]
    if transpose: matrixes = [m.T for m in matrixes]
    matrixes = [m.tolist() for m in matrixes]
    return matrixes

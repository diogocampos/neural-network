import numpy as np
import pytest

from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork, f1_scores, classify


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


def test_set_random_weights(example):
    network = init_network(example)
    network.set_random_weights()

    shapes = [t.shape for t in network.weights]
    expected_shapes = [(len(t), len(t[0])) for t in example['weights']]
    assert shapes == expected_shapes


def test_propagate(example):
    network, dataset = init_network_and_dataset(example)
    activations = network.propagate(dataset.features)

    activations = round_matrixes(activations, decimals=5)
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


def test_f1_scores_binary():
    expectations = [ [0.], [1.], [0.], [1.], [0.] ]
    predictions = [ [.9], [.9], [.1], [.9], [.1] ]
    expected_scores = [0.8]

    scores = f1_scores(np.array(expectations), np.array(predictions))
    assert scores.tolist() == expected_scores


def test_f1_scores_multiclass():
    expectations = [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]
    predictions = [
        [.9, .1, .1],
        [.9, .1, .1],
        [.1, .1, .9],
    ]
    expected_scores = [1/1.5, 0., 1.]

    scores = f1_scores(np.array(expectations), np.array(predictions))
    assert scores.tolist() == expected_scores


def test_classify():
    predictions = [
        [.1, .3, .2],
        [.3, .2, .1],
        [.2, .1, .3],
    ]
    expected_classes = [
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
    ]

    classes = classify(np.array(predictions))
    assert classes.tolist() == expected_classes


## Funções auxiliares

def init_network(example):
    """
    Inicializa uma rede neural com os valores do exemplo dado.
    """
    network = NeuralNetwork(*example['network'])
    network.set_weights(example['weights'])
    return network


def init_network_and_dataset(example):
    """
    Inicializa uma rede neural e um dataset com os valores do exemplo dado.
    """
    network = init_network(example)
    dataset = Dataset(example['dataset'])
    return network, dataset


def round_matrixes(matrixes, decimals=5):
    """
    Dada uma lista de arrays NumPy, retorna os arrays convertidos para listas,
    com seus valores arrendondados para o número desejado de casas decimais.
    """
    matrixes = [np.round_(m, decimals=decimals) for m in matrixes]
    matrixes = [m.tolist() for m in matrixes]
    return matrixes

#!/usr/bin/env python3

import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


def backpropagation(network, dataset):
    return network.gradients(dataset)

def numeric_verification(network, dataset):
    return network.numeric_gradients(dataset)


def main(argv, calculate_gradients=backpropagation):
    try:
        network_file, weights_file, dataset_file = argv[1:]
    except ValueError:
        usage = 'Usage:  %s NETWORK_FILE WEIGHTS_FILE DATASET_FILE'
        print(usage % argv[0], file=sys.stderr)
        return 1

    network = NeuralNetwork(*parsing.parse_network_file(network_file))
    network.set_weights(parsing.parse_weights_file(weights_file))
    dataset = Dataset(parsing.parse_dataset_file(dataset_file))

    # Calcula os gradientes usando a função fornecida
    gradients = calculate_gradients(network, dataset)

    # Imprime as matrizes de gradientes, uma matriz (camada) por linha:
    # - linhas separadas por ponto-e-vírgula
    # - elementos separados por vírgula, com 5 casas decimais
    for matrix in gradients:
        rows = [', '.join('%.5f' % val for val in row) for row in matrix]
        print('; '.join(rows))


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)

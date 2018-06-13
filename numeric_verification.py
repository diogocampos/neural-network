#!/usr/bin/env python3

import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


USAGE = 'Uso:  $ %s NETWORK_FILE WEIGHTS_FILE DATASET_FILE'


def main(argv):
    try:
        network_file, weights_file, dataset_file = argv[1:]
    except ValueError:
        print(USAGE % argv[0], file=sys.stderr)
        return 1

    network = NeuralNetwork(*parsing.parse_network_file(network_file))
    network.set_weights(parsing.parse_weights_file(weights_file))
    dataset = Dataset(parsing.parse_dataset_file(dataset_file))

    # Calcula os gradientes por estimativa numérica
    gradients = network.numeric_gradients(dataset)

    # Imprime as matrizes de gradientes, uma matriz (camada) por linha:
    # - linhas separadas por ponto-e-vírgula
    # - elementos separados por vírgula, com 5 casas decimais
    for matrix in gradients:
        rows = [', '.join('%.5f' % val for val in row) for row in matrix]
        print('; '.join(rows))


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)

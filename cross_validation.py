#!/usr/bin/env python3

import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


LAMBDA = 0.001
HIDDEN_LAYERS = [5, 5]

TRAINING_PARAMS = {
    'alpha': 1.0,
    'beta': 0.8,
    'mindelta': 1e-9,
}

NUM_FOLDS = 10


def main(argv):
    try:
        dataset_file = argv[1]
    except IndexError:
        print('Usage:  %s DATASET_FILE' % argv[0], file=sys.stderr)
        return 1

    instances = parsing.parse_dataset_file(dataset_file)
    dataset = Dataset(instances, normalize=True)

    structure = [dataset.num_inputs()] + HIDDEN_LAYERS + [dataset.num_outputs()]
    network = NeuralNetwork(LAMBDA, structure)

    cross_validation(network, dataset, NUM_FOLDS, **TRAINING_PARAMS)


def cross_validation(network, dataset, num_folds, **training_params):
    print('lambda = %r, structure = %r' % (network.lambda_, network.structure))
    print(', '.join('%s = %r' % (k, v) for k, v in training_params.items()))

    folds = dataset.random_folds(num_folds)

    for i in range(len(folds)):
        test_set = folds[i]
        training_sets = folds[:i] + folds[i+1:]
        print('Fold %d/%d:' % (i + 1, len(folds)))

        try:
            for j_t in network.train(training_sets, **training_params):
                print('    J_t =', j_t, end='\r')
        except KeyboardInterrupt:
            pass

        j_cv, f1_score = network.evaluate(test_set)
        print('\n    J_cv = %r, F1_score = %r' % (j_cv, f1_score))


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)

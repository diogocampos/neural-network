#!/usr/bin/env python3

import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


LAMBDA = 0.001
HIDDEN_LAYERS = [5, 5]

TRAINING_PARAMS = {
    'alpha': 0.01,
    'momentum': 0.99,
    'mindelta': 1e-9,
}

NUM_FOLDS = 10


def main(argv):
    try:
        dataset_file = argv[1]
    except IndexError:
        print('Uso:  $ %s DATASET_FILE' % argv[0], file=sys.stderr)
        return 1

    instances = parsing.parse_dataset_file(dataset_file)
    dataset = Dataset(instances, normalize=True)
    folds = dataset.random_folds(NUM_FOLDS)

    structure = [dataset.num_inputs()] + HIDDEN_LAYERS + [dataset.num_outputs()]
    network = NeuralNetwork(LAMBDA, structure)
    print('lambda = %r, structure = %r' % (network.lambda_, network.structure))
    print(', '.join('%s = %r' % (k, v) for k, v in TRAINING_PARAMS.items()))

    for i in range(len(folds)):
        test_set = folds[i]
        training_sets = folds[:i] + folds[i+1:]
        print('Fold %d/%d:' % (i + 1, len(folds)))

        for jt in network.train(training_sets, **TRAINING_PARAMS):
            print('  J_t =', jt, end='\r')

        jcv = network.evaluate(test_set)
        print('\n  J_cv =', jcv)


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)

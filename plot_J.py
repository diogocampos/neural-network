#!/usr/bin/env python3

import json
import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


SKIP = 10
MAX_SAMPLES = 1000
NUM_BATCHES = 10


def main(argv):
    if len(argv) < 3:
        usage = 'Usage:  %s RESULT_FILE.json DATASET_FILE'
        print(usage % argv[0], file=sys.stderr)
        return 1

    result_file = argv[1]
    dataset_file = argv[2]

    with open(result_file, 'r') as file:
        result = json.loads(file.read())

    network = NeuralNetwork(result['lambda'], result['structure'])
    dataset = Dataset(parsing.parse_dataset_file(dataset_file), normalize=True)
    batches = dataset.random_folds(NUM_BATCHES)

    counter = 1
    print('Iterations,J_t')

    for j_t in network.train(batches, **result['training'], skip=SKIP):
        print('%s,%s' % (counter * SKIP, j_t))
        counter += 1
        if counter >= MAX_SAMPLES: break
    else:
        print('%s,%s' % (counter * SKIP, j_t))


if __name__ == '__main__':
    sys.exit(main(sys.argv))

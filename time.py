#!/usr/bin/env python3

import sys
import timeit

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork

path = 'tests/fixtures/exemplo2/%s.txt'

network = NeuralNetwork(*parsing.parse_network_file(path % 'network'))
network.set_weights(parsing.parse_weights_file(path % 'initial_weights'))
dataset = Dataset(parsing.parse_dataset_file(path % 'dataset'))

results = timeit.repeat(
    'network.gradients(dataset)',
    number=10000,
    globals=globals())

for r in results:
    print(r)

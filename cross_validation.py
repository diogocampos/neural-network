#!/usr/bin/env python3

import json
from pathlib import Path
import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


# Variações de parâmetros da rede
LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
NUMS_HIDDEN_LAYERS = [1, 2, 3]
NUMS_NEURONS_PER_LAYER = [2, 4, 6, 8, 10]

# Parâmetros de treinamento
PARAMS = {
    'alpha': 1.0,
    'beta': 0.7,
    'mindelta': 1e-9,
}

# Número de folds da validação cruzada
NUM_FOLDS = 10

# Diretório onde serão salvos os resultados
RESULTS_DIR = 'results'


def main(argv):
    try:
        dataset_file = argv[1]
    except IndexError:
        print('Usage:  %s DATASET_FILE' % argv[0], file=sys.stderr)
        return 1

    instances = parsing.parse_dataset_file(dataset_file)
    dataset = Dataset(instances, normalize=True)

    n_inputs = dataset.num_inputs()
    n_outputs = dataset.num_outputs()
    structures = ([n_inputs] + hidden_layers + [n_outputs] for hidden_layers in
        hidden_layer_variations(NUMS_HIDDEN_LAYERS, NUMS_NEURONS_PER_LAYER))

    for network in network_variations(LAMBDAS, structures):
        results = cross_validation(network, dataset, NUM_FOLDS, **PARAMS)
        save_results(RESULTS_DIR, results)


def hidden_layer_variations(nums_layers, nums_neurons_per_layer):
    for n_layers in nums_layers:
        for n_neurons in nums_neurons_per_layer:
            yield n_layers * [n_neurons]


def network_variations(lambdas, structures):
    for structure in structures:
        for lambda_ in lambdas:
            yield NeuralNetwork(lambda_, structure)


def cross_validation(network, dataset, num_folds, **training_params):
    print()
    print('lambda = %r, structure = %r' % (network.lambda_, network.structure))
    print(', '.join('%s = %r' % (k, v) for k, v in training_params.items()))

    folds = dataset.random_folds(num_folds)
    fold_results = []

    for i in range(len(folds)):
        test_set = folds[i]
        training_sets = folds[:i] + folds[i+1:]
        print('Fold %d/%d:' % (i + 1, len(folds)))

        for j_t in network.train(training_sets, **training_params):
            print('    J_t =', j_t, end='\r')

        j_cv, f1_score = network.evaluate(test_set)
        print('\n    J_cv = %r, F1_score = %r' % (j_cv, f1_score))

        fold_results.append({
            'j_t': j_t,
            'j_cv': j_cv,
            'f1_score': f1_score,
        })

    return {
        'lambda': network.lambda_,
        'structure': network.structure,
        'training': training_params,
        'folds': fold_results,
    }


def save_results(dirname, results):
    structure = '-'.join(str(size) for size in results['structure'])
    filename = '%s-%r.json' % (structure, results['lambda'])
    path = Path(__file__).parent / dirname / filename

    with open(path, 'w') as file:
        string = json.dumps(results, indent=2)
        file.write(string)

    print('Results saved to', path)


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)

#!/usr/bin/env python3

import json
from pathlib import Path
import sys

from trabalho2 import parsing
from trabalho2.dataset import Dataset
from trabalho2.neural_network import NeuralNetwork


# Variações de parâmetros da rede neural
HIDDEN_LAYERS = [[5, n] for n in range(1, 6)]
LAMBDAS = [0.01, 0.1, 1.0]

# Parâmetros de treinamento
PARAMS = {
    'alpha': 1.0,
    'beta': 0.5,
    'mindelta': 1e-7,
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

    structures = ([dataset.num_inputs()] + hidden + [dataset.num_outputs()]
        for hidden in HIDDEN_LAYERS)

    for structure in structures:
        for lambda_ in LAMBDAS:
            network = NeuralNetwork(lambda_, structure)
            results = cross_validation(network, dataset, NUM_FOLDS, **PARAMS)
            save_results(RESULTS_DIR, results)


def cross_validation(network, dataset, num_folds, **training_params):
    print()
    print('structure = %r, lambda = %r' % (network.structure, network.lambda_))
    print(', '.join('%s = %r' % (k, v) for k, v in training_params.items()))

    folds = dataset.random_folds(num_folds)
    fold_results = []

    for i in range(len(folds)):
        test_set = folds[i]
        training_sets = folds[:i] + folds[i+1:]
        print('Fold %d/%d:' % (i + 1, len(folds)))

        for j_t in network.train(training_sets, **training_params):
            print('   J_t  =', j_t, end='\r')

        j_cv, f1_score = network.evaluate(test_set)
        print('\n   J_cv = %r,\tF1_score = %r' % (j_cv, f1_score))

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
    # monta o nome do arquivo: es,tru,tu,ra-lambda.json
    structure = ','.join(str(size) for size in results['structure'])
    filename = '%s-%r.json' % (structure, results['lambda'])
    path = Path(__file__).parent / dirname / filename

    with open(path, 'w') as file:
        string = json.dumps(results, indent=2)
        file.write(string)

    print('Results saved to', path)


if __name__ == '__main__':
    result = main(sys.argv)
    sys.exit(result)

#!/usr/bin/env python3

import numpy as np

import json
from pathlib import Path
import sys


def main(argv):

    if len(argv) < 2:
        print('Usage: %s RESULTS_DIR [--csv]' % argv[0], file=sys.stderr)
        return 1

    dirname = argv[1]
    use_csv = (len(argv) > 2 and argv[2] == '--csv')

    results = load_results(dirname)
    average_jcv = lambda result: np.mean(get_values(result, 'j_cv'))
    sorted_results = sorted(results, key=average_jcv)

    print_results(sorted_results, use_csv)


def load_results(dirname):

    files = Path(dirname).glob('*.json')
    results = []

    for filename in files:
        with open(filename, 'r') as file:
            data = json.loads(file.read())
            results.append(data)

    return results


def print_results(results, use_csv=False):

    if use_csv:
        row_format = '"%s",%s,%s,%s,%s,%s,%s,%s'
    else:
        row_format = '%-12s %-10s %-22s %-22s %-22s %-22s %-22s %-22s'

    table = format_results(results)
    for row in table:
        print(row_format % tuple(row))


def format_results(results):

    table = [[
        'Structure',
        'Lambda',
        'Avg J_t',
        'Stddev J_t',
        'Avg J_cv',
        'Stddev J_cv',
        'Avg F1_score',
        'Stddev F1_score'
    ]]

    for result in results:
        row = [
            ' '.join(str(val) for val in result['structure']),
            result['lambda'],
            np.mean(get_values(result, 'j_t')),
            np.std(get_values(result, 'j_t')),
            np.mean(get_values(result, 'j_cv')),
            np.std(get_values(result, 'j_cv')),
            np.mean(get_values(result, 'f1_score')),
            np.std(get_values(result, 'f1_score')),
        ]
        table.append(row)

    return table


def get_values(result, key):
    values = [fold[key] for fold in result['folds']]
    return np.array(values)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

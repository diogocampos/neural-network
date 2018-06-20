#!/usr/bin/env python3

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
    sorted_results = sorted(results, key=lambda r: get_average(r, 'j_cv'))

    print_results(sorted_results, use_csv)


def load_results(dirname):

    files = Path(dirname).glob('*.json')
    results = []

    for filename in files:
        with open(filename, 'r') as file:
            data = json.loads(file.read())
            results.append(data)

    return results


def get_average(result, key):
    # Calcula o valor médio de fold[key] para um arquivo de resultados.

    values = [fold[key] for fold in result['folds']]
    average = sum(values) / len(values)
    return average


def print_results(results, use_csv=False):

    if use_csv:
        row_format = '%s;%s;%s;%s'
    else:
        row_format = '%-20s %-20s %-20s %-20s'

    table = format_results(results)
    for row in table:
        print(row_format % tuple(row))


def format_results(results):

    table = [['Structure', 'Lambda', 'Average J_cv', 'Average F1_score']]

    for result in results:
        row = [
            result['structure'],
            result['lambda'],
            get_average(result, 'j_cv'),
            get_average(result, 'f1_score'),
        ]
        table.append(row)

    return table


if __name__ == '__main__':
    sys.exit(main(sys.argv))

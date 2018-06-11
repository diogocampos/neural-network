from pathlib import Path
import pytest


EXAMPLES = [
    {
        'name': 'exemplo1',
        'network': (0.00000, [1, 2, 1]),
        'weights': [
            [
                [0.40000, 0.10000],
                [0.30000, 0.20000],
            ],
            [
                [0.70000, 0.50000, 0.60000],
            ],
        ],
        'dataset': [
            ([0.13000], [0.90000]),
            ([0.42000], [0.23000]),
        ],
        'activations': [
            [
                [1.00000, 0.13000],
                [1.00000, 0.42000],
            ],
            [
                [1.00000, 0.60181, 0.58079],
                [1.00000, 0.60874, 0.59484],
            ],
            [
                [0.79403],
                [0.79597],
            ],
        ],
        'total_error': 0.82098,
    },
    {
        'name': 'exemplo2',
        'network': (0.25000, [2, 4, 3, 2]),
        'weights': [
            [
                [0.42000, 0.15000, 0.40000],
                [0.72000, 0.10000, 0.54000],
                [0.01000, 0.19000, 0.42000],
                [0.30000, 0.35000, 0.68000],
            ],
            [
                [0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
                [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
                [0.03000, 0.56000, 0.80000, 0.69000, 0.09000],
            ],
            [
                [0.04000, 0.87000, 0.42000, 0.53000],
                [0.17000, 0.10000, 0.95000, 0.69000],
            ],
        ],
        'dataset': [
            ([0.32000, 0.68000], [0.75000, 0.98000]),
            ([0.83000, 0.02000], [0.75000, 0.28000]),
        ],
        'activations': [
            [
                [1.00000, 0.32000, 0.68000],
                [1.00000, 0.83000, 0.02000],
            ],
            [
                [1.00000, 0.67700, 0.75384, 0.58817, 0.70566],
                [1.00000, 0.63472, 0.69292, 0.54391, 0.64659],
            ],
            [
                [1.00000, 0.87519, 0.89296, 0.81480],
                [1.00000, 0.86020, 0.88336, 0.79791],
            ],
            [
                [0.83318, 0.84132],
                [0.82953, 0.83832],
            ],
        ],
        'total_error': 1.90351,
    },
]

@pytest.fixture(params=EXAMPLES)
def example(request):
    example = request.param
    dir = Path('tests/fixtures', example['name'])
    example['files'] = {
        'network': dir / 'network.txt',
        'weights': dir / 'initial_weights.txt',
        'dataset': dir / 'dataset.txt',
    }
    return example

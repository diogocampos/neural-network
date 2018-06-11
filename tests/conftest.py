from pathlib import Path
import pytest


EXAMPLES = [
    {
        'name': 'exemplo1',
        'network': (0.0, [1, 2, 1]),
        'weights': [
            [[0.4, 0.1], [0.3, 0.2]],
            [[0.7, 0.5, 0.6]],
        ],
        'dataset': [
            ([0.13], [0.9]),
            ([0.42], [0.23]),
        ],
        'activations': [
            [[1.00000, 0.13000], [1.00000, 0.42000]],
            [[1.00000, 0.60181, 0.58079], [1.00000, 0.60874, 0.59484]],
            [[0.79403], [0.79597]],
        ],
        'total_error': 0.82098,
    },
    {
        'name': 'exemplo2',
        'network': (0.25, [2, 4, 3, 2]),
        'weights': [
            [[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]],
            [[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]],
            [[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]],
        ],
        'dataset': [
            ([0.32, 0.68], [0.75, 0.98]),
            ([0.83, 0.02], [0.75, 0.28]),
        ],
        'activations': [
            [[1.00000, 0.32000, 0.68000], [1.00000, 0.83000, 0.02000]],
            [[1.00000, 0.67700, 0.75384, 0.58817, 0.70566], [1.00000, 0.63472, 0.69292, 0.54391, 0.64659]],
            [[1.00000, 0.87519, 0.89296, 0.81480], [1.00000, 0.86020, 0.88336, 0.79791]],
            [[0.83318, 0.84132], [0.82953, 0.83832]],
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

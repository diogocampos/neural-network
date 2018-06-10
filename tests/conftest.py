from pathlib import Path
import pytest

EXAMPLES = [
    {
        'name': 'exemplo1',
        'network': (0.0, [1, 2, 1]),
    },
    {
        'name': 'exemplo2',
        'network': (0.25, [2, 4, 3, 2]),
    },
]

@pytest.fixture(params=EXAMPLES)
def example(request):
    example = request.param
    dir = Path('tests/fixtures', example['name'])
    example['files'] = {
        'network': dir / 'network.txt',
    }
    return example

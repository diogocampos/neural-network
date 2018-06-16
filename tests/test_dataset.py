import pytest

from trabalho2.dataset import Dataset


def test_features(example):
    instances = example['dataset']
    dataset = Dataset(instances)
    assert dataset.features.tolist() == [i[0] for i in instances]


def test_expectations(example):
    instances = example['dataset']
    dataset = Dataset(instances)
    assert dataset.expectations.tolist() == [i[1] for i in instances]


def test_normalize():
    features = [
        [0.0, -1.0],
        [1.0,  1.0],
        [4.0,  0.0],
    ]
    normalized_features = [
        [0.00, 0.0],
        [0.25, 1.0],
        [1.00, 0.5],
    ]
    instances = [(xs, [0.0]) for xs in features]
    dataset = Dataset(instances, normalize=True)
    assert dataset.features.tolist() == normalized_features


@pytest.mark.parametrize('num_instances, num_batches, batch_sizes', [
    (10, 2, [5, 5]),
    (8, 3, [3, 3, 2]),
])
def test_minibatches(num_instances, num_batches, batch_sizes):
    dataset = Dataset(num_instances * [([0], [0])])
    batches = dataset.minibatches(num_batches)
    assert [len(batch.features) for batch in batches] == batch_sizes

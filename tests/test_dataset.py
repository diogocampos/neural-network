from trabalho2.dataset import Dataset


def test_features(example):
    instances = example['dataset']
    dataset = Dataset(instances)
    assert dataset.features.tolist() == [i[0] for i in instances]


def test_expectations(example):
    instances = example['dataset']
    dataset = Dataset(instances)
    assert dataset.expectations.tolist() == [i[1] for i in instances]

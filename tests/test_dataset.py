from trabalho2.dataset import Dataset


def test_features(example):
    instances = example['dataset']
    dataset = Dataset(instances)
    assert dataset.features.tolist() == [i[0] for i in instances]


def test_outputs(example):
    instances = example['dataset']
    dataset = Dataset(instances)
    assert dataset.outputs.tolist() == [i[1] for i in instances]

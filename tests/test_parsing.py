from trabalho2 import parsing

def test_parse_network_file(example):
    filename = example['files']['network']
    network = parsing.parse_network_file(filename)
    assert network == example['network']

def test_parse_weights_file(example):
    filename = example['files']['weights']
    weights = parsing.parse_weights_file(filename)
    assert weights == example['weights']

def test_parse_dataset_file(example):
    filename = example['files']['dataset']
    dataset = parsing.parse_dataset_file(filename)
    assert dataset == example['dataset']

from trabalho2 import parsing

def test_parse_network_file(example):
    filename = example['files']['network']
    network = parsing.parse_network_file(filename)
    assert network == example['network']

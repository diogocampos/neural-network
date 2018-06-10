
def parse_network_file(filename):
    with open(filename, 'r') as file:
        regularization = file.readline()
        structure = file.readlines()

    regularization = float(regularization)
    structure = [int(layer_size) for layer_size in structure]
    return regularization, structure


def parse_weights_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    layers = []
    for line in lines:
        neurons = [neuron.split(',') for neuron in line.split(';')]
        layer = [[float(weight) for weight in neuron] for neuron in neurons]
        layers.append(layer)
    return layers


def parse_network_file(filename):
    with open(filename, 'r') as file:
        regularization = file.readline()
        structure = file.readlines()

    regularization = float(regularization)
    structure = [int(layer_size) for layer_size in structure]
    return regularization, structure

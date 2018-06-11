
def parse_network_file(filename):
    # Faz a leitura de arquivos do tipo network.txt.
    # Retorna o fator de regularização e a lista de tamanhos das camadas.

    with open(filename, 'r') as file:
        lambda_ = file.readline()
        structure = file.readlines()

    lambda_ = float(lambda_)
    structure = [int(layer_size) for layer_size in structure]
    return lambda_, structure


def parse_weights_file(filename):
    # Faz a leitura de arquivos do tipo initial_weights.txt.
    # Retorna uma lista de matrizes theta.
    # (cada linha de uma matriz tem os pesos de um neurônio, incluindo o bias)

    with open(filename, 'r') as file:
        lines = file.readlines()

    thetas = []
    for line in lines:
        neurons = [neuron.split(',') for neuron in line.split(';')]
        theta = [[float(weight) for weight in neuron] for neuron in neurons]
        thetas.append(theta)
    return thetas


def parse_dataset_file(filename):
    # Faz a leitura de arquivos do tipo dataset.txt.
    # Retorna uma lista de instâncias de treinamento.
    # (cada instância é uma lista de duas sub-listas: atributos e saídas)

    with open(filename, 'r') as file:
        lines = file.readlines()

    instances = []
    for line in lines:
        features, outputs = [values.split(',') for values in line.split(';')]
        features = [float(value) for value in features]
        outputs = [float(value) for value in outputs]
        instances.append([features, outputs])
    return instances

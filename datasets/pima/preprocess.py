from pathlib import Path

dir = Path(__file__).parent

with open(dir / 'pima.tsv', 'r') as input:
    input.readline()  # descarta a linha de cabeçalho
    lines = input.readlines()

rows = [line.strip().split('\t') for line in lines]
instances = [(values[:-1], values[-1]) for values in rows]

with open(dir / 'dataset.txt', 'w') as output:
    for xs, y in instances:
        line = ','.join(xs) + ';' + y + '\n'
        output.write(line)

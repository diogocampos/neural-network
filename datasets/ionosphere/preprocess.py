from pathlib import Path

dir = Path(__file__).parent

with open(dir / 'ionosphere.data', 'r') as input:
    lines = input.readlines()

rows = [line.strip().split(',') for line in lines]
instances = [([values[0]] + values[2:-1], values[-1]) for values in rows]

with open(dir / 'dataset.txt', 'w') as output:
    for xs, y in instances:
        assert y in ['b', 'g']
        class_ = '1' if y == 'g' else '0'
        line = ','.join(xs) + ';' + class_ + '\n'
        output.write(line)

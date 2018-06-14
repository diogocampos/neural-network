from pathlib import Path

dir = Path(__file__).parent

with open(dir / 'wdbc.data', 'r') as input:
    lines = input.readlines()

rows = [line.strip().split(',') for line in lines]
instances = [(values[2:], values[1]) for values in rows]

with open(dir / 'dataset.txt', 'w') as output:
    for xs, y in instances:
        assert y in ['B', 'M']
        class_ = '1' if y == 'M' else '0'
        line = ','.join(xs) + ';' + class_ + '\n'
        output.write(line)

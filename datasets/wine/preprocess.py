from pathlib import Path

dir = Path(__file__).parent

with open(dir / 'wine.data', 'r') as input:
    lines = input.readlines()

rows = [line.strip().split(',') for line in lines]
instances = [(values[1:], values[0]) for values in rows]

one_hot = {
    '1': '1,0,0',
    '2': '0,1,0',
    '3': '0,0,1',
}

with open(dir / 'dataset.txt', 'w') as output:
    for xs, y in instances:
        assert y in ['1', '2', '3']
        line = ','.join(xs) + ';' + one_hot[y] + '\n'
        output.write(line)

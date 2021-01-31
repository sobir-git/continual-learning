import argparse
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./logs')
args = parser.parse_args()
folder = args.logdir


def _parse_ids(line: str):
    return eval(line)


def read_ids(file):
    ids = set()
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        ids.update(_parse_ids(line))
    return ids


# collect ids
ids_in_file = OrderedDict()
for filename in sorted(os.listdir(folder)):
    path = folder + '/' + filename
    if os.path.isfile(path) and filename.endswith('.txt'):
        try:
            ids = read_ids(path)
        except SyntaxError:
            print(f'Error parsing {path}')
        else:
            ids_in_file[filename] = ids

# print sizes
for name, ids in ids_in_file.items():
    print(f'{name[:-8]}: {len(ids)}')

# check for unexpected overlaps
print('Intersections:')
keys = list(ids_in_file.keys())
for i in range(len(keys) - 1):
    for j in range(i + 1, len(keys)):
        k1 = keys[i]
        k2 = keys[j]
        print(f'{k1[:-8]} - {k2[:-8]}: {len(ids_in_file[k1].intersection(ids_in_file[k2]))}')

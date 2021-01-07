import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./logs')
args = parser.parse_args()
folder = args.logdir

fCLF_TRAIN_IDS = 'clf_train_ids.txt'
fCLF_VAL_IDS = 'clf_val_ids.txt'
fCTRL_TRAIN_IDS = 'ctrl_train_ids.txt'
fCTRL_VAL_IDS = 'ctrl_val_ids.txt'
fTEST_IDS = 'test_ids.txt'


def _parse_ids(line: str):
    return eval(line)


def read_ids(filename):
    ids = set()

    with open(folder + '/' + filename) as f:
        lines = f.readlines()
    for line in lines:
        ids.update(_parse_ids(line))
    return ids


# gather ids
clf_train_ids = read_ids(fCLF_TRAIN_IDS)
clf_val_ids = read_ids(fCLF_VAL_IDS)
ctrl_val_ids = read_ids(fCTRL_VAL_IDS)
ctrl_train_ids = read_ids(fCTRL_TRAIN_IDS)
test_ids = read_ids(fTEST_IDS)

mapping = dict(zip(['clf_train', 'clf_val', 'ctrl_train', 'ctrl_val', 'test'],
                   [clf_train_ids, clf_val_ids, ctrl_train_ids, ctrl_val_ids, test_ids]))

# print sizes
for name, ids in mapping.items():
    print(f'{name}: {len(ids)}')

# check for unexpected overlaps
print('Intersections:')
keys = list(mapping.keys())
for i in range(len(keys) - 1):
    for j in range(i + 1, len(keys)):
        k1 = keys[i]
        k2 = keys[j]
        print(f'{k1} - {k2}: {len(mapping[k1].intersection(mapping[k2]))}')

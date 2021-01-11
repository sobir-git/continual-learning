import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import wandb

import argparse

from exp2.data import prepare_data, create_loader
from exp2.feature_extractor import create_models
from exp2.memory import MemoryManager
from utils import get_default_device, get_console_logger

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str)
parser.add_argument('--source_project', type=str)
parser.add_argument('--config_artifact', type=str)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--phase', type=int)
args = parser.parse_args()

run = wandb.init(job_type='feeding_classifiers', project='classifier-outputs')
device = get_default_device()
console_logger = get_console_logger()


# load configs
def load_config_from_artifact(artifact):
    config_dir = Path(artifact.download())
    with open(config_dir / 'combined-config.yaml') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)


config = load_config_from_artifact(run.use_artifact(args.source_project + '/' + args.config_artifact))

# prepare data
data = prepare_data(config)

# load memory
memory_manager = MemoryManager(config, data)
artifact_name = f'{args.source_project}/memory-ids-{args.run_id}:v0'
memory_manager.load_from_artifact(run.use_artifact(artifact_name, type='ids'), args.phase)

# load classifiers and feature extractor
feature_extractor, classifier_constructor = create_models(config, device)
class_order = data.class_order
n_class_per_phase = config.n_classes_per_phase
classifiers = []
for i in range(args.phase):
    classes = class_order[i * n_class_per_phase:(i + 1) * n_class_per_phase]
    classifier = classifier_constructor(classes, idx=i)
    classifiers.append(classifier)

# load classifier checkpoints
for clf in classifiers:
    artifact_name = f'{args.source_project}/classifier-{clf.idx}-{args.run_id}:v0'
    artifact = run.use_artifact(artifact_name)
    artifact_dir = Path(artifact.download())
    clf.load_from_checkpoint(artifact_dir / f'classifier_{clf.idx}.pt')

# generate training data
trainset = memory_manager.ctrl_memory.get_dataset(train=True).mix(memory_manager.shared_memory.get_dataset(train=True))
trainset, valset = trainset.split(test_size=config.val_size)
testset = data.get_phase_data(phase=args.phase)[2]

# classifier outputs for training set
clf_train_outputs = torch.zeros(args.epochs, len(classifiers), len(trainset),
                                n_class_per_phase + config.other, dtype=torch.float32,
                                device=device)  # (epoch, clf, sample, outputs)
clf_val_outputs = torch.zeros(len(classifiers), len(valset),
                              n_class_per_phase + config.other, dtype=torch.float32,
                              device=device)  # (clf, sample, outputs)
clf_test_outputs = torch.zeros(len(classifiers), len(testset),
                               n_class_per_phase + config.other, dtype=torch.float32,
                               device=device)  # (clf, sample, outputs)

config.batch_size = args.batch_size
trainloader = create_loader(config, trainset)
valloader = create_loader(config, valset)
testloader = create_loader(config, testset)

console_logger.info('trainset size: %d', len(trainset))
console_logger.info('validation size: %d', len(valset))
console_logger.info('test size: %d', len(testset))

# gather classifier outputs for the trainset
console_logger.info('Gathering training set outputs')
for epoch in range(args.epochs):
    console_logger.info('Epoch %s / %s', epoch, args.epochs)
    i = 0
    for inputs, labels, ids in trainloader:
        inputs = inputs.to(device)
        features = feature_extractor(inputs)
        bsize = features.size(0)
        for j, clf in enumerate(classifiers):
            clf_train_outputs[epoch, j, i:i + bsize] = clf(features).cpu()
        i += bsize

# gather classifier outputs for the valset and testset
for tensor, loader in zip([clf_test_outputs, clf_val_outputs],
                          [testloader, valloader]):
    console_logger.info(f'Gathering {["testing", "validation"][loader is valloader]} set outputs')
    i = 0
    for inputs, labels, ids in loader:
        inputs = inputs.to(device)
        features = feature_extractor(inputs)
        bsize = features.size(0)
        for j, clf in enumerate(classifiers):
            tensor[j, i:i + bsize] = clf(features).cpu()
        i += bsize

# upload artifacts
console_logger.info("Uploading artifacts")
artifact = wandb.Artifact(f'classifier-outputs-{args.phase}-{wandb.run.id}', 'classifier_outputs')

for split in ['train', 'val', 'test']:
    tensorname = f'clf_{split}_outputs'
    tensor = globals()[tensorname]

    with artifact.new_file(tensorname + '.pkl', 'wb') as f:
        pickle.dump(tensor, f)

wandb.log_artifact(artifact)

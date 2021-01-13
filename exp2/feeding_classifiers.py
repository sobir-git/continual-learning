import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import numpy as np
import yaml
import wandb

import argparse

from wandb.apis.public import Run

from exp2.controller import Controller
from exp2.data import prepare_data, create_loader
from exp2.feature_extractor import create_models
from exp2.memory import MemoryManager
from utils import get_default_device, get_console_logger

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--phase', type=int)
args = parser.parse_args()


def aslist(s: Sequence):
    if type(s) is list:
        return s
    if isinstance(s, np.ndarray) or isinstance(s, torch.Tensor):
        return s.tolist()
    return list(s)


def get_all_run_artifacts(run: Run, mode='used'):
    if mode == 'used':
        artifacts = run.used_artifacts()
    else:
        artifacts = run.logged_artifacts()
    artifacts = [api.artifact(f'{run.entity}/{run.project}/{art.name}') for art in artifacts]
    artifacts_dict = {art.name: art for art in artifacts}
    return artifacts_dict


# load configs
def load_config_from_artifact(artifact):
    config_dir = Path(artifact.download())
    with open(config_dir / 'combined-config.yaml') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)


def load_config_from_run(run: Run):
    artifacts = get_all_run_artifacts(run, 'used')
    art = artifacts[f'config-{run.id}:v0']
    art_dir = Path(art.download())
    config_dict = yaml.safe_load((art_dir / 'combined-config.yaml').open())
    config = wandb.Config()
    config.update(config_dict)
    return config


device = get_default_device()
console_logger = get_console_logger()
run = wandb.init(job_type='feeding_classifiers', project='classifier-outputs')
api = wandb.Api()
past_run: Run = api.run(args.run)
config = load_config_from_run(past_run)

# prepare data
data = prepare_data(config)

# load memory
memory_manager = MemoryManager(config, data)
past_logged_artifacts = get_all_run_artifacts(past_run, 'logged')
artifact = past_logged_artifacts[f'memory-ids-{past_run.id}:v0']
memory_manager.load_from_artifact(artifact, args.phase)

# load classifiers and feature extractor
feature_extractor, classifier_constructor = create_models(config, device)
class_order = data.class_order
n_class_per_phase = config.n_classes_per_phase
classifiers = []
for i in range(args.phase):
    classes = class_order[i * n_class_per_phase:(i + 1) * n_class_per_phase]
    classifier = classifier_constructor(classes, idx=i)
    classifier.eval()
    classifiers.append(classifier)

# load classifier checkpoints
for clf in classifiers:
    artifact = past_logged_artifacts[f'classifier-{clf.idx}-{past_run.id}:v0']
    artifact_dir = Path(artifact.download())
    clf.load_from_checkpoint(artifact_dir / f'classifier_{clf.idx}.pt')

# generate training data
trainset = memory_manager.ctrl_memory.get_dataset(train=True).mix(memory_manager.shared_memory.get_dataset(train=True))
trainset, valset = trainset.split(test_size=config.val_size)
testset = data.get_phase_data(phase=args.phase)[2]
clf_output_size = n_class_per_phase + config.other

clf_outputs = {
    'train': np.zeros((args.epochs, len(trainset), len(classifiers), clf_output_size), dtype=float),
    'val': np.zeros((len(valset), len(classifiers), clf_output_size), dtype=float),
    'test': np.zeros((len(testset), len(classifiers), clf_output_size), dtype=float),
}

labels = {
    'train': {
        'class': np.empty((args.epochs, len(trainset)), dtype=int),
        'classifier': np.empty((args.epochs, len(trainset)), dtype=int)
    },
    'val': {
        'class': np.empty((len(valset)), dtype=int),
        'classifier': np.empty((len(valset)), dtype=int)
    },
    'test': {
        'class': np.empty((len(testset)), dtype=int),
        'classifier': np.empty((len(testset)), dtype=int)
    },
}

config.update({'batch_size': args.batch_size}, allow_val_change=True)
trainloader = create_loader(config, trainset, shuffle=False)
valloader = create_loader(config, valset, shuffle=False)
testloader = create_loader(config, testset, shuffle=False)

console_logger.info('trainset size: %d', len(trainset))
console_logger.info('validation size: %d', len(valset))
console_logger.info('test size: %d', len(testset))

# load a raw controller just to map class labels to classifiers
controller = Controller(config, None, classifiers)


@torch.no_grad()
def get_classifier_outputs(classifiers, loader, output, cls_labels, clf_labels):
    i = 0
    for inputs, labels, ids in loader:
        inputs = inputs.to(device)
        bsize = inputs.size(0)
        features = feature_extractor(inputs)
        for j, clf in enumerate(classifiers):
            output[i:i + bsize, j] = clf(features).detach().cpu()
        cls_labels[i:i + bsize] = labels
        clf_labels[i:i + bsize] = controller.group_labels(labels)
        i += bsize


# gather classifier outputs for the trainset
console_logger.info('Gathering training set outputs')
for epoch in range(args.epochs):
    console_logger.info('Epoch %s / %s', epoch, args.epochs)
    get_classifier_outputs(classifiers, trainloader, clf_outputs['train'][epoch], labels['train']['class'][epoch],
                           labels['train']['classifier'][epoch])

# gather classifier outputs for the valset and testset
for split, loader in zip(['test', 'val'],
                         [testloader, valloader]):
    console_logger.info(f'Gathering {["testing", "validation"][split == "val"]} set outputs')
    get_classifier_outputs(classifiers, loader, clf_outputs[split], labels[split]['class'], labels[split]['classifier'])

# upload artifacts
console_logger.info("Uploading artifacts")
artifact = wandb.Artifact(f'classifier-outputs-{args.phase}-{wandb.run.id}', 'classifier_outputs',
                          metadata={'dataset': config.dataset, 'source_run': args.run, 'phase': args.phase,
                                    'epochs': args.epochs})

with artifact.new_file('clf_outputs.pkl', 'wb') as f:
    pickle.dump(clf_outputs, f)

with artifact.new_file('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

with artifact.new_file('meta.pkl', 'wb') as f:
    metadata = {'class_order': data.class_order, 'cls_to_clf_idx': controller.cls_to_clf_idx, 'epochs': args.epochs,
                'n_class_per_phase': n_class_per_phase, 'train_size': len(trainset),
                'test_size': len(testset), 'val_size': len(valset), 'phase': args.phase}
    pickle.dump(metadata, f)

with artifact.new_file('config.yaml', 'w') as f:
    yaml.dump(config.as_dict(), f)

wandb.log_artifact(artifact)

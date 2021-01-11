import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import wandb

import argparse

from exp2.controller import Controller
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

clf_outputs = {
    'train': torch.zeros(args.epochs, len(classifiers), len(trainset),
                         n_class_per_phase + config.other, dtype=torch.float32,
                         device=device),  # (epoch, clf, sample, outputs)
    'val': torch.zeros(len(classifiers), len(valset),
                       n_class_per_phase + config.other, dtype=torch.float32,
                       device=device),  # (clf, sample, outputs)
    'test': torch.zeros(len(classifiers), len(testset),
                        n_class_per_phase + config.other, dtype=torch.float32,
                        device=device),  # (clf, sample, outputs)
}

labels = {
    'train': {
        'class': torch.empty(args.epochs, len(trainset), dtype=torch.int32),
        'classifier': torch.empty(args.epochs, len(trainset), dtype=torch.int32)
    },
    'val': {
        'class': torch.empty(len(valset), dtype=torch.int32),
        'classifier': torch.empty(len(valset), dtype=torch.int32)
    },
    'test': {
        'class': torch.empty(len(testset), dtype=torch.int32),
        'classifier': torch.empty(len(testset), dtype=torch.int32)
    },
}

config.batch_size = args.batch_size
trainloader = create_loader(config, trainset)
valloader = create_loader(config, valset)
testloader = create_loader(config, testset)

console_logger.info('trainset size: %d', len(trainset))
console_logger.info('validation size: %d', len(valset))
console_logger.info('test size: %d', len(testset))

# load a raw controller just to map class labels to classifiers
controller = Controller(config, None, classifiers)


def get_classifier_outputs(classifiers, loader, output, cls_labels, clf_labels):
    i = 0
    for inputs, labels, ids in loader:
        inputs = inputs.to(device)
        bsize = inputs.size(0)
        features = feature_extractor(inputs)
        for j, clf in enumerate(classifiers):
            output[j, i:i + bsize] = clf(features).cpu()
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
artifact = wandb.Artifact(f'classifier-outputs-{args.phase}-{wandb.run.id}', 'classifier_outputs')

with artifact.new_file('clf_outputs.pkl', 'wb') as f:
    pickle.dump(clf_outputs, f)

with artifact.new_file('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

wandb.log_artifact(artifact)

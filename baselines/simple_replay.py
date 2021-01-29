import argparse
from typing import List

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

import utils
from exp2.classifier import Checkpoint
from exp2.config import load_configs
from exp2.data import prepare_data, create_loader
from exp2.feature_extractor import load_pretrained
from exp2.memory import MemoryManagerBasic
from exp2.models import model_mapping
from exp2.utils import split_model
from logger import Logger
from utils import get_default_device, AverageMeter, TrainingStopper, cutmix_data

console_logger = utils.get_console_logger(name='main')
DEVICE = get_default_device()


def create_model(config, n_classes):
    if config.pretrained:
        pretrained = load_pretrained(config.model)
        fe, head_constructor = split_model(config, pretrained)
        fe.eval()
        head = head_constructor(n_classes=n_classes)
        _model = torch.nn.Sequential(fe, head)
        _model.eval = lambda: head.eval()
        _model.train = lambda: head.train()
    else:
        _model = model_mapping[config.model](num_classes=n_classes)
    model: Checkpoint = Checkpoint.wrap(_model, checkpoint_file=wandb.run.dir + '/' + 'checkpoint.pth')
    return model


@torch.no_grad()
def evaluate_model(config, logger, model, dataloaders: List[DataLoader], weights: List = None, log_prefx='test'):
    """Evaluates the model on multiple dataloaders with different weights."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if weights is None:
        weights = [1] * len(dataloaders)

    for w, loader in zip(weights, dataloaders):
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, 1)

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size * w)
            acc_meter.update(torch.eq(predictions, labels).type(torch.FloatTensor).mean(), batch_size * w)

    logger.log({log_prefx + '_loss': loss_meter.avg, log_prefx + '_acc': acc_meter.avg})
    return loss_meter.avg


def create_optimizer(config, model):
    if config.faster_output_learning_rate:
        final = get_final_layer(model)
        all_except_final = list(filter(lambda p: all(p is not i for i in final.parameters()), model.parameters()))
        optimizer = torch.optim.SGD([{'params': all_except_final},
                                     {'params': final.parameters(), 'lr': config.lr * 10}],
                                    momentum=0.9, lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    return optimizer


def train_model(config, model: Checkpoint, train_loader: DataLoader, val_loaders: List[DataLoader],
                val_weights: List = None, logger: Logger = None):
    optimizer = create_optimizer(config, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    non_blocking = config.torch['non_blocking']
    stopper = TrainingStopper(config)
    model.remove_checkpoint()
    for ep in range(config.epochs):
        model.train()
        loss_meter = AverageMeter()

        # train
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=non_blocking), labels.to(DEVICE, non_blocking=non_blocking)

            do_cutmix = config.regularization == 'cutmix' and np.random.rand(1) < config.cutmix_prob
            if do_cutmix > 0:
                inputs, labels_a, labels_b, lam = cutmix_data(x=inputs, y=labels, alpha=config.cutmix_alpha)

            # Forward, backward passes then step
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b) \
                if do_cutmix \
                else criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size)
        logger.log({'train_loss': loss_meter.avg, 'epoch': ep})

        # validate
        if val_loaders is not None:
            val_loss = evaluate_model(config, logger, model, val_loaders, val_weights, log_prefx='val')
            model.checkpoint(optimizer, val_loss, epoch=ep)
            stopper.update(val_loss)

        # schedule learning rate
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        logger.log({'lr': lr})

        logger.commit()

        if stopper.do_stop():
            console_logger.info(f'Early stopping at epoch: {ep}')
            return


def get_final_layer(model) -> torch.nn.Linear:
    assert isinstance(model, torch.nn.Sequential)
    while not isinstance(model[-1], torch.nn.Linear):
        model = model[-1]
    return model[-1]


@torch.no_grad()
def maybe_reset_model_weights(config, model):
    if config.reset_weights == 'none':
        return model

    console_logger.info('Resetting model weights')
    final = get_final_layer(model)
    if config.reset_weights == 'output':
        # get final layer
        final.weight.fill_(0.0)
    elif config.reset_weights == 'all':
        n_classes = final.out_features
        model = create_model(config, n_classes).to(DEVICE)
    else:
        raise ValueError(f'config.reset_weights should be one of "none", "output", and "all".')
    return model


def get_dataset_weight(dataset):
    if len(dataset) == 0:
        return 0
    return len(dataset.classes) / len(dataset)


def run(config):
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)
    train_sz = int(config.memory_size * (1 - config.val_size))
    val_sz = config.memory_size - train_sz
    memory_manager = MemoryManagerBasic(sizes=[train_sz, val_sz], source=data.train_source,
                                        train_transform=data.train_transform,
                                        test_transform=data.test_transform, names=['train', 'val'])
    model = create_model(config, len(data.class_order)).to(DEVICE)
    logger.log({'class_order': data.class_order})

    for phase in range(1, config.n_phases + 1):
        do_train = config.phase is None or config.phase == phase
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # GDumb updates memory before training
        if config.method == 'gdumb':
            console_logger.info('Updating memory')
            memory_manager.update_memories(trainset)
            memory_manager.log_memory_sizes()

        # train model on memory samples
        if do_train:
            model = maybe_reset_model_weights(config, model)

            memory_trainset = memory_manager['train'].get_dataset(train=True)
            memory_valset = memory_manager['val'].get_dataset(train=False)

            weights = None
            trainset_train = None
            trainset_val = None
            # prepare train and validation sets
            if config.method == 'gdumb':
                # GDumb only trains on memory samples
                train_loader = create_loader(config, memory_trainset, shuffle=True)
                val_loaders = [create_loader(config, memory_valset)]
            else:
                # Simple replay, trains on current data + memory samples
                trainset_train, trainset_val = trainset.split(test_size=config.val_size)
                if phase == 1:
                    memory_trainset = None
                train_loader = create_loader(config, main_dataset=trainset_train, memoryset=memory_trainset,
                                             shuffle=True)
                val_loaders = [create_loader(config, trainset_val),
                               create_loader(config, memory_valset)]
                weights = [get_dataset_weight(trainset_val), get_dataset_weight(memory_valset)]

            console_logger.info(f'Training the model')
            train_model(config, model, train_loader, val_loaders, val_weights=weights, logger=logger)

            # Simple replay updates memory
            if config.method == 'simple_replay':
                console_logger.info('Updating memory')
                # only trained samples can go into train memory to maintain unbiased validation in the future
                memory_manager.update_memories(trainset_train, memories=[memory_manager['train']])
                memory_manager.update_memories(trainset_val, memories=[memory_manager['val']])
                memory_manager.log_memory_sizes()

            # test the model
            console_logger.info('Testing the model')
            model.load_best()

            cumul_test_loader = create_loader(config, cumul_testset)
            evaluate_model(config, logger, model, dataloaders=[cumul_test_loader])
            logger.commit()

            if config.phase == phase:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='*', default='./config/gdumb.yaml', help='Config file')
    parser.add_argument('--wandb_group', type=str, help='W&B group in experiments')
    parser.add_argument('--project', type=str, help='W&B project name')
    args = parser.parse_args()

    config_dict = load_configs(args.configs)

    # init wandb and get its wrapped config
    init_dict = dict()
    if args.project:
        init_dict['project'] = args.project
    if args.wandb_group:
        init_dict['group'] = args.wandb_group

    wandb.init(config=config_dict, **init_dict)
    config = wandb.config

    # run
    run(config)

import argparse
from typing import List

import numpy as np
import torch
import torch.optim.lr_scheduler as sch
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import utils
from exp2.config import load_configs
from exp2.controller import BiC
from exp2.data import prepare_data, create_loader
from exp2.memory import MemoryManagerBasic
from exp2.models import model_mapping
from exp2.models.efficientnet import split_efficientnet, EfficientNet
from exp2.models.simple_net import split_simple_net_20_classes
from exp2.models.splitting import load_pretrained, log_architecture
from exp2.models.utils import Checkpoint, ClassMapping
from logger import Logger
from utils import get_default_device, AverageMeter, TrainingStopper, cutmix_data

console_logger = utils.get_console_logger(name='main')
DEVICE = get_default_device()


def create_model(config, n_classes):
    if config.pretrained:
        pretrained = load_pretrained(config.model)
        if isinstance(pretrained, EfficientNet):
            fe, head_constructor = split_efficientnet(config, pretrained)
        else:
            fe, head_constructor = split_simple_net_20_classes(config, pretrained)

        fe.eval()
        head = head_constructor(n_classes=n_classes)
        _model = torch.nn.Sequential(fe, head)
        _model.eval = lambda *args: head.eval(*args)
        _model.train = lambda *args: head.train(*args)
    else:  # initialize a model from scratch
        _model = model_mapping[config.model](num_classes=n_classes)
    # enable the model to store checkpoints
    model: Checkpoint = Checkpoint.wrap(_model, checkpoint_file=wandb.run.dir + '/' + 'checkpoint.pth')
    return model


@torch.no_grad()
def evaluate_model(config, logger, model, dataloaders: List[DataLoader], bic: BiC = None, weights: List = None,
                   log_prefx='test', class_order=None, log_confusion_matrix=False):
    """Evaluates the model on multiple dataloaders with different weights."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if weights is None:
        weights = [1] * len(dataloaders)

    all_predictions = []
    all_labels = []

    if bic is not None:
        console_logger.info('Evaluating with BiC layer.')
    class_map = model.class_map
    n_classes = len(class_map.classes)
    for w, loader in zip(weights, dataloaders):
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs[:, :n_classes]  # select only relevant neurons
            loc_labels = class_map.localize_labels(labels)
            if bic:
                outputs = bic(outputs)
            loss = criterion(outputs, loc_labels)
            predictions = class_map.globalize_labels(torch.argmax(outputs, 1))
            all_predictions.append(predictions)
            all_labels.append(labels)

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size * w)
            acc_meter.update(torch.eq(predictions, labels).type(torch.FloatTensor).mean(), batch_size * w)

    if log_confusion_matrix:
        all_predictions = torch.cat(all_predictions).cpu()
        all_labels = torch.cat(all_labels).cpu()
        cm = confusion_matrix(all_labels, all_predictions, labels=class_order)
        logger.log_confusion_matrix(cm, labels=class_order)
    logger.log({log_prefx + '_loss': loss_meter.avg, log_prefx + '_acc': acc_meter.avg})
    return loss_meter.avg, acc_meter.avg


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


def create_lr_scheduler(config, optimizer):
    if config.lr_scheduler == 'exp':
        return sch.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.gamma)
    elif config.lr_scheduler == 'reduce_on_plateu':
        return sch.ReduceLROnPlateau(optimizer, 'min', factor=config.gamma,
                                     patience=config.lr_patience,
                                     verbose=True, min_lr=0.00001)
    else:
        raise ValueError(f"config.lr_scheduler be one of 'exp', 'reduce_on_plateu' but got {config.lr_scheduler}")


def scheduler_step(lr_scheduler, loss):
    if isinstance(lr_scheduler, sch.ExponentialLR):
        lr_scheduler.step()
    elif isinstance(lr_scheduler, sch.StepLR):
        lr_scheduler.step()
    elif isinstance(lr_scheduler, sch.ReduceLROnPlateau):
        lr_scheduler.step(loss)


def get_last_learning_rate(lr_scheduler):
    try:
        return lr_scheduler.get_last_lr()[0]
    except AttributeError:
        pass

    try:
        return lr_scheduler._last_lr[0]
    except AttributeError:
        pass

    return lr_scheduler.optimizer.param_groups[0]['lr']


def train_model(config, model: Checkpoint, train_loader: DataLoader, val_loaders: List[DataLoader],
                val_weights: List = None, logger: Logger = None):
    model.train()
    optimizer = create_optimizer(config, model)
    lr_scheduler = create_lr_scheduler(config, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    non_blocking = config.torch['non_blocking']
    stopper = TrainingStopper(config)
    model.remove_checkpoint()
    class_map = model.class_map
    n_classes = len(class_map.classes)
    for ep in range(config.epochs):
        loss_meter = AverageMeter()

        # train
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=non_blocking), labels.to(DEVICE, non_blocking=non_blocking)

            # Forward, backward passes then step
            outputs = model(inputs)
            outputs = outputs[:, :n_classes]  # select only relevant neurons
            loc_labels = class_map.localize_labels(labels)
            do_cutmix = config.regularization == 'cutmix' and np.random.rand(1) < config.cutmix_prob
            if do_cutmix > 0:
                inputs, labels_a, labels_b, lam = cutmix_data(x=inputs, y=loc_labels, alpha=config.cutmix_alpha)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, loc_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size)
        logger.log({'train_loss': loss_meter.avg, 'epoch': ep})

        # validate
        if val_loaders is not None:
            val_loss, val_acc = evaluate_model(config, logger, model, val_loaders, weights=val_weights,
                                               log_prefx='val')
            val_metric = val_loss if config.val_metric == 'loss' else -val_acc
            model.checkpoint(optimizer, val_metric, epoch=ep)
            stopper.update(val_metric)

            # schedule learning rate
            scheduler_step(lr_scheduler, val_metric)

        logger.log({'lr': get_last_learning_rate(lr_scheduler)})
        logger.commit()

        if stopper.do_stop():
            console_logger.info(f'Early stopping at epoch: {ep}')
            break

    # Training finished, load best checkpoint
    model.load_best()


def get_final_layer(model) -> torch.nn.Linear:
    while not isinstance(model, torch.nn.Linear):
        last_child = list(model.children())[-1]
        model = last_child
    return model


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
    n_classes = len(data.class_order)
    model = create_model(config, n_classes=n_classes).to(DEVICE)
    bic = BiC().to(DEVICE)
    class_map = model.class_map = ClassMapping()
    log_architecture('model', model, input_data=torch.randn(1, *config.input_size, device=DEVICE))
    logger.log({'class_order': data.class_order})

    for phase in range(1, config.n_phases + 1):
        do_train = config.phase is None or config.phase == phase
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)
        bic.extend(trainset.classes)
        class_map.extend(trainset.classes)

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
            config.update({'lr': max(config.min_lr, config.lr * config.global_gamma)}, allow_val_change=True)

            # Simple replay updates memory
            if config.method == 'simple_replay':
                console_logger.info('Updating memory')
                # only trained samples can go into train memory to maintain unbiased validation in the future
                memory_manager.update_memories(trainset_train, memories=[memory_manager['train']])
                memory_manager.update_memories(trainset_val, memories=[memory_manager['val']])
                memory_manager.log_memory_sizes()

            # Train BiC
            if phase > 1 and config.bic:
                dataset_for_bic = memory_manager['val'].get_dataset(train=True)
                loader_for_bic = create_loader(config, dataset_for_bic)
                train_bic(bic, config, logger, loader_for_bic, model, trainset.classes)

            # test the model
            console_logger.info('Testing the model')

            cumul_test_loader = create_loader(config, cumul_testset)
            evaluate_model(config, logger, model, bic=bic, dataloaders=[cumul_test_loader],
                           class_order=data.class_order, log_confusion_matrix=True)
            logger.commit()

            if config.phase == phase:
                return


def train_bic(bic, config, logger, loader, model, classes):
    bic.set_biased_classes(classes)
    console_logger.info('Training BiC')
    model.eval()
    n_classes = len(bic.classes)

    def forward_fn(batch):
        non_blocking = config.torch['non_blocking']
        inputs, labels, _ = batch
        inputs, labels = inputs.to(DEVICE, non_blocking=non_blocking), labels.to(DEVICE, non_blocking=non_blocking)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs[:, :n_classes]
        return outputs

    bic.train_(config, loader, forward_fn, logger)


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

import argparse
from typing import List

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as sch

import utils
from exp2.models.utils import Checkpoint
from exp2.models.splitting import load_pretrained
from exp2.config import load_configs
from exp2.data import prepare_data, create_loader
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
def evaluate_model(config, logger, model, mask, dataloaders: List[DataLoader], weights: List = None, log_prefx='test',
                   class_order=None, log_confusion_matrix=False):
    """Evaluates the model on multiple dataloaders with different weights."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if weights is None:
        weights = [1] * len(dataloaders)

    all_predictions = []
    all_labels = []

    bic = None
    if hasattr(model, 'BiC'):
        console_logger.info('Model has BiC layer.')
        bic = model.BiC

    mask = mask.to(DEVICE)
    for w, loader in zip(weights, dataloaders):
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            outputs.data[:, ~mask] = -1e31
            if bic:
                outputs = apply_bic(outputs, bic['alpha'], bic['beta'], bic['bic_mask'])
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, 1)
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


def apply_bic(outputs, alpha, beta, bic_mask):
    outputs = outputs.clone()
    outputs[:, bic_mask] = outputs[:, bic_mask] * alpha + beta
    return outputs


def train_bic(config, model, mask, bic_mask, loader: DataLoader, logger):
    # BiC parameters
    alpha = torch.tensor([1.], dtype=torch.float32, requires_grad=True, device=DEVICE)
    beta = torch.tensor([0.], dtype=torch.float32, requires_grad=True, device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=[alpha, beta], lr=config.bic_lr, momentum=0.9)
    mask = mask.to(DEVICE)
    bic_mask = bic_mask.to(DEVICE)
    non_blocking = config.torch['non_blocking']

    model.eval()
    for ep in range(config.bic_epochs):
        loss_meter = AverageMeter()

        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=non_blocking), labels.to(DEVICE, non_blocking=non_blocking)

            # Forward, backward passes then step
            with torch.no_grad():
                outputs = model(inputs)
                outputs.data[:, ~mask] = -1e31

            # apply BiC layer
            outputs = apply_bic(outputs, alpha, beta, bic_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size)
        logger.log({'bic_train_loss': loss_meter.avg, 'bic_epoch': ep})
        logger.commit()

    model.BiC = {'bic_mask': bic_mask, 'alpha': alpha, 'beta': beta}


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


def train_model(config, model: Checkpoint, mask, train_loader: DataLoader, val_loaders: List[DataLoader],
                val_weights: List = None, logger: Logger = None):
    if hasattr(model, 'BiC'):
        del model.BiC

    optimizer = create_optimizer(config, model)
    lr_scheduler = create_lr_scheduler(config, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    non_blocking = config.torch['non_blocking']
    stopper = TrainingStopper(config)
    model.remove_checkpoint()
    mask = mask.to(DEVICE)
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
            outputs.data[:, ~mask] = -1e31
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
            val_loss, val_acc = evaluate_model(config, logger, model, mask, val_loaders, val_weights, log_prefx='val')
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
    n_classes = len(data.class_order)
    model = create_model(config, n_classes=n_classes).to(DEVICE)
    mask = torch.zeros(n_classes, dtype=torch.bool)
    bic_mask = None

    logger.log({'class_order': data.class_order})

    for phase in range(1, config.n_phases + 1):
        do_train = config.phase is None or config.phase == phase
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # update mask
        mask[trainset.classes] = 1

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
            train_model(config, model, mask, train_loader, val_loaders, val_weights=weights, logger=logger)
            config.update({'lr': max(config.min_lr, config.lr * 0.5)}, allow_val_change=True)

            # Simple replay updates memory
            if config.method == 'simple_replay':
                console_logger.info('Updating memory')
                # only trained samples can go into train memory to maintain unbiased validation in the future
                memory_manager.update_memories(trainset_train, memories=[memory_manager['train']])
                memory_manager.update_memories(trainset_val, memories=[memory_manager['val']])
                memory_manager.log_memory_sizes()

            # Train BiC
            if phase > 1 and config.apply_bic:
                bic_mask = torch.zeros(n_classes, dtype=torch.bool)
                bic_mask[trainset.classes] = 1
                console_logger.info('Training BiC')
                dataset_for_bic = memory_manager['val'].get_dataset(train=True)
                loader_for_bic = create_loader(config, dataset_for_bic)
                train_bic(config, model, mask, bic_mask, loader_for_bic, logger)
                console_logger.info(f'BiC parameters: {model.BiC["alpha"].item(), model.BiC["beta"].item()}')

            # test the model
            console_logger.info('Testing the model')

            cumul_test_loader = create_loader(config, cumul_testset)
            evaluate_model(config, logger, model, mask, dataloaders=[cumul_test_loader], class_order=data.class_order,
                           log_confusion_matrix=True)
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

import argparse

import torch
import wandb
from torch.utils.data import DataLoader

import utils
from exp2.classifier import Checkpoint
from exp2.config import load_configs
from exp2.data import prepare_data, PartialDataset
from exp2.memory import Memory
from exp2.models import model_mapping
from logger import Logger
from utils import get_default_device, AverageMeter

console_logger = utils.get_console_logger(name='main')
DEVICE = get_default_device()


def create_model(config, num_classes):
    model = model_mapping[config.model](num_classes=num_classes)
    model: Checkpoint = Checkpoint.wrap(model, checkpoint_file=wandb.run.dir + '/' + 'checkpoint.pth')
    return model


@torch.no_grad()
def test_model(config, model, dataset, logger: Logger, prefx='test'):
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=config.batch_size)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for inputs, labels, _ in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predictions = torch.argmax(outputs, 1)

        batch_size = len(inputs)
        loss_meter.update(loss, batch_size)
        acc_meter.update(torch.eq(predictions, labels).type(torch.FloatTensor).mean(), batch_size)

    logger.log({prefx + '_loss': loss_meter.avg, prefx + '_acc': acc_meter.avg})
    return loss_meter.avg


def train_model(config, model: Checkpoint, trainset: PartialDataset, valset: PartialDataset, logger: Logger):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True,
                              pin_memory=config.torch['pin_memory'], num_workers=config.torch['num_workers'])
    non_blocking = config.torch['non_blocking']
    for ep in range(config.epochs):
        loss_meter = AverageMeter()

        # train
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=non_blocking), labels.to(DEVICE, non_blocking=non_blocking)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size)
        logger.log({'train_loss': loss_meter.avg, 'epoch': ep})

        # validate
        if len(valset) > 0:
            val_loss = test_model(config, model, valset, logger, prefx='val')
            model.checkpoint(optimizer, val_loss, epoch=ep)

        # schedule learning rate
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        logger.log({'lr': lr})

        logger.commit()


def update_memories(dataset, memory, val_memory):
    split_idx = int(len(dataset) * memory.max_size / (memory.max_size + val_memory.max_size))
    memory.update(ids=dataset.ids[:split_idx], new_classes=dataset.classes)
    val_memory.update(ids=dataset.ids[split_idx:], new_classes=dataset.classes)
    a, b = memory.get_n_samples(), val_memory.get_n_samples()
    console_logger.info(f'Memory sizes (train + val): {a} + {b} = {a + b}')


def run(config):
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)

    train_memory_sz = int(config.memory_size * (1 - config.val_size))
    memory = Memory(train_memory_sz, data.train_source, train_transform=data.train_transform,
                    test_transform=data.test_transform)
    val_memory = Memory(config.memory_size - train_memory_sz, data.train_source, train_transform=data.train_transform,
                        test_transform=data.test_transform)

    model = create_model(config, len(data.class_order)).to(DEVICE)
    logger.log({'class_order': data.class_order})

    for phase in range(1, config.n_phases + 1):
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # update memory
        console_logger.info('Updating memory')
        update_memories(trainset, memory, val_memory)

        # train model on memory samples
        console_logger.info(f'Training the model')
        train_model(config, model, memory.get_dataset(train=True), val_memory.get_dataset(train=False), logger)

        # test the model
        console_logger.info('Testing the model')
        model.load_best()
        test_model(config, model, cumul_testset, logger)
        logger.commit()


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

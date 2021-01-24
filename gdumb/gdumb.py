import argparse

import torch
import wandb
from torch.utils.data import DataLoader

import utils
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


def train_model(config, model, dataset: PartialDataset, logger: Logger):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    trainset, valset = dataset.split(test_size=config.val_size)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    for ep in range(config.epochs):
        loss_meter = AverageMeter()

        # train
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            loss_meter.update(loss, batch_size)
        logger.log({'train_loss': loss_meter.avg, 'epoch': ep})

        # validate
        test_model(config, model, valset, logger, prefx='val')

        # schedule learning rate
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]
        logger.log({'lr': lr})

        logger.commit()


def run(config):
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)
    memory = Memory(config.memory_size, data.train_source, train_transform=data.train_transform,
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
        memory.update(trainset.ids, new_classes=trainset.classes)
        console_logger.info(f'Memory size: {memory.get_n_samples()}')

        # train model on memory samples
        console_logger.info(f'Training the model')
        train_model(config, model, memory.get_dataset(train=True), logger)

        # test the model
        console_logger.info('Testing the model')
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
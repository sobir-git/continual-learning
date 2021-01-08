import wandb
import numpy as np

import utils
from exp2.config import parse_args, make_nested
from exp2.data import prepare_data
from exp2.memory import create_memory_storages, update_memories, log_total_memory_sizes
from exp2.model import Model
from logger import Logger, dict_deep_update

console_logger = utils.get_console_logger(name='main')


def run(config):
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)
    clf_memory, ctrl_memory, shared_memory = create_memory_storages(config, data)
    model = Model(config, logger=logger)
    logger.log({'class_order': data.class_order})

    # here comes the training algorithm
    for phase in range(1, config.n_phases + 1):
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)
        model.phase_start(phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # add samples to ctrl_memory, and remove those from trainset
        console_logger.info('Updating controller memory')
        added_ids = ctrl_memory.update(trainset.ids, new_classes=trainset.classes)
        left_ids = np.setdiff1d(trainset.ids, added_ids)
        trainset = trainset.subset(left_ids)

        # train a new classifier on new samples
        console_logger.info('Training a new classifier')
        model.train_new_classifier(trainset, clf_memory=clf_memory, ctrl_memory=ctrl_memory,
                                   shared_memory=shared_memory)

        # update classifier and share memory
        console_logger.info('Updating classifier and shared memory')
        update_memories(trainset, clf_memory=clf_memory, shared_memory=shared_memory)

        # train a new controller
        console_logger.info('Training a new controller')
        model.train_a_new_controller(ctrl_memory=ctrl_memory, shared_memory=shared_memory)

        # test the model
        console_logger.info('Testing the model')
        model.phase_end()
        model.test(cumul_testset)
        log_total_memory_sizes(clf_memory, ctrl_memory, shared_memory)


if __name__ == '__main__':
    import os
    import yaml

    args = parse_args()
    default_config = dict()
    if os.path.isfile(args.defaults):
        console_logger.info('Loading config defaults: %s', args.defaults)
        with open(args.defaults) as f:
            y = yaml.safe_load(f)
            default_config.update(y)

    config = dict()
    if os.path.isfile(args.config):
        console_logger.info('Loading config: %s', args.config)
        with open(args.config) as f:
            y = yaml.safe_load(f)
            config.update(y)
    config = make_nested(config, ['clf', 'ctrl'])
    default_config = make_nested(default_config, ['clf', 'ctrl'])

    # update defaults
    dict_deep_update(default_config, config)
    final_config = default_config

    # init wandb and get its wrapped config
    group = args.wandb_group
    wandb.init(project='exp2', group=group, config=final_config)
    config = wandb.config

    # create log directory
    os.makedirs(config.logdir, exist_ok=True)

    # run
    run(config)

import wandb

import utils
from exp2.config import parse_args
from exp2.data import prepare_data
from exp2.memory import create_memory_storages, update_memories
from exp2.model import Model
from logger import Logger


def run(config):
    console_logger = utils.get_console_logger(config.logdir, name='main')
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)
    clf_memory, ctrl_memory = create_memory_storages(config, data)
    model = Model(config, logger=logger)
    logger.log({'class_order': data.class_order})

    for phase in range(1, config.n_phases + 1):
        model.phase_start(phase)
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # train a new classifier on new samples
        console_logger.info('Training a new classifier')
        otherset = clf_memory.get_dataset() if config.other else None
        model.train_new_classifier(trainset, otherset=otherset)

        # add the new training samples to memory
        console_logger.info('Updating memory')
        update_memories(ctrl_memory, clf_memory, trainset)
        #
        # # update previous classifiers
        # if config.update_classifiers:
        #     console_logger.info('Updating previous classifiers')
        #     model.update_prev_classifiers(memory.get_dataset())

        # train a new controller
        console_logger.info('Training a new controller')
        model.train_a_new_controller(ctrl_memory.get_dataset())

        # test the model
        console_logger.info('Testing the model')
        model.phase_end()
        model.test(cumul_testset)


if __name__ == '__main__':
    config = parse_args()
    group = config.wandb_group or 'test'
    config_exclude_keys = config.get_excluded_keys() + ['wandb_group']
    wandb.init(project='exp2', group=group, config=config, config_exclude_keys=config_exclude_keys)
    run(config)

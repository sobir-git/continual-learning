import wandb

import utils
from exp2.config import parse_args
from exp2.data import prepare_data
from exp2.memory import Memory
from exp2.model import Model
from logger import Logger


def run(config):
    console_logger = utils.get_console_logger(config.logdir, name='main')
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)
    memory = Memory(config, data.train_source, data.train_transform, data.test_transform)
    model = Model(config, logger=logger)

    for phase in range(1, config.n_phases + 1):
        model.phase_start(phase)
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # train a new classifier on new samples
        console_logger.info('Training a new classifier')
        otherset = memory.get_dataset() if config.other else None
        model.train_new_classifier(trainset, otherset=otherset)

        # add the new training samples to memory
        console_logger.info('Updating memory')
        memory.update(ids=trainset.ids, new_classes=trainset.classes)
        #
        # # update previous classifiers
        # if config.update_classifiers:
        #     console_logger.info('Updating previous classifiers')
        #     model.update_prev_classifiers(memory.get_dataset())

        # train a new controller
        console_logger.info('Training a new controller')
        model.train_a_new_controller(memory.get_dataset())

        # test the model
        console_logger.info('Testing the model')
        model.phase_end()
        model.test(cumul_testset)


if __name__ == '__main__':
    config = parse_args()
    group = config.wandb_group or 'test'
    wandb.init(project='exp2', group=group, config=config, config_exclude_keys={'wandb_group'})
    run(config)

import numpy as np
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
        console_logger.info(f'== Starting phase {phase} ==')
        logger.pin('phase', phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)

        # compute importance scores of the training samples, (before mixing with otherset!)
        if config.memory_sampler == 'loss_aware' and phase > 1:
            console_logger.info('Computing importance scores')
            # here the importance scores are the maximum logit of the previous controller
            new_ids, fresh_scores = model.compute_fresh_importance_scores(trainset)
        else:
            # equal scores
            new_ids = trainset.ids[:]
            fresh_scores = np.ones_like(new_ids)

        console_logger.info('Training a new classifier')
        # train a new classifier on new samples, (by assigning otherset first)
        if config.other:  # assign otherset
            trainset.set_otherset(memory.get_dataset())
        model.train_new_classifier(trainset)

        console_logger.info('Updating memory')
        # add new training samples to memory
        memory.update(new_ids, fresh_scores, trainset.classes)

        # update previous classifiers
        if phase > 1 and config.update_classifiers:
            console_logger.info('Updating previous classifiers')
            model.update_prev_classifiers(memory.get_dataset())

        console_logger.info('Training a new controller')
        # train a new controller
        model.train_a_new_controller(memory.get_dataset())

        # update memory
        if config.memory_sampler != 'greedy' and config.update_scores:
            console_logger.info('Updating importance scores in memory')
            # recompute scores with the new controller, these scores are more accurate
            ids, scores = model.recompute_importance_scores(memory.get_dataset())
            memory.update_scores(ids, scores)

        console_logger.info('Testing the model')
        # test the model
        model.test(cumul_testset)


if __name__ == '__main__':
    config = parse_args()
    group = config.wandb_group or 'test'
    wandb.init(project='exp2', group=group, config=config, config_exclude_keys={'wandb_group'})
    run(config)

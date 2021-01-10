import numpy as np
import wandb
import yaml

import utils
from exp2.config import parse_args, load_configs
from exp2.data import prepare_data
from exp2.memory import MemoryManager
from exp2.model import Model
from logger import Logger

console_logger = utils.get_console_logger(name='main')


def run(config):
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))
    # prepare data
    data = prepare_data(config)
    memory_manager = MemoryManager(config, data)
    clf_memory, ctrl_memory, shared_memory = memory_manager.get_memories()
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
        memory_manager.update_memories(trainset, phase=phase)

        # train a new controller
        console_logger.info('Training a new controller')
        model.train_a_new_controller(ctrl_memory=ctrl_memory, shared_memory=shared_memory)

        # test the model
        console_logger.info('Testing the model')
        model.phase_end()
        model.test(cumul_testset)

    # inform phase end
    memory_manager.on_training_end()


if __name__ == '__main__':
    import os

    args = parse_args()
    config_files = [args.defaults] + args.configs
    config_dict = load_configs(config_files)

    # init wandb and get its wrapped config
    group = args.wandb_group
    wandb.init(project='exp2', group=group, config=config_dict, job_type='clf-training')
    config = wandb.config

    # extend logging directory with the current unique run name
    config.update({'logdir': os.path.join(config.logdir, wandb.run.name)}, allow_val_change=True)

    # upload config files
    config_yaml = yaml.dump(config_dict)
    config_art = wandb.Artifact('project-source', type='configs')
    for file in config_files:
        config_art.add_file(file)
    with config_art.new_file('combined-config.yaml') as f:
        f.write(config_yaml)
    wandb.run.use_artifact(config_art)

    # create log directory
    os.makedirs(config.logdir, exist_ok=True)

    # run
    run(config)

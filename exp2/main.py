import wandb
import yaml

import utils
from exp2.config import parse_args, load_configs
from exp2.data import prepare_data
from exp2.model import JointModel
from logger import Logger

console_logger = utils.get_console_logger(name='main')


def run(config):
    logger = Logger(config, console_logger=console_logger)
    console_logger.info('config:' + str(config))

    # prepare data
    data = prepare_data(config)

    # create model
    model = JointModel(config, data.train_source, data.train_transform, data.test_transform, logger=logger)

    # phases start with 1, classifier indices start with 0 (phase - 1)
    for phase in range(1, config.n_phases + 1):
        console_logger.info(f'>>> >>> Starting phase {phase} <<< <<<')
        logger.pin('phase', phase)
        model.phase_start(phase)

        # get the new samples
        trainset, testset, cumul_testset = data.get_phase_data(phase)
        model.on_receive_phase_data(trainset)

        # test the model
        console_logger.info('Testing the model')
        model.test(cumul_testset)
        logger.commit()
        model.phase_end()


if __name__ == '__main__':
    import os

    args = parse_args()
    config_files = [args.defaults] + args.configs
    config_dict = load_configs(config_files)

    # init wandb and get its wrapped config
    group = args.wandb_group
    init_dict = dict()
    if args.project:
        init_dict['project'] = args.project
    if args.wandb_group:
        init_dict['group'] = args.wandb_group

    wandb.init(config=config_dict, **init_dict)
    config = wandb.config

    # extend logging directory with the current unique run name
    config.update({'logdir': os.path.join(config.logdir, wandb.run.name)}, allow_val_change=True)

    # upload config files
    config_yaml = yaml.dump(config_dict)
    config_art = wandb.Artifact(f'config-{wandb.run.id}', type='configs')
    for file in config_files:
        config_art.add_file(file)
    with config_art.new_file('combined-config.yaml') as f:
        f.write(config_yaml)
    wandb.run.use_artifact(config_art)

    # create log directory
    os.makedirs(config.logdir, exist_ok=True)

    # run
    run(config)

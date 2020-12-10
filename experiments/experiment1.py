import torch
import wandb
from torch import optim

import models.concrete.single
from dataloader import VisionDataset
from logger import Logger
from models.bnet_base import BranchNet, BnetTrainer
from opts import parse_args
from training import StandardTrainer
from utils import AverageMeter, get_console_logger, save_pretrained_model, load_pretrained_model, \
    get_default_device, Timer

device = get_default_device()


def create_optimizer(opt, model):
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
        return optimizer


def create_scheduler(opt, optimizer, n_epochs=None):
    scheduler = None
    if opt.scheduler == 'const':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    elif opt.scheduler == 'exp':
        assert n_epochs is not None, "Number of epochs is required for the exponential learning rate"
        gamma = (opt.minlr / opt.maxlr) ** (1 / n_epochs)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return scheduler


def exp1(opt):
    opt.exp_name += opt.model
    console_logger = get_console_logger(folder=opt.log_dir + '/' + opt.exp_name + '/')
    model = getattr(models.concrete, opt.model)(opt).to(device)
    if opt.watch:
        wandb.watch(model)
    wandb.run.summary['model_graph'] = str(model)

    class_order = list(range(10))
    wandb.config.update({'class_order': class_order})

    vd = VisionDataset(opt, class_order=class_order)
    optimizer = create_optimizer(opt, model)
    console_logger.info(f'Running with device {device}')
    console_logger.info("==> Opts for this training: " + str(opt))
    timer = Timer()

    if isinstance(model, BranchNet):
        trainer = BnetTrainer(opt, model, None, get_default_device(), optimizer)
    else:
        trainer = StandardTrainer(opt, None, model, get_default_device(), optimizer)

    # pretraining
    if opt.num_pretrain_classes > 0:
        try:
            console_logger.info('Trying to load pretrained model...')
            model = load_pretrained_model(opt, model, console_logger)
            pretrain = False
        except Exception as e:
            console_logger.info(f'Failed to load pretrained model: {e}')
            pretrain = True

        if pretrain:
            logger = Logger(opt, console_logger, pref='pretrain')
            trainer.set_logger(logger)

            assert opt.num_pretrain_passes > 0
            console_logger.info(f'==> Starting pretraining')

            if not opt.refresh_scheduler:
                total_epochs = opt.num_pretrain_passes + opt.num_tasks
                scheduler = create_scheduler(opt, optimizer, total_epochs)
            else:
                scheduler = create_scheduler(opt, optimizer, opt.num_pretrain_passes)

            for epoch in range(1, opt.num_pretrain_passes + 1):
                logger.log({'learning_rate': scheduler.get_last_lr()[0], 'epoch': epoch}, commit=True)
                with timer:
                    trainer.train(vd.pretrain_loader)
                console_logger.info(f'Epoch time: {timer.values[-1]:.1f}s')
                scheduler.step()

                with timer:
                    acc = trainer.test(loader=vd.pretest_loader, mask=vd.pretrain_mask, classnames=vd.class_names)
                console_logger.info(f'Test time: {timer.values[-1]:.1f}')
            console_logger.info(f'==> Pretraining completed! Acc: [{acc:.3f}]')

            if opt.old_exp_name != '':
                save_pretrained_model(opt, model)

    # Class-Incremental training
    if opt.num_tasks > 0:
        logger = Logger(opt, console_logger, pref='cl')
        trainer.set_logger(logger)

        console_logger.info(f'==> Starting Class-Incremental training')
        mask = vd.pretrain_mask.clone() if opt.num_pretrain_classes > 0 else torch.zeros(vd.n_classes_in_whole_dataset)
        dataloaders = vd.get_ci_dataloaders()
        cl_accuracy_meter = AverageMeter()
        if opt.refresh_scheduler:
            scheduler = create_scheduler(opt, optimizer, n_epochs=opt.num_tasks)

        for phase, (trainloader, testloader, class_list, phase_mask) in enumerate(dataloaders, start=1):
            console_logger.info(f'==> Phase {phase}: learning classes {", ".join(map(str, class_list))}')
            logger.log({'learning_rate': scheduler.get_last_lr()[0], 'phase': phase}, commit=True)
            with timer:
                trainer.train(dataloader=trainloader, num_loops=opt.num_loops)
                console_logger.info(f'Phase time: {timer.values[-1]:.1f}')
            scheduler.step()

            # accumulate masks, because we want to test on all seen classes
            mask += phase_mask

            # this is the accuracy for all classes seen so far
            with timer:
                acc = trainer.test(loader=testloader, classnames=vd.class_names, mask=mask)
            cl_accuracy_meter.update(acc)
            console_logger.info(f'Test time: {timer.values[-1]:.1f}')

        console_logger.info(f'==> CL training completed! AverageAcc: [{cl_accuracy_meter.avg:.3f}]')


if __name__ == '__main__':
    cmd_opts = parse_args()

    wandb.init(project="experiment1", config=cmd_opts)

    exp1(cmd_opts)

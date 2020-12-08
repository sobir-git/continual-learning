import torch
import wandb
from torch import optim

import models.concrete.single
from dataloader import VisionDataset
from opts import parse_args
from training import Trainer
from utils import AverageMeter, get_logger, save_pretrained_model, load_pretrained_model, \
    get_default_device

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
    model = getattr(models.concrete, opt.model)(opt).to(device)
    if opt.watch:
        wandb.watch(model)
    wandb.run.summary['model_graph:'] = str(model)
    opt.exp_name += opt.model
    class_order = list(range(10))
    wandb.config.update({'class_order': class_order})
    vd = VisionDataset(opt, class_order=class_order)

    optimizer = create_optimizer(opt, model)
    scheduler = create_scheduler(opt, optimizer, opt.num_pretrain_passes)

    logger = get_logger(folder=opt.log_dir + '/' + opt.exp_name + '/')
    logger.info(f'Running with device {device}')
    logger.info("==> Opts for this training: " + str(opt))

    # pretraining
    if opt.num_pretrain_classes > 0:
        trainer = Trainer(opt, logger, device=device, type='pre')
        try:
            logger.info('Trying to load pretrained model...')
            model = load_pretrained_model(opt, model, logger)
            pretrain = False
        except Exception as e:
            logger.info(f'Failed to load pretrained model: {e}')
            pretrain = True

        if pretrain:
            assert opt.num_pretrain_passes > 0
            logger.info(f'==> Starting pretraining')
            for epoch in range(1, opt.num_pretrain_passes + 1):
                wandb.log({'learning_rate': scheduler.get_last_lr(), 'epoch': epoch})
                trainer.train(loader=vd.pretrain_loader, model=model, optimizer=optimizer, step=epoch)
                scheduler.step()
                acc = trainer.test(loader=vd.pretest_loader, model=model, mask=vd.pretrain_mask,
                                   classnames=vd.class_names, step=epoch)

            logger.info(f'==> Pretraining completed! Acc: [{acc:.3f}]')

            if opt.old_exp_name != '':
                save_pretrained_model(opt, model)

    if opt.num_tasks > 0:
        # TODO: use another optimizer?
        # Class-Incremental training
        # We start with pretrain mask bvecause in testing we want pretrained classes included
        trainer = Trainer(opt, logger, device=device, type='cl')
        logger.info(f'==> Starting Class-Incremental training')
        mask = vd.pretrain_mask.clone() if opt.num_pretrain_classes > 0 else torch.zeros(vd.n_classes_in_whole_dataset)
        dataloaders = vd.get_ci_dataloaders()
        cl_accuracy_meter = AverageMeter()
        if opt.refresh_scheduler:
            scheduler = create_scheduler(opt, optimizer, n_epochs=opt.num_tasks)

        for phase, (trainloader, testloader, class_list, phase_mask) in enumerate(dataloaders, start=1):
            logger.info(f'==> Phase {phase}: learning classes {", ".join(map(str, class_list))}')
            wandb.log({'learning_rate': scheduler.get_last_lr(), 'phase': phase})
            trainer.train(loader=trainloader, model=model, optimizer=optimizer, step=phase)
            scheduler.step()

            # accumulate masks, because we want to test on all seen classes
            mask += phase_mask

            # this is the accuracy for all classes seen so far
            acc = trainer.test(loader=testloader, model=model, mask=mask, classnames=vd.class_names, step=phase)
            cl_accuracy_meter.update(acc)
        logger.info(f'==> CL training completed! AverageAcc: [{cl_accuracy_meter.avg:.3f}]')


if __name__ == '__main__':
    cmd_opts = parse_args()

    wandb.init(project="experiment1", config=cmd_opts)

    exp1(cmd_opts)

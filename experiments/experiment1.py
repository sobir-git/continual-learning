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


def schedule_lr(opt, optimizer, scheduler, epoch):
    # Handle lr scheduling
    assert epoch >= 1, "Make sure you index epochs from 1"
    if epoch == 1:  # Warm start of 1 epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.maxlr * 0.1
    elif epoch == 2:  # Then set to maxlr
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.maxlr
    else:  # Aand go!
        scheduler.step()


def exp1(opt):
    model = getattr(models.concrete.single, opt.model)(opt).to(device)
    if opt.watch:
        wandb.watch(model)
    wandb.run.summary['model_graph:'] = str(model)
    opt.exp_name += opt.model
    class_order = list(range(10))
    wandb.config.update({'class_order': class_order})
    vd = VisionDataset(opt, class_order=class_order)

    optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr)

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
                schedule_lr(opt, optimizer, scheduler, epoch)
                wandb.log({'lr': scheduler.get_last_lr(), 'epoch': epoch})
                trainer.train(loader=vd.pretrain_loader, model=model, optimizer=optimizer, step=epoch)
                acc = trainer.test(loader=vd.pretest_loader, model=model, mask=vd.pretrain_mask, step=epoch)
            logger.info(f'==> Pretraining completed! Acc: [{acc:.3f}]')
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
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr)
            scheduler_prior_steps = 0
        else:
            scheduler_prior_steps = opt.num_pretrain_passes

        for phase, (trainloader, testloader, class_list, phase_mask) in enumerate(dataloaders, start=1):
            schedule_lr(opt, optimizer, scheduler, scheduler_prior_steps + phase)
            wandb.log({'lr': scheduler.get_last_lr(), 'phase': phase})
            trainer.train(loader=trainloader, model=model, optimizer=optimizer, step=phase)

            # accumulate masks, because we want to test on all seen classes
            mask += phase_mask

            # this is the accuracy for all classes seen so far
            acc = trainer.test(loader=testloader, model=model, mask=mask, step=phase)
            cl_accuracy_meter.update(acc)
        logger.info(f'==> CL training completed! AverageAcc: [{cl_accuracy_meter.avg:.3f}]')


if __name__ == '__main__':
    cmd_opts = parse_args()

    wandb.init(project="experiment1", config=cmd_opts)

    exp1(cmd_opts)

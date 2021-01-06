import math

from torch.optim import lr_scheduler as sch


class ExponentialLR(sch.ExponentialLR):
    def __init__(self, cfg, optimizer):
        ctrl_min_lr = cfg.ctrl_min_lr
        lr = cfg.ctrl_lr
        n_epochs = cfg.ctrl_epochs
        log_gamma = math.log(ctrl_min_lr / lr) * (1 / n_epochs)
        gamma = math.exp(log_gamma)
        super(ExponentialLR, self).__init__(optimizer, gamma)

    def step(self, *args):
        super(ExponentialLR, self).step()


def get_controller_lr_scheduler(cfg, optimizer):
    if cfg.ctrl_lr_scheduler == 'exp':
        return ExponentialLR(cfg, optimizer)


def get_classifier_lr_scheduler(cfg, optimizer):
    return sch.ReduceLROnPlateau(optimizer, 'min', factor=cfg.lr_decay, patience=cfg.lr_patience,
                                 verbose=True, min_lr=0.00001)

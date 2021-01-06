import math

from torch.optim import lr_scheduler as sch

from exp2.config import Config


class ExponentialLR(sch.ExponentialLR):
    def __init__(self, cfg, optimizer):
        log_gamma = math.log(cfg.min_lr / cfg.lr) * (1 / cfg.epochs)
        gamma = math.exp(log_gamma)
        super(ExponentialLR, self).__init__(optimizer, gamma)

    def step(self, *args):
        super(ExponentialLR, self).step()


def _get_lr_scheduler(config, optimizer):
    if config.lr_scheduler == 'exp':
        return ExponentialLR(config, optimizer)

    return sch.ReduceLROnPlateau(optimizer, 'min', factor=config.lr_decay, patience=config.lr_patience,
                                 verbose=True, min_lr=0.00001)


def get_controller_lr_scheduler(config: Config, optimizer):
    return _get_lr_scheduler(config.ctrl, optimizer)


def get_classifier_lr_scheduler(config: Config, optimizer):
    return _get_lr_scheduler(config.clf, optimizer)

import math

from torch.optim import lr_scheduler as sch

from exp2.config import Config


def compute_gamma(min, max, times):
    """Solve for max * x^times = min"""
    log_gamma = math.log(min / max) * (1 / times)
    gamma = math.exp(log_gamma)
    return gamma


class ExponentialLR(sch.ExponentialLR):
    def __init__(self, cfg, optimizer):
        gamma = compute_gamma(cfg.min_lr, cfg.lr, cfg.epochs - 1)
        super(ExponentialLR, self).__init__(optimizer, gamma)

    def step(self, *args):
        super(ExponentialLR, self).step()


class StepLR(sch.StepLR):
    n_parts = 4

    def __init__(self, config, optimizer):
        step_size = config.epochs // self.n_parts
        gamma = compute_gamma(config.min_lr, config.lr, self.n_parts - 1)
        super(StepLR, self).__init__(optimizer, step_size, gamma)

    def step(self, *args):
        super().step()


class ConstantLR(StepLR):
    n_parts = 0.5


mapping = {'exp': ExponentialLR,
           'step': StepLR,
           'const': ConstantLR}


def _get_lr_scheduler(config, optimizer):
    return mapping[config.lr_scheduler](config, optimizer)
    #
    # return sch.ReduceLROnPlateau(optimizer, 'min', factor=config.lr_decay, patience=config.lr_patience,
    #                              verbose=True, min_lr=0.00001)


def get_controller_lr_scheduler(config: Config, optimizer):
    return _get_lr_scheduler(config.ctrl, optimizer)


def get_classifier_lr_scheduler(config: Config, optimizer):
    return _get_lr_scheduler(config.clf, optimizer)

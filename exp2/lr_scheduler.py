import math

from torch.optim import lr_scheduler as sch


def compute_gamma(min, max, times):
    """Solve for max * x^times = min"""
    log_gamma = math.log(min / max) * (1 / times)
    gamma = math.exp(log_gamma)
    return gamma


class ExponentialLR(sch.ExponentialLR):
    def __init__(self, cfg, optimizer):
        super(ExponentialLR, self).__init__(optimizer, cfg['gamma'])

    def step(self, *args):
        super(ExponentialLR, self).step()


class StepLR(sch.StepLR):
    def __init__(self, config, optimizer):
        n_parts = int(config['lr_scheduler'][-1])
        step_size = math.ceil(config['epochs'] / n_parts)
        gamma = compute_gamma(config['min_lr'], config['lr'], n_parts - 1)
        super(StepLR, self).__init__(optimizer, step_size, gamma)

    def step(self, *args):
        super().step()


class ConstantLR(sch.LambdaLR):
    def __init__(self, config, optimizer):
        lr_lambda = lambda epoch: 1
        super().__init__(optimizer, lr_lambda)

    def step(self, *args):
        super(ConstantLR, self).step()


class PolynomialLR(sch.LambdaLR):
    def __init__(self, config, optimizer):
        power = int(config['lr_scheduler'][-1])
        epochs = config['epochs']
        min_lr = config['min_lr']
        base_lr = config['lr']
        lr_lambda = lambda ep: (min_lr + (base_lr - min_lr) * (1 - ep / (epochs - 1)) ** power) / base_lr
        super(PolynomialLR, self).__init__(optimizer, lr_lambda)

    def step(self, *args):
        super(PolynomialLR, self).step()


mapping = {'exp': ExponentialLR,
           'step2': StepLR,
           'step3': StepLR,
           'step4': StepLR,
           'step5': StepLR,
           'poly1': PolynomialLR,
           'poly2': PolynomialLR,
           'poly3': PolynomialLR,
           'poly4': PolynomialLR,
           'const': ConstantLR}


def _get_lr_scheduler(config, optimizer):
    return mapping[config['lr_scheduler']](config, optimizer)
    #
    # return sch.ReduceLROnPlateau(optimizer, 'min', factor=config.lr_decay, patience=config.lr_patience,
    #                              verbose=True, min_lr=0.00001)


def get_controller_lr_scheduler(config, optimizer):
    return _get_lr_scheduler(config.ctrl, optimizer)


def get_classifier_lr_scheduler(config, optimizer):
    return _get_lr_scheduler(config.clf, optimizer)

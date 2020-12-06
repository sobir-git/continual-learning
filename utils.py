class AverageMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0

    @property
    def avg(self):
        if self.n == 0:
            return -1
        return self.sum / self.n

    def update(self, value, n=1):
        self.sum = self.sum + value * n
        self.n += n

    def reset(self):
        self.n = 0
        self.sum = 0


from time import time


class Timer(object):
    """
    Timer context manager. Usage:
        timer = Timer()
        with timer:  # can do this many times
            ... do some code

        timer.total #  or timer.values
    """

    def start(self):
        self._start = time()
        assert not self._running, "Timer is already running"
        self._running = True

    def finish(self):
        self._values.append(time() - self._start)
        assert self._running, "Timer was not running"
        self._running = False
        self._start = -1000000

    def get_timed_callable(self, f):
        assert callable(f)
        def f_(*args, **kwargs):
            self.start()
            r = f(*args, **kwargs)
            self.finish()
            return r
        return f_

    def get_timed_generator(self, gen):
        """
        Pythonically, a generator x is used commonly in two scenarios:
        1. for i in x:  ...
        2. list(x)
        In both cases python does this:
        1. it = x.__iter__()
        2. yield next(it)  # until StopIteration raised

        So we want to patch the gen.__iter__()
        Args:
            gen: a generator object

        Returns:
            New generator object that records its execution times to this timer.
        """
        def new_gen():
            it = iter(gen)
            while True:
                try:
                    self.start()
                    y = next(it)
                    self.finish()
                    yield y
                except StopIteration:
                    self.discard()
                    return

        return new_gen()

    def discard(self):
        assert self._running
        self._running = False
        self._start = -1000000

    def __init__(self):
        self._values = []
        self._running = False

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.finish()

    @property
    def values(self):
        return self._values.copy()

    @property
    def total(self):
        return sum(self._values)


# ============================================================
'''
Sourced from https://github.com/drimpossible/GDumb, with modifications
'''

import random
import torch
import numpy as np
import os
import logging


def get_logger(folder):
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    os.makedirs(folder, exist_ok=True)
    fh = logging.FileHandler(os.path.join(folder, 'checkpoint.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_accuracy(y_prob, y_true, mask=None):
    '''
    Calculates the task and class incremental accuracy of the model
    '''
    y_pred = torch.argmax(y_prob, 1)
    # assert (y_prob.size() == mask.size()), "Class mask does not match probabilities in output"
    masked_prob = torch.mul(y_prob, mask)
    y_pred_masked = torch.argmax(masked_prob, 1)
    acc_masked = torch.eq(y_pred_masked, y_true)
    return (acc_masked * 1.0).mean()


def seed_everything(seed):
    '''
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    '''
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True # An exemption for speed :P


def save_pretrained_model(opt, model, name='pretrained_model'):
    '''
    Used for saving the pretrained model, not for intermediate breaks in running the code.
    '''
    state = {'opt': opt,
             'state_dict': model.state_dict()}
    folder = opt.log_dir + opt.old_exp_name
    filename = folder + f'/{name}.pth.tar'
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, filename)


def load_pretrained_model(opt, model, logger, name='pretrained_model'):
    '''
    Used for loading the pretrained model, not for intermediate breaks in running the code.
    '''
    filepath = opt.log_dir + opt.old_exp_name + f'/{name}.pth.tar'
    assert (os.path.isfile(filepath))
    logger.debug("=> loading checkpoint '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=get_default_device())
    model.load_state_dict(checkpoint['state_dict'])
    return model


def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert (alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    # if torch.cuda.is_available():
    #     index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

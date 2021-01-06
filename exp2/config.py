import argparse


class Config:
    def __init__(self, opt: argparse.Namespace, delim='_'):
        self.__opt = opt
        self.__delim = delim

    def __getattr__(self, item):
        return getattr(self.__opt, item)

    def _get_subkwargs(self, prefix):
        """Get all attributes that start with prefix"""
        preflen = len(prefix)
        kwargs = dict()
        for k, v in self.__opt.__dict__.items():
            if k.startswith(prefix):
                kwargs[k[preflen:]] = v
        return kwargs

    def _get_subconfig(self, prefix):
        kwargs = self._get_subkwargs(prefix + self.__delim)
        if not kwargs:
            raise AttributeError(f'No such subconfig with {prefix}')
        opt = argparse.Namespace(**kwargs)
        return Config(opt, delim=self.__delim)

    @property
    def clf(self):
        return self._get_subconfig('clf')

    @property
    def ctrl(self):
        return self._get_subconfig('ctrl')

    def __repr__(self):
        return argparse.Namespace.__repr__(self.__opt)



def parse_args(args=None) -> Config:
    lr_choices = ['exp', 'step2', 'step3', 'step4', 'step5', 'poly1', 'poly2', 'poly3', 'poly4', 'const']
    default_lr_scheduler = 'poly1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./logs/', help='Directory where all logs are stored')
    parser.add_argument('--datadir', type=str, default='../data/', help='Directory where all datasets are stored')
    parser.add_argument('--n_phases', type=int, required=True, help='Number of classes per phase')
    parser.add_argument('--n_classes_per_phase', type=int, required=True,
                        help='Seed to random shuffle class order. -1 for not shuffling')
    parser.add_argument('--other', action='store_true', help='Train classifiers with "other" category')
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'TinyCIFAR10'])
    parser.add_argument('--update_classifiers', action='store_true', help='Update previous classifiers')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=1000, help='Memory size')
    # parser.add_argument('--memory_sampler', type=str, default='greedy', help='Memory sampler', choices=['greedy', 'loss_aware'])
    parser.add_argument('--update_scores', action='store_true', help='Update memory scores at each phase.')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation size when training classifiers and controller')
    parser.add_argument('--class_order_seed', type=int, default=-1,
                        help='Seed to random shuffle class order. -1 for not shuffling')
    parser.add_argument('--download', action='store_true', help='Whether to download the dataset')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='The pretrained model file. This is used get feature extractor from and a blueprint for classifiers.')
    parser.add_argument('--split_pos', type=int, required=True,
                        help='Split position for pretrained model to get feature extractor and classifier part')
    parser.add_argument('--clone_head', action='store_true',
                        help='Whether to clone from the pretrained model when initializing classifiers')
    parser.add_argument('--ctrl_pos', type=str, required=True, choices=['before', 'after'],
                        help='The position of the controller. before or after the classifiers')
    parser.add_argument('--ctrl_hidden_layer_scale', type=float, default=0.,
                        help='Set the size of the hidden layer as output size scaled by this number. If 0, no hidden layers.')
    parser.add_argument('--ctrl_hidden_activation', type=str, default='Sigmoid',
                        help='Activation function of hidden layer in controller.')

    parser.add_argument('--ctrl_lr', type=float, default=0.01, help='Learning rate of controller')
    parser.add_argument('--ctrl_lr_scheduler', type=str, default=default_lr_scheduler, choices=lr_choices,
                        help='Learning rate scheduler for controller')
    parser.add_argument('--ctrl_min_lr', type=float, default=1e-5, help='Learning rate of controller')
    parser.add_argument('--ctrl_epochs', type=int, default=20, help='Number of training epochs of controller')
    parser.add_argument('--ctrl_epochs_tol', type=int, default=4,
                        help='Number of training epochs without improvement before stopping.')
    parser.add_argument('--clf_lr', type=float, default=0.01, help='Learning rate of classifiers')
    parser.add_argument('--clf_min_lr', type=float, default=1e-5, help='Learning rate of controller')
    parser.add_argument('--clf_lr_scheduler', type=str, default=default_lr_scheduler, choices=lr_choices,
                        help='Learning rate scheduler for classifier')
    parser.add_argument('--clf_epochs', type=int, default=50, help='Number of training epochs of a new classifier')
    parser.add_argument('--clf_epochs_tol', type=int, default=4,
                        help='Number of training epochs without improvement before stopping.')
    parser.add_argument('--clf_update_epochs', type=int, default=50,
                        help='Number of training epochs when updating classifiers')
    parser.add_argument('--clf_update_epochs_tol', type=int, default=4,
                        help='Number of training epochs without improvement before stopping.')
    parser.add_argument('--balance_other_samplesize', action='store_true',
                        help='Class balancing considering the trainset/otherset sample size. '
                             'This will increase weight of otherset by that ratio.')
    parser.add_argument('--balance_other_classsize', action='store_true',
                        help='Class balancing considering the otherset/trainset number of classes. '
                             'This will increase weight of otherset by that ratio.')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay ')
    parser.add_argument('--lr_patience', type=int, default=3, help='Patience epochs before decaying lr')
    parser.add_argument('--lr_decay', type=float, default=0.25, help='Learning rate decay factor')

    parser.add_argument('--wandb_group', type=str, help='W&B group in experiments')

    if args is not None:
        opt = parser.parse_args(args)
    else:
        opt = parser.parse_args()
    config = Config(opt)
    return config

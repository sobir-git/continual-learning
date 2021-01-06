'''
Sourced from https://github.com/drimpossible/GDumb, with modifications
'''
import torch, torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
import numpy as np

from utils import get_default_device

device = get_default_device()


class CIDataLoader:
    """
    Class-Incremental dataloader.
    Note: The dataloaders that it provides for a particular set of classes, always have shuffle=True;
    """

    def __init__(self, dataset: Dataset, class_to_idx_mapping, **kwargs):
        self.dataset = dataset
        self.class_to_idx_mapping = class_to_idx_mapping
        self.kwargs = kwargs.copy()
        del self.kwargs['shuffle']

    def from_classes(self, class_list):
        """Return a dataloader containing the given classes"""
        indices = []
        for cls in class_list:
            indices += self.class_to_idx_mapping[cls]
        sampler = SubsetRandomSampler(indices)
        return DataLoader(self.dataset, sampler=sampler, **self.kwargs)


# SEE THE SUMMARY AT THE END
class VisionDataset(object):
    """
    Code to load the dataloaders. Should be easily readable and extendable to any new dataset.
    Should generate class_mask, cltrain_loader, cltest_loader; with support for pretraining dataloaders given
    as pretrain_loader and pretest_loader.
    """

    def __init__(self, opt, class_order=None):
        self.kwargs = {
            'num_workers': opt.workers,
            'batch_size': opt.batchsize,
            'shuffle': True,
            'pin_memory': True}
        self.opt = opt

        # Sets parameters of the dataset. For adding new datasets, please add the dataset details in `get_statistics` function.
        mean, std, self.n_classes_in_whole_dataset, opt.inp_size, opt.in_channels = get_statistics(opt.dataset)
        opt.total_num_classes = opt.num_pretrain_classes + opt.num_tasks * opt.num_classes_per_task
        self._class_order = class_order if class_order else list(range(self.n_classes_in_whole_dataset))
        # if there is no class order specified, we randomly shuffle CL class list
        if class_order is None: random.shuffle(self._class_order)  # Generates different class-to-task assignment
        assert len(self._class_order) == self.n_classes_in_whole_dataset
        assert self.n_classes_in_whole_dataset >= opt.num_pretrain_classes + opt.num_tasks * opt.num_classes_per_task

        # Remap the class order to a 0-n order, required for cross-entropy loss using class list
        self.target_transform = ReorderTargets(self._class_order)

        # Generates the standard data augmentation transforms
        train_augment, test_augment = get_augment_transforms(dataset=opt.dataset, inp_sz=opt.inp_size)
        self.train_transforms = torchvision.transforms.Compose(
            train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        self.test_transforms = torchvision.transforms.Compose(
            test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

        self._gen_cl_mapping()

    @property
    def class_order(self):
        return self._class_order.copy()

    @property
    def class_names(self):
        """
        Class names in order of class_orders
        """
        if self.opt.dataset == 'CIFAR10':
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            r = [None] * len(classes)
            for i, c in enumerate(classes):
                r[self.target_transform(i)] = c
            return r
        else:
            return None

    @property
    def cl_class_list(self):
        return self._class_order[self.opt.num_pretrain_classes: self.opt.total_num_classes]

    @property
    def pretrain_class_list(self):
        return self._class_order[:self.opt.num_pretrain_classes]

    def _get_dataset(self, train: bool, target_transform):
        """Create the train or test with the optionally given target_transform."""
        transforms = self.train_transforms if train else self.test_transforms
        # Support for *some* pytorch default loaders is provided. Code is made such that adding new datasets is super easy, given they are in ImageFolder format.
        if self.opt.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'KMNIST', 'FashionMNIST']:
            dataset = getattr(torchvision.datasets, self.opt.dataset)(root=self.opt.data_dir, train=train,
                                                                      download=self.opt.download,
                                                                      transform=transforms,
                                                                      target_transform=target_transform)
        elif self.opt.dataset == 'SVHN':
            split = 'train' if train else 'test'
            dataset = getattr(torchvision.datasets, self.opt.dataset)(root=self.opt.data_dir, split=split,
                                                                      download=self.opt.download,
                                                                      transform=transforms,
                                                                      target_transform=target_transform)
        else:
            subfolder = 'train' if train else 'test'  # ImageNet 'val' is labled as 'test' here.
            dataset = torchvision.datasets.ImageFolder(self.opt.data_dir + '/' + self.opt.dataset + '/' + subfolder,
                                                       transform=transforms, target_transform=target_transform)
        return dataset

    def _get_loader(self, train, class_list, target_transform=None):
        '''Creates a dataloader that loads only the indices(if specified).
        '''

        target_transform = target_transform or self.target_transform
        dataset = self._get_dataset(train, target_transform)
        class_labels_dict = self._train_class_labels_dict if train else self._test_class_labels_dict
        return CIDataLoader(dataset, class_labels_dict, **self.kwargs).from_classes(class_list)

    def _gen_cl_mapping(self):
        # Get the label -> [idx] mapping dictionary
        _train_dataset = self._get_dataset(True, self.target_transform)
        _test_dataset = self._get_dataset(False, self.target_transform)
        if self.opt.dataset == 'SVHN':
            self._train_class_labels_dict = classwise_split(targets=_train_dataset.labels)
            self._test_class_labels_dict = classwise_split(targets=_test_dataset.labels)
        else:
            self._train_class_labels_dict = classwise_split(targets=_train_dataset.targets)
            self._test_class_labels_dict = classwise_split(targets=_test_dataset.targets)

        # if pretraining is specified for a number of classes
        if self.opt.num_pretrain_classes > 0:
            # create the loaders for pretraining by providing the indices, training transformations which consists of
            # augmentations and normalization  and target transform, which trainsforms the targets in some way to
            # range 0 -- (n-1);
            # notice that target transforms should be same for train and test
            self.pretrain_loader = self._get_loader(class_list=self.pretrain_class_list,
                                                    train=True)
            self.pretest_loader = self._get_loader(class_list=self.pretrain_class_list,
                                                   train=False)

            self.pretrain_mask = torch.cat([torch.ones(self.opt.num_pretrain_classes),
                                            torch.zeros(
                                                self.n_classes_in_whole_dataset - self.opt.num_pretrain_classes)])

    def get_ci_dataloaders(self):
        npc = self.opt.num_pretrain_classes
        ncpt = self.opt.num_classes_per_task
        nt = self.opt.num_tasks
        assert len(self.cl_class_list) == ncpt * nt, self.cl_class_list

        r = []
        acc = 0
        for i in range(nt):
            train_class_list = self.cl_class_list[acc:acc + ncpt]
            test_class_list = self._class_order[:npc + acc + ncpt]
            train_loader = self._get_loader(train=True, class_list=train_class_list,
                                            target_transform=self.target_transform)
            test_loader = self._get_loader(train=False, class_list=test_class_list,
                                           target_transform=self.target_transform)
            task_mask = torch.zeros(self.n_classes_in_whole_dataset)
            task_mask[npc + acc:npc + acc + ncpt] = torch.ones(ncpt, dtype=torch.double)
            r.append((train_loader, test_loader, train_class_list, task_mask))
            acc += ncpt
        return r


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class ReorderTargets(object):
    """
    Converts the class-orders to start -- start + (n-1) irrespective of order passed.
    """

    def __init__(self, class_order, start=0):
        self.start = start
        self._class_order = np.array(class_order)

    @property
    def class_order(self):
        return self._class_order.copy()

    def __call__(self, target):
        """
        Takes a target, returns the order of this target, i.e., some number in range start .. start + n-1
        """
        return np.where(self._class_order == target)[0][0] + self.start


def get_augment_transforms(dataset, inp_sz):
    """
    Returns appropriate augmentation given dataset size and name
    Arguments:
        dataset (str): dataset name
    """
    if inp_sz == 32 or inp_sz == 28 or inp_sz == 64:
        train_augment = [torchvision.transforms.RandomCrop(inp_sz, padding=4)]
        test_augment = []
    else:
        train_augment = [torchvision.transforms.RandomResizedCrop(inp_sz)]
        test_augment = [torchvision.transforms.Resize(inp_sz + 32), torchvision.transforms.CenterCrop(inp_sz)]

    if dataset not in ['MNIST', 'SVHN', 'KMNIST']:
        train_augment.append(torchvision.transforms.RandomHorizontalFlip())

    return train_augment, test_augment


def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        targets (sequence): a sequence of targets
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict:
            class_labels_dict[targets[idx]].append(idx)
        else:
            class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict


def get_statistics(dataset):
    '''
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    '''
    assert (dataset in ['MNIST', 'KMNIST', 'EMNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'CINIC10',
                        'ImageNet100', 'ImageNet', 'TinyImagenet', 'TinyCIFAR10'])
    mean = {
        'MNIST': (0.1307,),
        'KMNIST': (0.1307,),
        'EMNIST': (0.1307,),
        'FashionMNIST': (0.1307,),
        'SVHN': (0.4377, 0.4438, 0.4728),
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'TinyCIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'CINIC10': (0.47889522, 0.47227842, 0.43047404),
        'TinyImagenet': (0.4802, 0.4481, 0.3975),
        'ImageNet100': (0.485, 0.456, 0.406),
        'ImageNet': (0.485, 0.456, 0.406),
    }

    std = {
        'MNIST': (0.3081,),
        'KMNIST': (0.3081,),
        'EMNIST': (0.3081,),
        'FashionMNIST': (0.3081,),
        'SVHN': (0.1969, 0.1999, 0.1958),
        'CIFAR10': (0.2023, 0.1994, 0.2010),
        'TinyCIFAR10': (0.2023, 0.1994, 0.2010),
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'CINIC10': (0.24205776, 0.23828046, 0.25874835),
        'TinyImagenet': (0.2302, 0.2265, 0.2262),
        'ImageNet100': (0.229, 0.224, 0.225),
        'ImageNet': (0.229, 0.224, 0.225),
    }

    classes = {
        'MNIST': 10,
        'KMNIST': 10,
        'EMNIST': 49,
        'FashionMNIST': 10,
        'SVHN': 10,
        'CIFAR10': 10,
        'TinyCIFAR10': 10,
        'CIFAR100': 100,
        'CINIC10': 10,
        'TinyImagenet': 200,
        'ImageNet100': 100,
        'ImageNet': 1000,
    }

    in_channels = {
        'MNIST': 1,
        'KMNIST': 1,
        'EMNIST': 1,
        'FashionMNIST': 1,
        'SVHN': 3,
        'CIFAR10': 3,
        'TinyCIFAR10': 3,
        'CIFAR100': 3,
        'CINIC10': 3,
        'TinyImagenet': 3,
        'ImageNet100': 3,
        'ImageNet': 3,
    }

    inp_size = {
        'MNIST': 28,
        'KMNIST': 28,
        'EMNIST': 28,
        'FashionMNIST': 28,
        'SVHN': 32,
        'CIFAR10': 32,
        'TinyCIFAR10': 32,
        'CIFAR100': 32,
        'CINIC10': 32,
        'TinyImagenet': 64,
        'ImageNet100': 224,
        'ImageNet': 224,
    }
    return mean[dataset], std[dataset], classes[dataset], inp_size[dataset], in_channels[dataset]

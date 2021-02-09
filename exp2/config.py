import argparse
from collections import defaultdict
from typing import Sequence

import yaml

from logger import dict_deep_update, assure_subdict
from utils import get_console_logger

console_logger = get_console_logger(__name__)


def make_nested(d, subkeys: Sequence[str] = None, delim='_'):
    """Makes the dictionary nested (one level), for keys that start with one of the given subkeys.
    Returns a new dictionary."""
    subkeys = subkeys or []
    new_d = defaultdict(dict)
    delim_len = len(delim)
    for k, v in d.items():
        try:
            subkey = next((s for s in subkeys if k.startswith(s + delim)))
            new_d[subkey][k[len(subkey) + delim_len:]] = v
        except StopIteration:
            if isinstance(v, dict):
                new_d[k].update(v)
            else:
                new_d[k] = v
    return dict(new_d)


def load_configs(config_files, subkeys=None):
    """Load configs on top of each other, overriding one another.
    Also assure that the base config has every key that subsequent configs have.
    """
    config_dict = dict()
    for path in config_files:
        console_logger.info('Loading config %s', path)
        with open(path) as f:
            y = yaml.safe_load(f)
            y = make_nested(y, subkeys)
            if config_dict:
                assure_subdict(y, config_dict)
            dict_deep_update(config_dict, y)
    return config_dict


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--defaults', type=str, default='./config/config-defaults.yaml', help='Default config file')
    parser.add_argument('--configs', type=str, nargs='*', default='./config/config.yaml', help='Config file')
    parser.add_argument('--wandb_group', type=str, help='W&B group in experiments')
    parser.add_argument('--project', type=str, help='W&B project name')

    if args is not None:
        opt = parser.parse_args(args)
    else:
        opt = parser.parse_args()
    return opt

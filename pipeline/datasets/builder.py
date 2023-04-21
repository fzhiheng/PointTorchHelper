# -- coding: utf-8 --

import copy
from addict import Dict
from torch.utils.data import DataLoader
from ..utils.registry import Registry, build_from_cfg

# @Time : 2023/4/21 20:53
# @Author : Zhiheng Feng
# @File : builder.py
# @Software : PyCharm
# -- coding: utf-8 --


DATASETS = Registry('dataset')
DATALOADERS = Registry('dataloader')
COLLATE_FNS = Registry('collate_fn')
PIPELINES = Registry('pipeline')

DATALOADERS['DataLoader'] = DataLoader

def build_dataset(cfg: Dict, default_args: Dict = None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset, cfg: Dict):
    copy_cfg = copy.deepcopy(cfg)
    collate_fn = None
    if 'collate_fn' in cfg:
        # cfg['collate_fn']['datasets'] = dataset
        collate_fn_cfg = copy_cfg.pop('collate_fn')
        # collate_fn = build_from_cfg(collate_fn_cfg, COLLATE_FNS)
        collate_fn = COLLATE_FNS.get(collate_fn_cfg)

    copy_cfg['dataset'] = dataset
    copy_cfg['collate_fn'] = collate_fn
    dataloader = build_from_cfg(copy_cfg, DATALOADERS)
    return dataloader


def build_dataloader_from_cfg(cfg: Dict, default_args: Dict = None):
    dataset = build_dataset(cfg.dataset)
    dataloader = build_dataloader(dataset, cfg.dataloader)
    return dataloader




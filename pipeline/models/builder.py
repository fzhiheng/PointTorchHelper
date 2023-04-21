# -- coding: utf-8 --

from addict import Dict

from ..utils.registry import Registry, build_from_cfg

# @Time : 2023/4/21 21:16
# @Author : Zhiheng Feng
# @File : builder.py
# @Software : PyCharm

MODELS = Registry("models")

def build_model(cfg: Dict, default_args: Dict = None):
    model = build_from_cfg(cfg, MODELS, default_args)
    return model
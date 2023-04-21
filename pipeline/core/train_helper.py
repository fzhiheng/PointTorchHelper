# -- coding: utf-8 --
import os
import copy
import logging
from typing import List
from functools import singledispatchmethod

import torch
from addict import Dict

# @Time : 2023/4/21 21:11
# @Author : Zhiheng Feng
# @File : train_helper.py
# @Software : PyCharm
# -- coding: utf-8 --

"""
    TrainHelper 完成optimizer, scheduler, 模型加载、移动到GPU、保存等功能,
"""


# TODO  将所有参数作为接口
class TrainHelper(object):

    def __init__(self, model, helper_config: Dict, logger: logging.Logger):
        """

        Args:
            model:
            train_config: keys include 'devices', 'ckpt','optimizer' and 'scheduler'
            logger:
        """

        super().__init__()
        self.net = model
        self.logger = logger

        self.lr_cliping = helper_config.lr_cliping
        self.print_interval = helper_config.print_interval

        self.check_keys(helper_config)

        self.optimizer_config = helper_config.optimizer
        self.scheduler_config = helper_config.scheduler
        self.devices = self.get_devices_ids(helper_config.devices)
        self.logger.info('Number of {} parameters: {}'.format(self.net.__class__.__name__, self.para_nums))

        self.optimizer = self.creat_optimizer()
        self.scheduler = self.creat_scheduler()

        self.load_model(helper_config.ckpt)
        self.move_model(self.devices)

    def check_keys(self, config: Dict):
        target_keys = {'devices', 'ckpt', 'optimizer', 'scheduler'}
        src_keys = set(list(config.keys()))
        if target_keys > src_keys:
            raise KeyError(f'{target_keys - src_keys} should be in helper_config')

    def get_devices_ids(self, devices_config) -> List:
        """

        Args:
            devices_config: 传入的device配置参数,如['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

        Returns:
            将输入的字符串设备转换成torch.device的列表进行返回
            for example:输入为['cuda:0'] 返回为[device(type='cuda:0')]

        Raises:
            ValueError:输入的列表元素不是'cpu'或者'cuda:0

        """
        example = f"您输入的devices参数为{devices_config},请输入正确的devices配置参数如 ['cpu']或['cuda:0', cuda:1]"
        devices_config = list(devices_config)
        if (len(devices_config) == 0):
            raise ValueError(example)
        if not 'cuda' in devices_config[0] and not 'cpu' in devices_config[0]:
            raise ValueError(example)
        if not torch.cuda.is_available() or devices_config[0] == 'cpu':
            return [torch.device('cpu')]
        else:
            return [torch.device(x) for x in devices_config]

    def move_model(self, devices: list):
        """

        Args:
            devices: 传入一个torch.device的列表

        Returns:
            根据devices,将模型放到对应的设备上

        """
        if len(devices) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=devices)
        self.net.to(devices[0])
        if not self.optimizer is None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(devices[0])
        self.logger.info(f'{self.net.__class__.__name__} trained on : {devices}')

    def load_model(self, ckpt_path: str):
        """

        Args:
            ckpt_path: 预加载模型路径

        Returns:
            加载模型

        """
        self.global_state = {}
        if ckpt_path:
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.devices[0])
                self.logger.info('Loading pretrain model {}'.format(ckpt_path))
                if len(self.devices) > 1:
                    self.net.load_state_dict(self.add_prefix(checkpoint['model_state_dict']))
                else:
                    self.net.load_state_dict(self.strip_prefix(checkpoint['model_state_dict']))

                if 'optimizer' in checkpoint.keys() and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint.keys() and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.global_state = checkpoint.setdefault('global_state', {})
            else:
                self.logger.warning(f'{ckpt_path} is not is file, train from scratch')
        else:
            self.logger.warning('train from scratch')

    def set_net_param(self, param_dict):
        if len(self.devices) > 1:
            self.net.load_state_dict(self.add_prefix(param_dict))
        else:
            self.net.load_state_dict(self.strip_prefix(param_dict))

    def get_net_param(self):
        current_param = self.net.module.state_dict() if len(self.devices) > 1 else self.net.state_dict()
        return current_param

    def set_net_param_from_file(self, ckpt_path, key='model_state_dict'):
        checkpoint = torch.load(ckpt_path, map_location=self.devices[0])
        self.set_net_param(checkpoint[key])

    def creat_optimizer(self):
        """

        Returns: 创建一个优化器

        """
        if self.optimizer_config:
            optimizer_config = copy.deepcopy(self.optimizer_config)
            optimizer_type = optimizer_config.pop('type')
            optimizer = eval('torch.optim.{}'.format(optimizer_type))(self.net.parameters(), **optimizer_config)
            self.logger.info(f'the optimizer of your model {self.net.__class__.__name__} is {optimizer}!')
            return optimizer
        else:
            self.logger.warning(f'your model {self.net.__class__.__name__} has no optimizer!')
            return None

    def creat_scheduler(self):
        """

        Returns: 创建一个scheduler

        """
        if self.scheduler_config:
            scheduler_config = copy.deepcopy(self.scheduler_config)
            scheduler_type = scheduler_config.pop('type')
            scheduler = eval('torch.optim.lr_scheduler.{}'.format(scheduler_type))(self.optimizer, **scheduler_config)
            self.logger.info(f'the scheduler of your model {self.net.__class__.__name__} is {scheduler}!')
            return scheduler
        else:
            self.logger.warning(f'your model {self.net.__class__.__name__} has no scheduler!')
            return None

    @property
    def para_nums(self):
        """

        Returns:
            获取模型中参数的数量

        """
        return sum([p.data.nelement() for p in self.net.parameters()])

    def get_learing_rate(self) -> str:
        """

        Returns:
            返回优化器中各参数组的学习率

        """

        lr_str = 'param_group '
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr_str += f"{i}-th {param_group['lr']}"
        return lr_str

    @property
    def param_groups_lr(self) -> List[float]:
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def save_checkpoint(self, checkpoint_path, global_state=None, **kwargs):
        """

        Args:
            checkpoint_path: 模型保存路径
            **kwargs:

        Returns: 保存最新的模型和ckpt_save_type决定的模型

        """
        self.global_state = global_state
        save_state = {
            'model_state_dict': self.net.module.state_dict() if len(self.devices) > 1 else self.net.state_dict()
        }
        if self.optimizer:
            save_state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler:
            save_state['scheduler'] = self.scheduler.state_dict()
        if self.global_state:
            save_state['global_state'] = self.global_state
        save_state.update(kwargs)
        torch.save(save_state, checkpoint_path)
        self.logger.info(f'Save {self.net.__class__.__name__} to {checkpoint_path}')

    def strip_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        """

        Args:
            state_dict: 存储的模型字典
            prefix:

        Returns:去前缀后的模型字典,该方法用于将模型加载到单卡时的处理操作

        """
        for key in state_dict.keys():
            if not key.startswith(prefix):
                return state_dict
            else:
                break
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            stripped_state_dict[key.replace(prefix, '')] = state_dict.pop(key)
        return stripped_state_dict

    def add_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        """

        Args:
            state_dict: 存储的模型字典
            prefix: 前缀

        Returns:
            加上前缀后的模型字典,该方法用于将模型加载到多卡时的处理操作
        """

        # 考虑到模型都是统一保存的，因此只需要检查第一个就可以了
        for key in state_dict.keys():
            if key.startswith(prefix):
                return state_dict
            else:
                break
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            key2 = prefix + key
            stripped_state_dict[key2] = state_dict.pop(key)
        return stripped_state_dict

    def freeze_params(self, need_train_mods):
        all_modules = set([name for name, _ in self.net.named_children()])
        train_modules = set()
        no_train_module = set()
        for mod in need_train_mods:
            if mod.startswith('-'):
                no_train_module.add(mod[1:])
            elif mod == 'all':
                train_modules = all_modules
            else:
                train_modules.add(mod)
        if not train_modules <= all_modules:
            raise ValueError(f"The modules {train_modules} you want to train are not in {all_modules}")
        if not no_train_module < all_modules:
            raise ValueError(f"The modules {no_train_module} you do not want to train are not in {all_modules}")
        train_modules = train_modules - no_train_module
        for name, children in self.net.named_children():
            train_flag = name in train_modules
            for para in children.parameters():
                para.requires_grad = train_flag


    @singledispatchmethod
    def update_train_scheduler(self, current_flag):
        """ 支持分阶段训练功能，根据当前训练的epoch冻结或解冻网络中参数

        Args:
            current_flag: 当前训练的进展标志
            分阶段训练的依据是模型中设置的train_scheduler

        Returns: 冻结或解冻网络中参数

        Raises: 如果没有给模型设置train_scheduler，会抛出ValueError的错误

        Examples:
            train_scheduler={'a':["all"], 'b':["all", "-backbone"]}
            标志位'a'表示，只要传入的标志是'a',就将'a'对应的网络模块解冻，其中-号表示冻结
        """
        if self.train_scheduler is None:
            raise ValueError(f"The train_scheduler is None, if you want to stage the network training, you should set train_scheduler first!")
        need_train_mods = self.train_scheduler[current_flag]
        self.freeze_params(need_train_mods)


    @update_train_scheduler.register(numbers.Integral)
    def _(self, current_flag):
        if self.train_scheduler is None:
            raise ValueError(f"The train_scheduler is None, if you want to stage the network training, you should set train_scheduler first!")
        keys = list(self.train_scheduler.keys())
        for key in keys:
            if not isinstance(key, numbers.Integral):
                raise ValueError(
                    f"the current_flag is numbers.Number, but the keys of train_scheduler is {type(key)}")

        keys = sorted(keys)
        index = bisect.bisect_right(keys, current_flag) - 1
        need_train_mods = self.train_scheduler[keys[index]]
        self.freeze_params(need_train_mods)

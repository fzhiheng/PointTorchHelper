# -- coding: utf-8 --
import collections

from ..builder import PIPELINES
from ...utils.registry import build_from_cfg
# @Time : 2023/4/21 21:19
# @Author : Zhiheng Feng
# @File : compose.py
# @Software : PyCharm
# Copyright (c) OpenMMLab. All rights reserved.


@PIPELINES.register
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, processes):
        if not isinstance(processes, collections.abc.Sequence):
            raise TypeError(f"processes must be an collections.abc.Sequence,but get {type(processes)}")
        self.transforms = []
        for process in processes:
            if isinstance(process, dict):
                process = build_from_cfg(process, PIPELINES)
                self.transforms.append(process)
            elif callable(process):
                self.transforms.append(process)
            else:
                raise TypeError('transform must be callable or a dict')


    def __call__(self, data:dict)->dict:
        """Call function to apply process sequentially.

        Args:
            data (dict): A result dict contains the data to process.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string

    # def _init_pre_processes(self, pre_processes):
    #     self.aug = []
    #     if pre_processes is not None:
    #         for aug in pre_processes:
    #             if 'args' not in aug:
    #                 args = {}
    #             else:
    #                 args = aug['args']
    #             if isinstance(args, dict):
    #                 cls = eval(aug['type'])(**args)
    #             else:
    #                 cls = eval(aug['type'])(args)
    #             self.aug.append(cls)

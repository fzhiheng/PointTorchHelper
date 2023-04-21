# -- coding: utf-8 --

# @Time : 2022/10/12 22:21
# @Author : Zhiheng Feng
# @File : time_utils.py
# @Software : PyCharm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        return self

    def add(self, val):
        self.sum += val

    @property
    def avg(self):
        return 0 if self.count == 0  else self.sum / self.count


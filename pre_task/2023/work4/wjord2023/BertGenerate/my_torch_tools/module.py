import inspect
from torch import nn
import collections
from matplotlib import pyplot as plt
from IPython import display
import numpy
import torch


class HyperParameters:
    """超参数基类"""

    # 写了两个同名方法，是为了提醒子类可以重写这个方法
    def save_hyperparameters(self, ignore=[]):  # type: ignore
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """保存函数参数为类属性, 忽略ignore中的参数, 忽略私有属性, 忽略self属性,
        可以用于__init__方法中保存函数参数为类属性"""
        # 获取调用方法的上下文帧
        frame = inspect.currentframe().f_back  # type: ignore
        # 从上下文帧中获取函数参数
        _, _, _, local_vars = inspect.getargvalues(frame)  # type: ignore
        # 保存函数参数为类属性
        # 忽略ignore中的参数, 忽略私有属性, 忽略self属性
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        # 保存函数参数为类属性
        for k, v in self.hparams.items():
            setattr(self, k, v)


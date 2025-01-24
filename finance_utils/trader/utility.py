#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sy,Sang"
__version__ = ""
__license__ = "GPLv3"
__maintainer__ = "Sy, Sang"
__email__ = "martin9le@163.com"
__status__ = "Development"
__credits__ = []
__date__ = ""
__copyright__ = ""

# 系统模块
import copy
import pickle
import json
from typing import Union, Self
from collections import namedtuple
from abc import ABC, abstractmethod

# 项目模块

# 外部模块
import numpy


# 代码块

class UtilityFunction:
    """效用函数"""

    @abstractmethod
    def __call__(self, x: float, *args, **kwargs):
        pass


class LinearUtilityFunction:
    """线性效用曲线"""

    def __call__(self, x: float, *args, **kwargs):
        return x


class LogUtilityFunction(UtilityFunction):
    """log效用函数"""

    def __init__(self, base: float = None):
        super().__init__()
        self.base = base

    def __call__(self, x: float, *args, **kwargs):
        if self.base is None:
            return numpy.log(x)
        else:
            return numpy.log(x) / numpy.log(self.base)


class PowerUtilityFunction(UtilityFunction, ):

    def __init__(self, exp=None):
        super().__init__()
        self.exp = exp

    def __call__(self, x: float, *args, **kwargs):
        if self.exp is None:
            return x ** 2
        else:
            return x ** self.exp


class TimeLossUtilityFunction(UtilityFunction):
    """损失时间效用函数"""

    def __init__(self, risk_free_rate: float = 1.03):
        self.risk_free_rate = risk_free_rate

    def __call__(self, x: float, *args, **kwargs):
        return -1 * numpy.log(self.risk_free_rate / x) / numpy.log(self.risk_free_rate)


if __name__ == "__main__":
    u = TimeLossUtilityFunction(1.03)
    print([u(i) for i in numpy.arange(0.1, 2, 0.1)])

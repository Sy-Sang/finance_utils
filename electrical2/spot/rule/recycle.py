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

class Recycle(ABC):
    """回收机制(接口)"""

    def __init__(self, spot_list, submit_list, benefits):
        self.spot_list = spot_list
        self.submit_list = submit_list
        self.benefits = benefits

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


class AnarchismRecycle(Recycle):
    def __call__(self, *args, **kwargs):
        return 0


class SampleRecycle(Recycle):
    def __call__(self, trigger_rate=0.05, punishment_rate: float = 0.5, *args, **kwargs):
        punishment = 0
        spot_array = numpy.array(self.spot_list).astype(float)
        power_deviation = abs(numpy.sum(self.submit_list) / numpy.sum(spot_array[:, 2]) - 1)
        if power_deviation >= trigger_rate:
            punishment = max(0, self.benefits * punishment_rate)
        else:
            pass
        return punishment


if __name__ == "__main__":
    pass

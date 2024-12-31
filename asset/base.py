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
from enum import Enum

# 项目模块

# 外部模块
import numpy


# 代码块
class Asset(ABC):
    def clone(self):
        """克隆"""
        return copy.deepcopy(self)

    @abstractmethod
    def settlement(self, *args, **kwargs) -> float:
        """结算现金流"""
        pass

    @abstractmethod
    def payoff(self, *args, **kwargs) -> float:
        """损益"""
        pass


if __name__ == "__main__":
    pass

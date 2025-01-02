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
from functools import wraps

# 项目模块

# 外部模块
import numpy


# 代码块

class PositionType(Enum):
    """头寸类型"""
    long = 1
    short = -1


class LoanRole(Enum):
    """角色"""
    borrower = 1
    lender = -1


class OptionType(Enum):
    """期权类型"""
    call = 1
    put = -1


class Asset(ABC):
    """金融资产"""

    def __init__(self, name: Union[str, int], shares: Union[int, float]):
        self.name = name
        self.shares = shares

    def clone(self) -> Self:
        """克隆"""
        return copy.deepcopy(self)

    @staticmethod
    def with_shares(func):
        """装饰器，自动将返回值乘以 shares"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result * self.shares

        return wrapper

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def purchase_cost(self, position: PositionType, *args, **kwargs):
        """购入成本"""
        pass

    @abstractmethod
    def payoff(self, position: PositionType, *args, **kwargs) -> float:
        """损益曲线"""
        pass

    @abstractmethod
    def max_purchaseable(self, *args, **kwargs) -> Self:
        """最大购买量"""
        pass


if __name__ == "__main__":
    pass

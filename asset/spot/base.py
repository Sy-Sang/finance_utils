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
from enum import Enum

# 项目模块
from finance_utils.asset.base import Asset

# 外部模块
import numpy


# 代码块

class Position(Enum):
    """头寸"""
    long = 1
    short = -1


class BaseSpot(Asset):
    """现货"""

    def __init__(self, price: float, position: Position = Position.long, shares: float = 1):
        self.price = price
        self.position = position
        self.shares = shares

    def settlement(self, *args, **kwargs) -> float:
        return (self.price * self.shares) * (-1 * self.position.value)

    def payoff(self, x: float, *args, **kwargs) -> float:
        """损益"""
        return (x - self.price) * self.position.value * self.shares

    def value(self, x, *args, **kwargs):
        """资产价值"""
        return x * self.shares * self.position.value


if __name__ == "__main__":
    print([BaseSpot(100, shares=i).settlement() for i in range(100)])

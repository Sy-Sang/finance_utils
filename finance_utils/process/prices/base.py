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
from finance_utils.trader.base import Trader
from finance_utils.asset.base import Asset

# 外部模块
import numpy


# 代码块

class PriceProcess:

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def get_price(self, *args, **kwargs):
        """获取价格"""
        pass


class PriceProcessGroup:
    def __init__(self, pp: PriceProcess, ba: Asset, bt: Trader, num: int):
        plist = []
        alist = []
        tlist = []
        for i in range(num):
            pass


if __name__ == "__main__":
    pass

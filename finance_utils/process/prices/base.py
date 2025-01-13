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
from typing import Union, Self, List
from collections import namedtuple
from abc import ABC, abstractmethod

# 项目模块
from finance_utils.trader.base import Trader
from finance_utils.asset.base import Asset

# 外部模块
import numpy

# 代码块

PricePathValue = namedtuple("PathValue", ["timestamp", "price", "dic"])


class PriceProcess:
    """价格过程"""

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def get_price(self, *args, **kwargs) -> PricePathValue:
        """获取价格"""
        pass

    @abstractmethod
    def multi_pathing(self, *args, **kwargs):
        """多条路径"""
        pass


class MultiPathing:
    """多条路径"""

    def __init__(self, timeline: List, processes: List[PriceProcess], traders: List[Trader]):
        self.timeline = timeline
        self.processes = processes
        self.trades = traders
        self.size = (
            len(self.timeline),
            len(self.processes)
        )


if __name__ == "__main__":
    pass

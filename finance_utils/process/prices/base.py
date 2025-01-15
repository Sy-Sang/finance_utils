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
from finance_utils.types import *
from finance_utils.namedtuples import *
from finance_utils.trader.base import Trader
from finance_utils.asset.base import Asset

# 外部模块
import numpy


# 代码块


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

    @classmethod
    def sharp(cls, yield_list, risk_free_rate: float):
        """夏普比率"""
        return (yield_list[-1] - risk_free_rate) / numpy.std(yield_list, ddof=1)


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
        self.constructor = {
            "timeline": self.timeline,
            "processes": self.processes,
            "traders": self.trades
        }

    def clone(self):
        return type(self)(**self.constructor)

    def clone_with_new_trader(self, base_trader: Trader):
        """变更交易员"""
        trader_list = []
        for i in range(self.size[1]):
            trader_list.append(
                base_trader.clone(f"{base_trader.name}_copy_{i}")
            )
        constructor = copy.deepcopy(self.constructor)
        constructor["traders"] = trader_list
        return type(self)(**constructor)


if __name__ == "__main__":
    pass

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
from typing import Union, Self, List, Tuple
from collections import namedtuple
from abc import ABC, abstractmethod
from collections.abc import Iterator

# 项目模块
from finance_utils.uniontypes import *
from finance_utils.namedtuples import *
from finance_utils.trader.base import Trader
from finance_utils.asset.base import Asset

# 外部模块
import numpy


# 代码块


class PriceProcess(ABC):
    """价格过程"""

    def __init__(self, s0, *args, **kwargs):
        self.s0 = s0
        self.timeline: list[TimeStamp] = []

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __getitem__(self, item) -> PricePathValue:
        pass

    @abstractmethod
    def get_price(self, timestamp: TimeStr, *args, **kwargs) -> PricePathValue:
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

    def process_iter(self):
        """过程迭代器"""
        for i, t in enumerate(self.timeline):
            path_data = self[i]
            yield i, t, path_data


class MultiPathing:
    """多条路径"""

    def __init__(self, timeline: list[TimeStamp], processes: List[PriceProcess], traders: List[Trader]):
        self.timeline = timeline
        self.processes = processes
        self.trades = traders
        self.size = (
            len(self.timeline),
            len(self.processes)
        )
        self.length, self.width = self.size
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

    def clear(self):
        for j in range(self.width):
            self.trades[j].clear()

    def process_iter(self):
        for i, t in enumerate(self.timeline):
            dic = {}
            paths = []
            for j in range(self.width):
                path = self.processes[j][i]
                paths.append(path)
                root = list(path.dic)[0]
                dic[root] = path.dic[root]
            yield i, t, paths


if __name__ == "__main__":
    pass

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
from finance_utils.asset.base import *

# 外部模块
import numpy

# 代码块

InvestmentRec = namedtuple("InvestmentRec", ["timestamp", "investment"])


class TradeBookUnit(ABC):
    """交易记录单元"""

    def __init__(self, timestamp: TimeStr):
        self.timestamp = TimeStamp(timestamp)


class TradeBook(ABC):
    """交易记录"""

    def __init__(self):
        self.book: list[TradeBookUnit] = []

    def clone(self):
        """克隆"""
        return copy.deepcopy(self)

    def clear(self):
        """清空book"""
        self.book: list[TradeBookUnit] = []

    def sort(self):
        """对订单记录进行排序"""
        time_array = numpy.array([i.timestamp.timestamp() for i in self.book])
        is_sorted = numpy.all(time_array[:-1] <= time_array[1:])
        if is_sorted:
            pass
        else:
            sort_index = numpy.argsort(time_array)
            sorted_book = []
            for i in sort_index:
                sorted_book.append(self.book[i])
            self.book = sorted_book

    @abstractmethod
    def append(self, *args, **kwargs):
        """添加交易记录"""
        pass

    def timestamp_domain(self) -> tuple[TimeStamp, TimeStamp]:
        """订单表时间区间"""
        return self.book[0].timestamp, self.book[-1].timestamp

    @abstractmethod
    def long_quantity(self, *args, **kwargs) -> float:
        """多头持仓"""
        pass

    @abstractmethod
    def short_quantity(self, *args, **kwargs) -> float:
        """空头持仓"""
        pass

    @abstractmethod
    def in_position_quantity(self, *args, **kwargs) -> float:
        """持仓量"""
        pass

    @abstractmethod
    def holding_cost(self, *args, **kwargs) -> float:
        """持仓成本"""
        pass

    @abstractmethod
    def value(self, *args, **kwargs) -> float:
        """持仓价值"""
        pass

    @abstractmethod
    def payoff(self, *args, **kwargs) -> float:
        """持仓损益"""
        pass

    @abstractmethod
    def simplify(self, *args, **kwargs):
        """重整交易记录"""
        pass


class Trader:
    """交易员"""

    def __init__(self, name: str, investment: float, initial_timestamp: TimeStr):
        self.name = name
        self.investment_flow: list[InvestmentRec] = [InvestmentRec(TimeStamp(initial_timestamp), investment)]
        self.capital = investment
        self.position: dict[str, TradeBook] = {}
        self.constructor = {
            "name": name,
            "investment": investment,
            "initial_timestamp": initial_timestamp
        }

    def __repr__(self):
        return str([self.name, self.capital, self.position.keys()])

    def in_position(self, name: str):
        """是否在仓位中"""
        if name in self.position.keys():
            return self.position[name].in_position_quantity()
        else:
            return False

    def clone(self, new_name: str = None):
        """拷贝新trader"""
        constructor = copy.deepcopy(self.constructor)
        if new_name is None:
            pass
        else:
            constructor["name"] = new_name
        return type(self)(**self.constructor)

    def clear(self):
        self.position: dict[str, TradeBook] = {}

    def new_investment(self, investment: float, timestamp: TimeStr):
        """增加资金"""
        self.capital += investment
        self.investment_flow.append(InvestmentRec(TimeStamp(timestamp), investment))

    def value(self, **kwargs):
        """总价值"""
        pv = 0
        for k, v in kwargs.items():
            if k in self.position.keys():
                pv += self.position[k].value(**v)
        return pv + self.capital

    def total_investment(self):
        """总投资"""
        return sum([i.investment for i in self.investment_flow])

    def net_worth_rate(self, **kwargs):
        """交易净值"""
        value = self.value(**kwargs)
        return value / self.total_investment()

    def payoff(self, **kwargs):
        """损益函数"""
        pv = 0
        for k, v in kwargs.items():
            if k in self.position.keys():
                pv += self.position[k].payoff(**v)
        return pv + self.capital


if __name__ == "__main__":
    pass

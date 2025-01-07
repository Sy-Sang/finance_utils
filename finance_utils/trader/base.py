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


class TradeBook(ABC):
    """交易记录"""

    @abstractmethod
    def append(self, *args, **kwargs):
        """添加交易记录"""
        pass

    @abstractmethod
    def in_position_quantity(self, *args, **kwargs) -> float:
        """持仓量"""
        pass

    @abstractmethod
    def value(self, *args, **kwargs) -> float:
        """持仓价值"""
        pass

    @abstractmethod
    def payoff(self, *args, **kwargs) -> float:
        """持仓损益"""
        pass


class Trader:
    """交易员"""

    def __init__(self, investment: float, initial_timestamp: TimeStr):
        self.investment_flow: list[InvestmentRec] = [InvestmentRec(TimeStamp(initial_timestamp), investment)]
        self.capital = investment
        self.position: dict[str, TradeBook] = {}

    def __repr__(self):
        return str([self.investment_flow, self.capital, self.position])

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

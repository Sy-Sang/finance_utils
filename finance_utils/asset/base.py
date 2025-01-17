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
from typing import Any, Union, Self
from collections import namedtuple
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps

# 项目模块
from easy_datetime.timestamp import TimeStamp
from finance_utils.uniontypes import *
from finance_utils.namedtuples import *

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

    def __init__(self, name: Any, lot_size: Rational, trade_delta: TradeDelta, *args, **kwargs):
        self.name = str(name)
        self.lot_size = float(lot_size)
        self.trade_delta = trade_delta

    def clone(self, *args, **kwargs) -> Self:
        """克隆"""
        return copy.deepcopy(self)

    @abstractmethod
    def __repr__(self):
        pass

    def max_purchase_quantity(self, price: Rational, capital: Rational, *args, **kwargs):
        """最大购买量"""
        if self.lot_size is None:
            shares = capital / price
        else:
            shares = (capital // (price * self.lot_size)) * self.lot_size
        return shares

    @abstractmethod
    def purchased_to(self, *args, **kwargs):
        """买入"""
        pass

    @abstractmethod
    def sold_to(self, *args, **kwargs):
        """卖出"""
        pass

    @classmethod
    def untradable(cls, timestamp: TimeStr, trade_delta: TradeDelta):
        """在timestamp时刻无法交易的时间段"""
        ts = TimeStamp(timestamp)
        for i in range(trade_delta.delta):
            ts = ts.accurate_to(trade_delta.temporal) - [trade_delta.temporal, i]
        return ts


if __name__ == "__main__":
    print(TimeStamp.now())
    print(
        Asset.untradable(TimeStamp.now(), TradeDelta("min", 1))
    )

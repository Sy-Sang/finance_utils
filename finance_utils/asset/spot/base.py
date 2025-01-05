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
from typing import Union, Self, TypeAlias
from collections import namedtuple
from enum import Enum

# 项目模块
from finance_utils.asset.base import *

# 外部模块
import numpy

# 代码块


SpotElement = namedtuple("SpotElement", ["name", "lot_size"])
TradeBookUnit = namedtuple("TradeBookUnit", ["timestamp", "price", "shares", "position"])


class Spot(Asset):
    """现货"""

    def __init__(
            self,
            name,
            lot_size: RealNum = None
    ):
        super().__init__(name, lot_size)
        self.elements = SpotElement(self.name, self.lot_size)
        self.trade_book = []

    def __repr__(self):
        return str(self.elements)

    @staticmethod
    def payoff(price: RealNum, x: RealNum, position: PositionType, *args, **kwargs):
        """损益"""
        return (x - price) * position.value


class SpotTradeBook:
    """现货交易记录"""

    def __init__(self, asset: Spot):
        self.asset = asset.clone()
        self.book: list[TradeBookUnit] = []

    def __repr__(self):
        return f"{self.asset}:{self.book}"

    def __bool__(self):
        if self.book:
            return True
        else:
            return False

    def append(self, timestamp: TimeStr, price: RealNum, shares: RealNum, position: PositionType):
        self.book.append(
            TradeBookUnit(TimeStamp(timestamp), price, shares, position)
        )

    def long_quantity(self):
        return sum([i.shares for i in self.book if i.position == PositionType.long])

    def short_quantity(self):
        return sum([i.shares for i in self.book if i.position == PositionType.short])

    def holding_cost(self):
        long_array = numpy.array([
            [i.price, i.shares] for i in self.book if i.position == PositionType.long
        ]).astype(float)
        short_book = numpy.array([
            [i.price, i.shares] for i in self.book if i.position == PositionType.short
        ]).astype(float)
        amount = numpy.sum(long_array[:, 0] * long_array[:, 1]) - numpy.sum(short_book[:, 0] * short_book[:, 1])
        quantity = numpy.sum(long_array[:, 1]) - numpy.sum(short_book[:, 1])
        return amount / quantity


if __name__ == "__main__":
    pass

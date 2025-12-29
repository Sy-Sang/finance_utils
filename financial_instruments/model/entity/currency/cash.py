#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""人民币"""

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
from typing import *
from collections import namedtuple

# 项目模块
from financial_instruments.model.fm import FinancialModule, ReactionEvent

# 外部模块
import numpy


# 代码块


class BasicCash(FinancialModule):
    def __init__(self, symbol, amount):
        super().__init__("cash")
        self.symbol = symbol
        self.amount = float(amount)

    def __repr__(self):
        return f"{type(self).__name__}:{{symbol={self.symbol}; amount={self.amount}}}"

    def __float__(self):
        return self.amount

    def __neg__(self):
        return BasicCash(self.symbol, -self.amount)

    def __abs__(self):
        return BasicCash(self.symbol, abs(self.amount))

    def _coerce(self, other):
        if isinstance(other, BasicCash):
            if self.symbol != other.symbol:
                raise TypeError(
                    f"Cannot operate on different currencies: "
                    f"{self.symbol} vs {other.symbol}"
                )
            return other.amount

        elif isinstance(other, (int, float)):
            return float(other)

        else:
            return NotImplemented

    def __add__(self, other: Any) -> "BasicCash":
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return BasicCash(self.symbol, self.amount + v)

    def __radd__(self, other: Any):
        return self.__add__(other)

    def __sub__(self, other: Any):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return BasicCash(self.symbol, self.amount - v)

    def __rsub__(self, other: Any):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return BasicCash(self.symbol, v - self.amount)

    def __mul__(self, other):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return BasicCash(self.symbol, self.amount * v)

    def __truediv__(self, other):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return BasicCash(self.symbol, self.amount / v)

    def exchange(self, target_symbol, exchange_rate):
        return BasicCash(target_symbol, self.amount * exchange_rate)


if __name__ == "__main__":
    cny = BasicCash('CNY', 100)

    print((cny + 100).exchange('USD', 1 / 7))
    print((cny + 100) * 2)
    print((cny + 100).collect_trade())

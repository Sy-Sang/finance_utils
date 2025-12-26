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
from typing import Union, Self
from collections import namedtuple

# 项目模块
from financial_instruments.entity.financial_object import Leg

# 外部模块
import numpy


# 代码块


class MoneyLeg(Leg):
    def __init__(self, code, amount):
        super().__init__()
        self.code = code
        self.amount = float(amount)

    def __repr__(self):
        return f"{self.code}:({self.amount})"

    def __float__(self):
        return self.amount

    def __neg__(self):
        return MoneyLeg(self.code, -self.amount)

    def __abs__(self):
        return MoneyLeg(self.code, abs(self.amount))

    def _coerce(self, other):
        if isinstance(other, MoneyLeg):
            if self.code != other.code:
                raise TypeError(
                    f"Cannot operate on different currencies: "
                    f"{self.code} vs {other.code}"
                )
            return other.amount

        elif isinstance(other, (int, float)):
            return float(other)

        else:
            return NotImplemented

    def __add__(self, other):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return MoneyLeg(self.code, self.amount + v)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return MoneyLeg(self.code, self.amount - v)

    def __rsub__(self, other):
        v = self._coerce(other)
        if v is NotImplemented:
            return NotImplemented
        return MoneyLeg(self.code, v - self.amount)

    def exchange(self, target_code, exchange_rate):
        return MoneyLeg(target_code, self.amount * exchange_rate)

    def forward(self, **kwargs):
        return None


if __name__ == "__main__":
    cny = MoneyLeg('CNY', 100)

    print((cny + 100).exchange('USD', 7))

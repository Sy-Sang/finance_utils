#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""债券"""

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
from easy_datetime.timestamp import TimeStamp
from financial_instruments.entity.financial_object import Leg, FinancialObject, FinancialInstance
from financial_instruments.entity.currency.cash import MoneyLeg

# 外部模块
import numpy


# 代码块

class BasicBondLeg(Leg):
    def __init__(
            self,
            code: str,
            face_amount: MoneyLeg,
            price: MoneyLeg,
            maturity_date: Any,
    ):
        super().__init__()
        self.code = code
        self.face_amount = face_amount
        self.price = price

        self.maturity_date = TimeStamp(maturity_date)

    def __repr__(self):
        return (
            f"BasicBondLeg("
            f"code={self.code}, "
            f"face={self.face_amount}, "
            f"price={self.price}, "
            f"maturity={self.maturity_date})"
        )

    def forward(self, **kwargs):
        if 'market_date' in kwargs:
            if TimeStamp(kwargs['market_date']) == self.maturity_date:
                return self.face_amount
            else:
                return 0
        else:
            raise TypeError(
                f"Cannot find market_date in {kwargs}"
            )


class BasicBondObject(FinancialObject):
    def __init__(self):
        super().__init__()

    def assemble(
            self,
            code: str,
            face_amount: MoneyLeg,
            price: MoneyLeg,
            maturity: Any,
    ) -> Iterable[Leg]:
        return [BasicBondLeg(code, face_amount, price, maturity)]


if __name__ == "__main__":
    mymoney = MoneyLeg("CNY", 100)
    bound1 = FinancialInstance(None, BasicBondObject(), "bound", MoneyLeg("CNY", 100), MoneyLeg("USD", 10),
                               '2025-12-31')
    bound2 = FinancialInstance(None, BasicBondObject(), "bound", MoneyLeg("CNY", 100), MoneyLeg("CNY", 50),
                               '2025-12-31')
    bound = FinancialInstance([mymoney, bound1, bound2])
    print(bound.forward(market_date='2025-12-31'))
    # print(bound.interest_rate('2025-1-1', 7))

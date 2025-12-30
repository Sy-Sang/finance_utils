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
from financial_instruments.mop.fm import *
from financial_instruments.model.currency.cash import BasicCash
from financial_instruments.model.payment.commonly import CashPay

# 外部模块
import numpy


# 代码块

class BasicBoundBid(Trade):
    def __init__(self, bound: "BasicBound"):
        super().__init__()
        self.bound = bound.copy()

    def __call__(self, *args, **kwargs):
        if 'market_date' in kwargs.keys():
            market_date = kwargs['market_date']
            if TimeStamp(market_date) < self.bound.maturity_date:
                if 'quantity' in kwargs.keys():
                    quantity = kwargs['quantity']
                    cost = float(self.bound.price) * quantity
                elif 'amount' in kwargs.keys():
                    amount = kwargs['amount']
                    quantity = amount // float(self.bound.price)
                    cost = float(self.bound.price) * quantity
                else:
                    cost = 0
            else:
                cost = 0
        else:
            cost = 0

        return [{
            "instance": {
                BasicCash(self.bound.price.symbol, cost)
            }
        }]


class BasicBoundBuy(Trade):
    def __init__(self, bound: "BasicBound"):
        super().__init__()
        self.bound = bound.copy()

    def __call__(self, *args, **kwargs):
        if 'market_date' in kwargs.keys() and 'quantity' in kwargs.keys():
            market_date = kwargs['market_date']
            if TimeStamp(market_date) < self.bound.maturity_date:
                quantity = kwargs['quantity']
                cost = float(self.bound.price) * quantity
            else:
                cost = 0
                quantity = 0
        else:
            cost = 0
            quantity = 0

        return [
            {
                "instance": self.bound,
                "attr": {
                    "quantity": quantity
                }
            },
            {
                "instance": CashPay(self.bound.price.symbol, cost),
                "attr": {}
            }
        ]


class BasicBound(FinancialModule):
    def __init__(
            self,
            face_amount: BasicCash,
            price: BasicCash,
            quantity: int,
            maturity_date: Any,
    ):
        super().__init__()
        self.face_amount = face_amount
        self.price = price
        self.quantity = quantity
        self.maturity_date = TimeStamp(maturity_date)
        self.buy = BasicBoundBuy(self)
        self.bid = BasicBoundBid(self)

    def __repr__(self):
        return (
            f"{type(self).__name__}.{self.id}:{{"
            f"face={self.face_amount}; "
            f"price={self.price}; "
            f"quantity={self.quantity}; "
            f"maturity={self.maturity_date})}}"
        )

    def reaction(self, **kwargs):
        if 'market_date' in kwargs:
            if TimeStamp(kwargs['market_date']) == self.maturity_date:
                return ReactionEvent(
                    self.id,
                    "bound_return",
                    False,
                    None,
                    self.face_amount.copy() * self.quantity
                )
            else:
                return None
        else:
            return None


if __name__ == "__main__":
    mymoney = BasicCash("CNY", 100)
    bound1 = BasicBound(BasicCash("CNY", 100), BasicCash("USD", 10), 1, '2025-12-31')
    bound2 = BasicBound(BasicCash("CNY", 100), BasicCash("USD", 20), 1, '2025-8-31')
    bound1.append(bound2)
    bound1.append(bound2)
    bound1.append(bound1)
    print([i.id for i in bound1.submodules.values()])

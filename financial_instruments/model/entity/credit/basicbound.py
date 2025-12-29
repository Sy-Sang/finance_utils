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
from financial_instruments.model.fm import FinancialModule, ReactionEvent, Trade
from financial_instruments.model.entity.currency.cash import BasicCash

# 外部模块
import numpy


# 代码块

class BasicBoundBuy(Trade):
    def __init__(self, price: BasicCash, exp_timestamp):
        super().__init__()
        self.price = price
        self.exp_timestamp = TimeStamp(exp_timestamp)

    def __call__(self, market_date, quantity):
        if TimeStamp(market_date) < self.exp_timestamp:
            cost = self.price * quantity
            return cost, quantity
        else:
            return None


class BasicBound(FinancialModule):
    def __init__(
            self,
            name: str,
            face_amount: BasicCash,
            price: BasicCash,
            quantity: int,
            maturity_date: Any,
    ):
        super().__init__(name)
        self.face_amount = face_amount
        self.price = price
        self.quantity = quantity
        self.maturity_date = TimeStamp(maturity_date)
        self.buy = BasicBoundBuy(self.price, self.maturity_date)

    def __repr__(self):
        return (
            f"{type(self).__name__}.{self.name}:{{"
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

    def trade(self, trade_type, *args, **kwargs):
        if trade_type == 'buy':
            bid = self.buy(kwargs['market_date'], kwargs['quantity'])
            if bid:
                cost, quantity = bid
                submodule = self.copy()
                submodule.quantity = quantity
                return {"cost": cost, "submodule": submodule}
            else:
                return None
        else:
            return None


if __name__ == "__main__":
    mymoney = BasicCash("CNY", 100)
    bound1 = BasicBound("bound", BasicCash("CNY", 100), BasicCash("USD", 10), 1, '2025-12-31')
    bound2 = BasicBound("bound", BasicCash("CNY", 100), BasicCash("USD", 20), 1, '2025-8-31')
    bound1.append(bound2)
    bound1.append(bound2)
    bound1.append(bound1)
    print(bound1.collect_trade('buy', market_date='2025-1-1', quantity=2))

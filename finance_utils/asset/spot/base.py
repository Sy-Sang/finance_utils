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
from finance_utils.trader.base import TradeBook, Trader

# 外部模块
import numpy


# 代码块


class Spot(Asset):
    """现货"""

    def __init__(
            self,
            name,
            lot_size: RealNum = None
    ):
        super().__init__(name, lot_size)
        self.constructor = {
            "name": self.name,
            "lot_size": self.lot_size
        }

    def __repr__(self):
        return str(self.constructor)

    @classmethod
    def payoff(cls, initial_price: float, x: float, position: PositionType, *args, **kwargs):
        """损益"""
        return (x - initial_price) * position.value

    def purchased_to(self, trader: Trader, price: float, capital: Union[RealNum, None], timestamp: TimeStr,
                     *args, **kwargs):
        available_capital = trader.capital if capital is None else min(trader.capital, capital)
        available_quantity = self.max_purchase_quantity(price, available_capital)
        if available_quantity > 0:
            trader.capital -= price * available_quantity
            if self.name in trader.position:
                trader.position[self.name].append(timestamp, price, available_quantity, PositionType.long)
            else:
                trader.position[self.name] = SpotTradeBook(self)
                trader.position[self.name].append(timestamp, price, available_quantity, PositionType.long)

    def sold_to(self, trader: Trader, price: RealNum, quantity: Union[RealNum, None],
                timestamp: TimeStr, *args, **kwargs):
        if self.name in trader.position:
            in_position_quantity = trader.position[self.name].in_position_quantity(timestamp)
            if quantity is None:
                available_quantity = in_position_quantity
            else:
                available_quantity = min(quantity, in_position_quantity)
            if available_quantity > 0:
                trader.capital += price * available_quantity
                trader.position[self.name].append(timestamp, price, available_quantity, PositionType.short)
            else:
                pass
        else:
            pass

    def percentage_purchase(self, trader: Trader, price: float, capital_percentage: float,
                            timestamp: TimeStr, *args,
                            **kwargs):
        """按比例买入"""
        capital = trader.capital * capital_percentage
        self.purchased_to(trader, price, capital, timestamp, *args, **kwargs)

    def percentage_sell(self, trader: Trader, price: float,
                        quantity_percentage: float,
                        timestamp: TimeStr, must_int: bool = True, *args, **kwargs):
        """按比例卖出"""
        quantity = trader.position[self.name].in_position_quantity(timestamp) * quantity_percentage
        quantity = quantity // 1 if must_int is True else quantity
        self.sold_to(trader, price, quantity, timestamp, *args, **kwargs)


class SpotTradeUnit:
    """现货交易记录单元"""

    def __init__(self, timestamp: TimeStr, price: float, shares: RealNum, position: PositionType):
        self.timestamp = TimeStamp(timestamp)
        self.price = price
        self.shares = float(shares)
        self.position = position

    def __repr__(self):
        return str([self.timestamp, self.price, self.shares, self.position])


class SpotTradeBook(TradeBook):
    """现货交易记录"""

    def __init__(self, asset: Spot):
        self.asset = asset.clone()
        self.book: list[SpotTradeUnit] = []

    def __repr__(self):
        return f"{self.asset}:{self.book}"

    def __bool__(self):
        if self.book:
            return True
        else:
            return False

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

    def append(self, timestamp: TimeStr, price: RealNum, shares: RealNum, position: PositionType):
        self.book.append(
            SpotTradeUnit(TimeStamp(timestamp), float(price), float(shares), position)
        )
        self.sort()

    def timestamp_domain(self) -> tuple[TimeStamp, TimeStamp]:
        """订单表时间区间"""
        return self.book[0].timestamp, self.book[-1].timestamp

    def interval_book(self, stdt: TimeStr, eddt: TimeStr) -> list[SpotTradeUnit]:
        """按时间区间的交易记录"""
        time_array = numpy.array([i.timestamp.timestamp() for i in self.book])
        interval_index, *_ = numpy.where(
            (time_array >= TimeStamp(stdt).timestamp()) & (time_array <= TimeStamp(eddt).timestamp())
        )
        sorted_book = []
        for i in interval_index:
            sorted_book.append(self.book[i])
        return sorted_book

    def get_book_item(self, *args) -> list[SpotTradeUnit]:
        t_domain = self.timestamp_domain()
        if len(args) == 0:
            return copy.deepcopy(self.book)
        elif len(args) == 1:
            return self.interval_book(t_domain[0], args[0])
        else:
            return self.interval_book(args[0], args[1])

    def long_quantity(self, *args) -> float:
        """买入量"""
        return sum([i.shares for i in self.get_book_item(*args) if i.position == PositionType.long])

    def short_quantity(self, *args) -> float:
        """卖出量"""
        return sum([i.shares for i in self.get_book_item(*args) if i.position == PositionType.short])

    def in_position_quantity(self, *args):
        return self.long_quantity(*args) - self.short_quantity(*args)

    def holding_cost(self, *args) -> float:
        """平均持仓成本"""
        long_list = [
            [i.price, i.shares] for i in self.get_book_item(*args) if i.position == PositionType.long
        ]
        short_list = [
            [i.price, i.shares] for i in self.get_book_item(*args) if i.position == PositionType.short
        ]

        long_array = numpy.array(long_list).astype(float) if long_list else numpy.array([[0, 0]])
        short_book = numpy.array(short_list).astype(float) if short_list else numpy.array([[0, 0]])
        amount = numpy.sum(long_array[:, 0] * long_array[:, 1]) - numpy.sum(short_book[:, 0] * short_book[:, 1])
        quantity = numpy.sum(long_array[:, 1]) - numpy.sum(short_book[:, 1])
        return amount / quantity

    def value(self, price: float, *args):
        return self.in_position_quantity(*args) * price

    def payoff(self, x: float, *args) -> float:
        """损益函数"""
        cost = self.holding_cost(*args)
        return (x - cost) * self.in_position_quantity(*args)

    def simplify(self):
        """化简"""
        shares = self.in_position_quantity()
        if shares > 0:
            stdt, eddt = self.timestamp_domain()
            cost = self.holding_cost()
            self.book = [SpotTradeUnit(eddt, cost, shares, PositionType.long)]
        else:
            self.book = []


if __name__ == "__main__":
    test_trader = Trader(100000)
    s = Spot("10001", 100)
    # s.trade().purchase(trader, 100, None, )
    s.percentage_purchase(test_trader, 101, 0.5, "2024-10-1")
    s.percentage_sell(test_trader, 102, 0.5, "2024-10-2")
    s.percentage_sell(test_trader, 103, 0.5, "2024-10-3")
    s.percentage_sell(test_trader, 104, 1, "2024-10-3")
    print(test_trader)
    print(test_trader.net_worth_rate(**{"10001": {"price": 100}}))

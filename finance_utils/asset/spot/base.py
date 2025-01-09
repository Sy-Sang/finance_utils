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
from finance_utils.trader.base import TradeBook, TradeBookUnit, Trader

# 外部模块
import numpy


# 代码块

class SpotTradeBookUnit(TradeBookUnit):
    """现货交易记录单元"""

    def __init__(self, timestamp: TimeStr, price: float, shares: RealNum, position: PositionType):
        super().__init__(timestamp)
        self.price = price
        self.shares = float(shares)
        self.position = position

    def __repr__(self):
        return str([self.timestamp, self.price, self.shares, self.position])


class SpotTradeBook(TradeBook):
    """现货交易记录"""

    def __init__(self, asset: Asset):
        super().__init__()
        self.asset = asset.clone()
        self.book: list[SpotTradeBookUnit]

    def __repr__(self):
        return f"{self.asset}:{self.book}"

    def __bool__(self):
        if self.book:
            return True
        else:
            return False

    def append(self, timestamp: TimeStr, price: RealNum, shares: RealNum, position: PositionType):
        self.book.append(
            SpotTradeBookUnit(TimeStamp(timestamp), float(price), float(shares), position)
        )
        self.sort()

    def interval_book(self, stdt: TimeStr, eddt: TimeStr) -> list[SpotTradeBookUnit]:
        """按时间区间的交易记录"""
        time_array = numpy.array([i.timestamp.timestamp() for i in self.book])
        interval_index, *_ = numpy.where(
            (time_array >= TimeStamp(stdt).timestamp()) & (time_array <= TimeStamp(eddt).timestamp())
        )
        sorted_book = []
        for i in interval_index:
            sorted_book.append(self.book[i])
        return sorted_book

    def get_book_item(self, *args) -> list[SpotTradeBookUnit]:
        """按时间顺序获取部分交易记录"""
        t_domain = self.timestamp_domain()
        if len(args) == 0:
            return copy.deepcopy(self.book)
        elif len(args) == 1:
            return self.interval_book(t_domain[0], args[0])
        else:
            return self.interval_book(args[0], args[1])

    def long_quantity(self, *args) -> float:
        return sum([i.shares for i in self.get_book_item(*args) if i.position == PositionType.long])

    def short_quantity(self, *args) -> float:
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
        if quantity > 0:
            return amount / quantity
        elif quantity < 0:
            raise Exception(f"{quantity} < 0")
        else:
            return numpy.nan

    def value(self, price: float, *args):
        return self.in_position_quantity(*args) * price

    def payoff(self, x: float, *args) -> float:
        """损益函数"""
        cost = self.holding_cost(*args)
        return (x - cost) * self.in_position_quantity(*args)

    def simplify(self):
        shares = self.in_position_quantity()
        stdt, eddt = self.timestamp_domain()
        if shares > 0:
            cost = self.holding_cost()
            self.book = [SpotTradeBookUnit(stdt, cost, shares, PositionType.long)]
        elif shares < 0:
            raise Exception(f"{shares} < 0")
        else:
            self.book = [SpotTradeBookUnit(stdt, 0, shares, PositionType.long)]


class Spot(Asset):
    """现货"""

    def __init__(
            self,
            name,
            lot_size: RealNum = None,
            trade_delta: Union[list, tuple] = ("day", 0)
    ):
        super().__init__(name, lot_size, trade_delta)
        self.constructor = {
            "name": self.name,
            "lot_size": self.lot_size,
            "trade_delta": self.trade_delta
        }

    def __repr__(self):
        return str(self.constructor)

    def to_spot_trade_book(self, book: TradeBook) -> SpotTradeBook:
        """转换为spotbook"""
        spot_book = SpotTradeBook(self)
        for i, b in enumerate(book.book):
            if isinstance(b, SpotTradeBookUnit):
                spot_book.append(b.timestamp, b.price, b.shares, b.position)
        return spot_book

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
                book = self.to_spot_trade_book(trader.position[self.name])
                book.append(timestamp, price, available_quantity, PositionType.long)
                # trader.position[self.name].append(timestamp, price, available_quantity, PositionType.long)
                trader.position[self.name] = book.clone()
            else:
                book = SpotTradeBook(self)
                book.append(timestamp, price, available_quantity, PositionType.long)
                trader.position[self.name] = book.clone()
                # trader.position[self.name] = SpotTradeBook(self)
                # trader.position[self.name].append(timestamp, price, available_quantity, PositionType.long)
            return available_quantity
        else:
            return 0

    def sold_to(self, trader: Trader, price: RealNum, quantity: Union[RealNum, None],
                timestamp: TimeStr, *args, **kwargs):
        if self.name in trader.position:
            in_position_quantity = trader.position[self.name].in_position_quantity(
                TimeStamp(timestamp).get_date() - self.trade_delta
            )
            if quantity is None:
                available_quantity = in_position_quantity
            else:
                available_quantity = min(quantity, in_position_quantity)
            if available_quantity > 0:
                trader.capital += price * available_quantity
                trader.position[self.name].append(timestamp, price, available_quantity, PositionType.short)
                return available_quantity
            else:
                return 0
        else:
            return -1

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

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
import warnings

# 项目模块
from finance_utils.uniontypes import *
from finance_utils.namedtuples import *

from finance_utils.asset.base import *
from finance_utils.trader.base import TradeBook, TradeBookUnit, Trader

# 外部模块
import numpy


# 代码块

class SpotTradeBookUnit(TradeBookUnit):
    """现货交易记录单元"""

    def __init__(self, timestamp: TimeStr, price: float, shares: Rational, position: PositionType):
        super().__init__(timestamp)
        self.price = price
        self.shares = float(shares)
        self.position = position
        self.tag = None

    def set_tag(self, new_tag: Any = None):
        """设置标签"""
        self.tag = new_tag
        return self

    def __repr__(self):
        return str([self.timestamp, self.price, self.shares, self.position])


class SpotTradeBook(TradeBook):
    """现货交易记录"""

    def __init__(self, asset: Asset):
        super().__init__()
        self.asset = asset
        self.book: list[SpotTradeBookUnit]

    def __repr__(self):
        return f"{self.asset}:{self.book}"

    def __bool__(self):
        if self.book:
            return True
        else:
            return False

    def append(self, timestamp: TimeStr, price: Rational, shares: Rational, position: PositionType):

        if self.book:
            last_timestamp = self.book[-1].timestamp
            if timestamp is None:
                ts = last_timestamp + ["microsec", 1]
            else:
                ts = TimeStamp(timestamp)
                if ts < last_timestamp:
                    raise Exception(f"Time sequence error:{ts} < {last_timestamp}")
                elif ts == last_timestamp:
                    ts += ["microsec", 1]
                else:
                    pass
        else:
            if timestamp is None:
                ts = TimeStamp.now()
            else:
                ts = TimeStamp(timestamp)
        self.book.append(
            SpotTradeBookUnit(ts, float(price), float(shares), position)
        )

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
        if args:
            return sum([i.shares for i in self.get_book_item(*args) if i.position == PositionType.long])
        else:
            return sum([i.shares for i in self.book if i.position == PositionType.long])

    def short_quantity(self, *args) -> float:
        if args:
            return sum([i.shares for i in self.get_book_item(*args) if i.position == PositionType.short])
        else:
            return sum([i.shares for i in self.book if i.position == PositionType.short])

    def in_position_quantity(self, *args):
        return self.long_quantity(*args) - self.short_quantity(*args)

    def holding_cost(self, *args) -> float:
        """平均持仓成本"""
        if args:
            book_item = self.get_book_item(*args)
        else:
            book_item = self.book
        long_items = [(i.price, i.shares) for i in book_item if i.position == PositionType.long]
        short_items = [(i.price, i.shares) for i in book_item if i.position == PositionType.short]
        long_array = numpy.array(long_items) if long_items else numpy.zeros((1, 2))
        short_array = numpy.array(short_items) if short_items else numpy.zeros((1, 2))
        total_amount = numpy.sum(long_array[:, 0] * long_array[:, 1]) - numpy.sum(short_array[:, 0] * short_array[:, 1])
        total_quantity = numpy.sum(long_array[:, 1]) - numpy.sum(short_array[:, 1])
        if total_quantity > 0:
            return total_amount / total_quantity
        elif total_quantity < 0:
            raise Exception(f"total_quantity:{total_quantity} < 0")
        else:
            return 0

    def quantity_and_cost(self, *args):
        """持仓量与持仓成本"""
        if args:
            book_item = self.get_book_item(*args)
        else:
            book_item = self.book
        long_items = [(i.price, i.shares) for i in book_item if i.position == PositionType.long]
        short_items = [(i.price, i.shares) for i in book_item if i.position == PositionType.short]
        long_array = numpy.array(long_items) if long_items else numpy.zeros((1, 2))
        short_array = numpy.array(short_items) if short_items else numpy.zeros((1, 2))
        total_amount = numpy.sum(long_array[:, 0] * long_array[:, 1]) - numpy.sum(short_array[:, 0] * short_array[:, 1])
        total_quantity = numpy.sum(long_array[:, 1]) - numpy.sum(short_array[:, 1])
        if total_quantity > 0:
            return total_quantity, total_amount / total_quantity
        elif total_quantity < 0:
            warnings.warn(f"quantity: {total_quantity}<0", category=UserWarning)
            return total_quantity, 0
        else:
            return 0, 0

    @classmethod
    def cls_in_position_quantity(cls, position: list[SpotTradeBookUnit]) -> Rational:
        """持仓量"""
        long_sum = sum([
            i.shares for i in position if i.position == PositionType.long
        ])
        short_sum = sum([
            i.shares for i in position if i.position == PositionType.short
        ])
        return long_sum - short_sum

    @classmethod
    def cls_holding_cost(cls, position: list[SpotTradeBookUnit]) -> Rational:
        """持仓成本"""
        long_list = [
            [i.price, i.shares] for i in position if i.position == PositionType.long
        ]
        short_list = [
            [i.price, i.shares] for i in position if i.position == PositionType.short
        ]

        long_array = numpy.array(long_list) if long_list else numpy.zeros((1, 2))
        short_book = numpy.array(short_list) if short_list else numpy.zeros((1, 2))
        amount = numpy.sum(long_array[:, 0] * long_array[:, 1]) - numpy.sum(short_book[:, 0] * short_book[:, 1])
        quantity = numpy.sum(long_array[:, 1]) - numpy.sum(short_book[:, 1])
        if quantity > 0:
            return amount / quantity
        elif quantity < 0:
            raise Exception(f"total_quantity: {quantity} < 0")
        else:
            return 0

    @classmethod
    def cls_quantity_and_cost(cls, position: list[SpotTradeBookUnit]):
        """获取现货book的量与成本"""
        long_list = [
            [i.price, i.shares] for i in position if i.position == PositionType.long
        ]
        short_list = [
            [i.price, i.shares] for i in position if i.position == PositionType.short
        ]

        long_array = numpy.array(long_list) if long_list else numpy.zeros((1, 2))
        short_book = numpy.array(short_list) if short_list else numpy.zeros((1, 2))
        amount = numpy.sum(long_array[:, 0] * long_array[:, 1]) - numpy.sum(short_book[:, 0] * short_book[:, 1])
        quantity = numpy.sum(long_array[:, 1]) - numpy.sum(short_book[:, 1])
        if quantity > 0:
            return quantity, amount / quantity
        elif quantity < 0:
            warnings.warn(f"quantity: {quantity}<0", category=UserWarning)
            return quantity, 0
        else:
            return 0, 0

    def value(self, price: float, *args):
        return self.in_position_quantity(*args) * price

    def payoff(self, x: float, *args) -> float:
        """损益函数"""
        cost = self.holding_cost(*args)
        return (x - cost) * self.in_position_quantity(*args)

    def simplify(self, timestamp: TimeStr):
        if timestamp is None:
            ts = self.next_timestamp()
        else:
            ts = TimeStamp(timestamp)

        stdt, eddt = self.timestamp_domain()
        if self.asset.trade_delta[1] != 0:
            unsellable_timestamp = Asset.untradable(ts, self.asset.trade_delta)
            unsellable_trades = []
            other_trades = []
            for i in self.book:
                if i.timestamp < unsellable_timestamp or i.position == PositionType.short:
                    other_trades.append(i)
                else:
                    unsellable_trades.append(i)
            holding_quantity, holding_cost = self.cls_quantity_and_cost(other_trades)
            self.book = [SpotTradeBookUnit(
                stdt, holding_cost, holding_quantity,
                PositionType.long
            )] + unsellable_trades
        else:
            holding_quantity, holding_cost = self.quantity_and_cost()
            sellable_unit = SpotTradeBookUnit(stdt, holding_cost, holding_quantity, PositionType.long)
            self.book = [
                sellable_unit
            ]


class Spot(Asset):
    """现货"""

    def __init__(
            self,
            name,
            lot_size: Rational = None,
            trade_delta: TradeDelta = TradeDelta("day", 0)
    ):
        super().__init__(name, lot_size, trade_delta)
        self.constructor = {
            "name": self.name,
            "lot_size": self.lot_size,
            "trade_delta": self.trade_delta
        }

    def __repr__(self):
        return str(self.constructor)

    def clone(self, new_name: str = None):
        constructor = copy.deepcopy(self.constructor)
        if new_name is None:
            pass
        else:
            constructor["name"] = new_name
        return type(self)(**constructor)

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

    def purchased_to(self, trader: Trader, price: float, capital: Union[Rational, None], timestamp: TimeStr,
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
            return available_quantity
        else:
            return 0

    def sold_to(self, trader: Trader, price: Rational, quantity: Union[Rational, None],
                timestamp: TimeStr, *args, **kwargs):
        if self.name in trader.position:
            if timestamp is None:
                ts = trader.position[self.name].next_timestamp()
            else:
                ts = TimeStamp(timestamp)
            untradable_timestamp = Asset.untradable(ts, self.trade_delta)
            book = []
            for i in trader.position[self.name].book:
                i: SpotTradeBookUnit
                if i.timestamp < untradable_timestamp or i.position == PositionType.short:
                    book.append(i)
            in_position_quantity = SpotTradeBook.cls_in_position_quantity(book)
            if quantity is None:
                available_quantity = in_position_quantity
            else:
                available_quantity = min(quantity, in_position_quantity)
            if available_quantity > 0:
                trader.capital += price * available_quantity
                trader.position[self.name].append(ts, price, available_quantity, PositionType.short)
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
    time0 = TimeStamp.now()
    test_trader = Trader("trader", 10000 * 100, "2020-1-1")
    s = Spot("10001", 100, TradeDelta("day", 1))
    s.purchased_to(test_trader, 10, 10000, "2023-1-1")
    for i in range(1000):
        t = TimeStamp("2024-1-1") + ["day", i]
        s.purchased_to(test_trader, 10, 10000, t)
        s.sold_to(test_trader, 11, None, t)
        test_trader.position_simplify(None)
        # print(len(test_trader.position[s.name].book))
        test_trader.position[s.name].holding_cost()
    print(TimeStamp.now() - time0)

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
from typing import Union, Self, Dict
from collections import namedtuple

# 项目模块
from finance_utils.asset.base import *
from finance_utils.asset.spot.base import Spot, SpotElement, SpotTradeBook

from easy_datetime.timestamp import TimeStamp

# 外部模块
import numpy


# 代码块


class SpotTrader:
    """基础交易员"""

    def __init__(self, capital: float):
        self.initial_capital = capital
        self.capital = capital
        self.position: Dict[Union[int, str], SpotTradeBook] = {}

    def __repr__(self):
        return str({
            "capital": self.capital,
            "spot_position": self.position
        })

    def capital_injection(self, capital: RealNum):
        self.capital += capital

    def in_position_quantity(self, name):
        if name in self.position:
            return self.position[name].long_quantity() - self.position[name].short_quantity()
        else:
            return 0

    def purchase(self, price: RealNum, capital: Union[RealNum, None], asset: Spot, timestamp: TimeStr, *args, **kwargs):
        """买入现货"""
        available_capital = self.capital if capital is None else min(self.capital, capital)
        available_quantity = asset.max_purchase_quantity(price, available_capital)
        if available_quantity > 0:
            self.capital -= price * available_quantity
            if asset.name in self.position:
                self.position[asset.name].append(timestamp, price, available_quantity, PositionType.long)
            else:
                self.position[asset.name] = SpotTradeBook(asset)
                self.position[asset.name].append(timestamp, price, available_quantity, PositionType.long)

    def sell(self, name: Union[int, str], price: RealNum, quantity: Union[RealNum, None], timestamp: TimeStr,
             *args, **kwargs):
        """卖出现货"""
        if name in self.position:
            in_position_quantity = self.in_position_quantity(name)
            if quantity is None:
                available_quantity = in_position_quantity
            else:
                available_quantity = min(quantity, in_position_quantity)
            if available_quantity > 0:
                self.capital += price * available_quantity
                self.position[name].append(timestamp, price, available_quantity, PositionType.short)
            else:
                pass
        else:
            pass

    def percentage_purchase(self, price: RealNum, capital_percentage: RealNum, asset: Spot, timestamp: TimeStr, *args,
                            **kwargs):
        capital = self.capital * capital_percentage
        self.purchase(price, capital, asset, timestamp, *args, **kwargs)

    def percentage_sell(self, name: Union[int, str], price: RealNum, quantity_percentage: Union[RealNum, None],
                        timestamp: TimeStr, must_int: bool = True, *args, **kwargs):
        quantity = self.in_position_quantity(name) * quantity_percentage
        quantity = quantity // 1 if must_int is True else quantity
        self.sell(name, price, quantity, timestamp, *args, **kwargs)

    def value(self, name: Union[int, str], price: RealNum):
        if name in self.position:
            in_position_quantity = self.in_position_quantity(name)
            return price * in_position_quantity
        else:
            return 0

    def position_value(self, newest_price_dic: Dict[Union[int, str], RealNum]):
        total_value = 0
        for k, v in newest_price_dic.items():
            total_value += self.value(k, v)
        return total_value + self.capital


if __name__ == "__main__":
    from matplotlib import pyplot

    timeline = TimeStamp.timestamp_range("2024-1-1", "2024-2-1", "hour", 1)
    p = [numpy.random.normal(100, 10) for _ in range(len(timeline))]
    p1 = [numpy.random.normal(100, 20) for _ in range(len(timeline))]

    s = Spot(10001, 100)
    s1 = Spot(10002, 100)
    #
    t = SpotTrader(20000)
    v = []
    for i, x in enumerate(timeline):
        o = numpy.random.randint(0, 4)
        if o == 0:
            t.purchase(p[i], None, s, timeline[i])
        elif o == 1:
            t.sell(s.name, p[i], None, timeline[i])
        if o == 2:
            t.purchase(p1[i], None, s1, timeline[i])
        if o == 3:
            t.sell(s1.name, p1[i], None, timeline[i])
        else:
            pass
        v.append(t.position_value({
            s.name: p[i],
            s1.name: p1[i]
        }))

    pyplot.subplot(1, 2, 1)
    pyplot.plot(p)
    pyplot.plot(p1)
    pyplot.subplot(1, 2, 2)
    pyplot.plot(v)
    pyplot.show()
    print(t)

    # t.purchase(100, 100 * 202, s, "2024-1-1")
    # t.purchase(101, 20000, s, "2024-1-2")
    # t.purchase(98, 10000, s, "2024-1-3")
    # t.percentage_sell(s.name, 90, 0.5, "2024-1-4")
    # t.percentage_sell(s.name, 95, 0.5, "2024-1-5")
    # t.percentage_sell(s.name, 95, 0.5, "2024-1-6")
    # print(t.value({10001: p[-1]}))

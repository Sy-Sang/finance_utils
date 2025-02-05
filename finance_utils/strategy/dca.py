#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dollar cost average
"""

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
from typing import Union, Self, List, Set
from collections import namedtuple

# 项目模块
from easy_datetime.timestamp import TimeStamp

from finance_utils.asset.spot.base import Spot
from finance_utils.trader.base import Trader
from finance_utils.process.prices.base import PriceProcess, PricePathValue
from finance_utils.process.prices.gbm import RVDecoupledGBM, GBM
from finance_utils.trader.utility import UtilityFunction, LogUtilityFunction, PowerUtilityFunction

# 外部模块
import numpy


# 代码块

def simple_dca(s0: float, index: int, timestamp: TimeStamp, path_context: PricePathValue, asset: Spot, trader: Trader,
               capital: float, trigger_set: Set[int]):
    """基础dca"""

    if index in trigger_set:
        asset.purchased_to(trader, path_context.price, capital, timestamp)
        trader.position_simplify(None)
    else:
        pass
    spot_yield = path_context.price / s0
    trader_yield = trader.net_worth_rate(**path_context.dic)
    return trader_yield, spot_yield


def price_qualified_dca(s0: float, index: int, timestamp: TimeStamp, path_context: PricePathValue, asset: Spot,
                        trader: Trader, capital: float, trigger_set: Set[int], purchase_qualif: float,
                        sell_qualif: float):
    """根据价格确定买卖点的dca"""
    if trader.in_position(asset.name) > 0:
        cost = trader.position[asset.name].holding_cost()
        if path_context.price < cost * purchase_qualif:
            asset.purchased_to(trader, path_context.price, capital, timestamp)
            trader.position_simplify(None)
        elif path_context.price > cost * sell_qualif:
            asset.sold_to(trader, path_context.price, None, timestamp)
            trader.position_simplify(None)
        else:
            pass
    else:
        if index in trigger_set:
            asset.purchased_to(trader, path_context.price, capital, timestamp)
            trader.position_simplify(None)
        else:
            pass
    spot_yield = path_context.price / s0
    trader_yield = trader.net_worth_rate(**path_context.dic)
    return trader_yield, spot_yield


if __name__ == "__main__":
    from matplotlib import pyplot

    stock = Spot("stock", 1)
    process = GBM("stock", 1, 0.03 / 100, 0.015, 100, TimeStamp.now().accurate_to("year"), "day", 1)
    trader = Trader("trader", 100000)
    trader2 = Trader("trader2", 100000)

    sy = []
    ty = []
    ty2 = []
    trigger_set = set(range(0, 100, 10))
    for i, t, path in process.process_iter():
        trader_yield, spot_yield = simple_dca(process.s0, i, t, path, stock, trader, 10000, trigger_set)
        trader2_yield, _ = price_qualified_dca(process.s0, i, t, path, stock, trader2, 10000, {0}, 0.99, 1.01)
        sy.append(spot_yield)
        ty.append(trader_yield)
        ty2.append(trader2_yield)

    pyplot.plot(sy)
    pyplot.plot(ty)
    pyplot.plot(ty2)
    pyplot.show()
    # print(trader.position["stock"])
    # print(ty)

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
from typing import Union, Self
from collections import namedtuple

# 项目模块
from finance_utils.asset.spot.base import Spot
from finance_utils.trader.base import Trader
from finance_utils.process.prices.base import MultiPathing
from finance_utils.process.prices.gbm import RVDecoupledGBM, GBM

# 外部模块
import numpy


# 代码块

class MonteCarloResult:
    """压力测试结果"""

    def __init__(self, yield_slice_list: list, price_slice_list: list):
        self.yield_slices = numpy.array(yield_slice_list)
        self.spot_slices = numpy.array(price_slice_list)
        self.length = len(self.yield_slices)
        self.width = len(self.yield_slices[0])

    def max_drawdown(self):
        """回撤"""
        drawdown_tuple = namedtuple("drawdown_tuple", ["yield_drawdown", "spot_drawdown"])
        return drawdown_tuple(numpy.min(self.yield_slices, axis=0), numpy.min(self.spot_slices, axis=0))

    def mean_sharp(self, risk_free_rate: float = 1.03):
        """平均收益率夏普比率"""

        def f(xlist):
            return (numpy.mean(xlist, axis=0) - risk_free_rate) / numpy.std(xlist, ddof=1, axis=0)

        sharp_tuple = namedtuple("sharp_tuple", ["yield_sharp", "spot_sharp"])

        return sharp_tuple(f(self.yield_slices), f(self.spot_slices))

    def slice_sharp(self, risk_free_rate: float = 1.03, slice_index: int = -1):
        """切片收益率夏普比率"""

        def f(xlist):
            return (xlist[slice_index] - risk_free_rate) / numpy.std(xlist, ddof=1, axis=0)

        sharp_tuple = namedtuple("sharp_tuple", ["yield_sharp", "spot_sharp"])

        return sharp_tuple(f(self.yield_slices), f(self.spot_slices))

    def slice_quantile_rate(self, risk_free_rate: float = 1.03, slice_index: int = -1):
        """切片分位数比例"""
        pass


class SpotCostAveragingPlan:
    """现货定投"""

    def __init__(self, mp: MultiPathing, asset: Spot):
        self.mp = mp
        self.asset = asset
        self.s0 = mp.processes[0].s0

    def simple_cap(self, capital: float, step: int):
        """简单cap"""
        self.mp.clear()
        yield_list = []
        price_list = []
        step_list = list(range(0, self.mp.length, step))
        for i, t in enumerate(self.mp.timeline):
            yield_slice = []
            price_slice = []
            for j in range(self.mp.width):
                path_data = self.mp.processes[j].get_price(t)
                price_slice.append(path_data.price / self.s0)
                if i in step_list:
                    self.asset.purchased_to(self.mp.trades[j], path_data.price, capital, t)
                    self.mp.trades[j].position_simplify(None)
                else:
                    pass
                yield_slice.append(self.mp.trades[j].net_worth_rate(**path_data.dic))
            yield_list.append(yield_slice)
            price_list.append(price_slice)

        return MonteCarloResult(yield_list, price_list)

    def monthly_cap(self, capital: float, days: list[int]):
        """按月度cap"""
        self.mp.clear()
        yield_list = []
        price_list = []
        for i, t in enumerate(self.mp.timeline):
            yield_slice = []
            price_slice = []
            for j in range(self.mp.width):
                path_data = self.mp.processes[j].get_price(t)
                price_slice.append(path_data.price / self.s0)
                if t.day in days:
                    self.asset.purchased_to(self.mp.trades[j], path_data.price, capital, t)
                    self.mp.trades[j].position_simplify(None)
                else:
                    pass
                yield_slice.append(self.mp.trades[j].net_worth_rate(**path_data.dic))
            yield_list.append(yield_slice)
            price_list.append(price_slice)

        return MonteCarloResult(yield_list, price_list)

    def price_qualified_cap(self, capital: float, purchase_qualif: float, sell_qualif: float):
        """根据价格有条件触发的cap"""
        self.mp.clear()
        yield_list = []
        price_list = []
        for i, t in enumerate(self.mp.timeline):
            yield_slice = []
            price_slice = []
            for j in range(self.mp.width):
                path_data = self.mp.processes[j].get_price(t)
                price_slice.append(path_data.price / self.s0)

                if self.mp.trades[j].in_position(self.asset.name) > 0:
                    cost = self.mp.trades[j].position[self.asset.name].holding_cost()
                    if path_data.price < cost * purchase_qualif:
                        self.asset.purchased_to(self.mp.trades[j], path_data.price, capital, t)
                        self.mp.trades[j].position_simplify(None)
                    elif path_data.price > cost * sell_qualif:
                        self.asset.sold_to(self.mp.trades[j], path_data.price, None, t)
                        self.mp.trades[j].position_simplify(None)
                    else:
                        pass
                else:
                    self.asset.purchased_to(self.mp.trades[j], path_data.price, capital, t)
                    self.mp.trades[j].position_simplify(None)
                yield_slice.append(self.mp.trades[j].net_worth_rate(**path_data.dic))
            yield_list.append(yield_slice)
            price_list.append(price_slice)

        return MonteCarloResult(yield_list, price_list)


if __name__ == "__main__":
    pass

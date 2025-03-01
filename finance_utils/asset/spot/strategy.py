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
from typing import Union, Self, Type
from collections import namedtuple

# 项目模块
from finance_utils.asset.spot.base import Spot
from finance_utils.trader.base import Trader
from finance_utils.process.prices.base import MultiPathing
from finance_utils.process.prices.gbm import RVDecoupledGBM, GBM
from finance_utils.trader.utility import UtilityFunction, LogUtilityFunction, PowerUtilityFunction
# 外部模块
import numpy


# 代码块

class MonteCarloResult:
    """压力测试结果"""

    slice_tuple = namedtuple("slice_tuple", ["strategy", "spot"])

    def __init__(self, yield_slice_list: list, price_slice_list: list):
        self.yield_slices = numpy.array(yield_slice_list)
        self.spot_slices = numpy.array(price_slice_list)
        self.length = len(self.yield_slices)
        self.width = len(self.yield_slices[0])

    def max_drawdown(self):
        """回撤"""
        return self.slice_tuple(numpy.min(self.yield_slices, axis=0), numpy.min(self.spot_slices, axis=0))

    def mean_sharp(self, risk_free_rate: float = 1.03):
        """平均收益率夏普比率"""

        def f(xlist):
            return (numpy.mean(xlist, axis=0) - risk_free_rate) / numpy.std(xlist, ddof=1, axis=0)

        return self.slice_tuple(f(self.yield_slices), f(self.spot_slices))

    def slice_sharp(self, risk_free_rate: float = 1.03, slice_index: int = -1):
        """切片收益率夏普比率"""

        def f(xlist):
            return (xlist[slice_index] - risk_free_rate) / numpy.std(xlist, ddof=1, axis=0)

        return self.slice_tuple(f(self.yield_slices), f(self.spot_slices))

    def quantile_path(self, quantile: float, rank_by_spot: bool = False):
        """分位数路径"""
        path_tuple = namedtuple("path_tuple", ["strategy", "spot", "i"])
        quantile_index = numpy.quantile(numpy.arange(self.width), quantile, method="nearest")
        if rank_by_spot:
            rank_index = numpy.argsort(self.spot_slices[-1])
        else:
            rank_index = numpy.argsort(self.yield_slices[-1])
        index = rank_index[quantile_index]
        return path_tuple(
            [i[index] for i in self.yield_slices],
            [i[index] for i in self.spot_slices],
            index
        )

    def slice_utility_ratio(
            self,
            q1: float,
            q2: float,
            risk_free_rate: float = 1.03,
            slice_index: int = -1,
            positive_utility_function: UtilityFunction = LogUtilityFunction(),
            negative_utility_function: UtilityFunction = PowerUtilityFunction(),
            *args, **kwargs
    ):
        """切片分位数比例"""

        def f(x1, x2):
            x1 = 1 + positive_utility_function(x1 - risk_free_rate, *args, **kwargs) if x1 >= risk_free_rate else 1
            x2 = 1 + negative_utility_function(x2 - risk_free_rate, *args, **kwargs) if x2 < risk_free_rate else 1
            return x1 / x2 - 1

        ydata = self.yield_slices[slice_index]
        sdata = self.spot_slices[slice_index]
        y1 = numpy.quantile(ydata, q1)
        y2 = numpy.quantile(ydata, q2)
        s1 = numpy.quantile(sdata, q1)
        s2 = numpy.quantile(sdata, q2)

        return self.slice_tuple(f(y1, y2), f(s1, s2))

    def quantile_diff_utility(
            self, slice_index: int = -1,
            positive_utility_function: UtilityFunction = LogUtilityFunction(),
            negative_utility_function: UtilityFunction = PowerUtilityFunction(),
            *args, **kwargs
    ):
        """分位数差分效用"""

        def f(x1, x2):
            y = positive_utility_function(x1 - x2, *args, **kwargs) if x1 >= x2 else -1 * negative_utility_function(
                x2 - x1, *args, **kwargs)
            return y

        ydata = self.yield_slices[slice_index]
        sdata = self.spot_slices[slice_index]
        ulist = []
        for i, y in enumerate(ydata):
            ulist.append(f(y, sdata[i]))

        return numpy.mean(ulist)


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

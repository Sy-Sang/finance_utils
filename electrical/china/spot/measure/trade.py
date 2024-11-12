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
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution
from data_utils.stochastic_utils.distributions.nonParametricDistribution import HistogramDist
from data_utils.solve_utils.equationNSolve import gradient_descent
from easy_utils.number_utils.number_utils import EasyFloat
from easy_datetime.temporal_utils import timer
from easy_utils.obj_utils.enumerable_utils import flatten

from finance_utils.electrical.china.spot.rule.recycle import Recycle, AnarchismRecycle, SampleRecycle
from finance_utils.electrical.china.spot.rule.settlement import province_new_energy_with_recycle

from finance_utils.electrical.china.spot.measure.internal.forecast import ForecastCurve, Epsilon

# 外部模块
import numpy
from scipy.optimize import differential_evolution

# 代码块

SearchResult = namedtuple("SearchResult", ["y", "total"])


class MarketSample:
    """市场样本"""

    def __init__(self, d: numpy.ndarray, r: numpy.ndarray, q: numpy.ndarray):
        self.dayahead_sample = d
        self.realtime_sample = r
        self.quantity_sample = q

    def trade_yield(self, xlist, recycle: Type[Recycle] = None, *args, **kwargs) -> numpy.ndarray:
        """交易收益"""
        return numpy.array([
            province_new_energy_with_recycle(
                self.dayahead_sample[i],
                self.realtime_sample[i],
                self.quantity_sample[i],
                xlist,
                recycle=recycle, *args, **kwargs) for i in range(len(self.dayahead_sample))
        ]).astype(float)

    def differential_evolution__search(self, recycle: Type[Recycle] = None,
                                       submitted_min: float = 0, submitted_max: float = None,
                                       *args, **kwargs):
        """差分进化搜索"""

        def target_fun(xlist):
            xlist = EasyFloat.put_in_range(submitted_min, submitted_max, *xlist)
            trade_yield = numpy.sum(self.trade_yield(xlist, recycle=recycle, *args, **kwargs))
            return -1 * trade_yield

        bounds = [[submitted_min, submitted_max] for _ in range(len(self.dayahead_sample[0]))]
        result = differential_evolution(target_fun, bounds)
        return SearchResult(result.x, -result.fun)

    def compare(self, xlist, ylist, recycle: Type[Recycle] = None, *args, **kwargs) -> numpy.ndarray:
        """对比"""
        diff = self.trade_yield(xlist, recycle=recycle, *args, **kwargs) - self.trade_yield(ylist, recycle=recycle,
                                                                                            *args, **kwargs)
        return numpy.sort(diff)

    def compare_dist(self, xlist, ylist, recycle: Type[Recycle] = None, *args, **kwargs) -> HistogramDist:
        """对比分布"""
        return HistogramDist(self.compare(xlist, ylist, recycle, *args, **kwargs))


class ForecastMarket:
    """预测市场信息"""

    def __init__(self, dayahead: ForecastCurve, realtime: ForecastCurve, quantity: ForecastCurve):
        self.dayahead = dayahead
        self.realtime = realtime
        self.quantity = quantity

    def typical_sample(self):
        """最典型样本"""
        return MarketSample(
            numpy.array([self.dayahead.value_list]).astype(float),
            numpy.array([self.realtime.value_list]).astype(float),
            numpy.array([self.quantity.value_list]).astype(float)
        )

    def random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10,
                      use_random=False) -> MarketSample:
        """随机市场样本"""
        dayahead_sample = self.dayahead.random_sample(first, end, num, use_random)
        realtime_sample = self.realtime.random_sample(first, end, num, use_random)
        quantity_sample = self.quantity.random_sample(first, end, num, use_random)
        return MarketSample(dayahead_sample, realtime_sample, quantity_sample)

    def diff_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10,
                           use_random=False) -> MarketSample:
        """差分随机市场样本"""
        dayahead_sample = self.dayahead.diff_random_sample(first, end, num, use_random)
        realtime_sample = self.realtime.diff_random_sample(first, end, num, use_random)
        quantity_sample = self.quantity.diff_random_sample(first, end, num, use_random)
        return MarketSample(dayahead_sample, realtime_sample, quantity_sample)


if __name__ == "__main__":
    from matplotlib import pyplot

    quantity_min = 0
    quantity_max = 30
    submitted_min = 0
    submitted_max = 30
    trigger_rate = 0.05
    punishment_rate = 0.5
    random_p = 0
    random_p_2 = 0.5

    dayahead = ForecastCurve([
        NormalDistribution(200, 20),
        NormalDistribution(201, 21),
        NormalDistribution(203, 22),
        NormalDistribution(102, 53),
    ])

    realtime = ForecastCurve([
        NormalDistribution(200 * 1.2, 20 * 2.5),
        NormalDistribution(201 * 0.8, 21 * 1.2),
        NormalDistribution(203 * 1.5, 22 * 1.3),
        NormalDistribution(102 * 0.5, 53 * 1.1),
    ])

    quantity = ForecastCurve([
        NormalDistribution(10, 5),
        NormalDistribution(9, 3),
        NormalDistribution(15, 1),
        NormalDistribution(25, 2),
    ], domain_min=quantity_min, domain_max=quantity_max)

    market = ForecastMarket(dayahead, realtime, quantity)

    # print(market.diff_random_sample(num=100, use_random=True).trade_yield(
    #     [10, 10, 10, 10], None, trigger_rate=0.05, punishment_rate=0.5, timer=True
    # ).tolist())

    ts = market.typical_sample().differential_evolution__search(
        None, 0, 25, trigger_rate=0.05, punishment_rate=0.5
    )

    s = market.random_sample(num=100, use_random=True).differential_evolution__search(
        None, 0, 25, trigger_rate=0.05, punishment_rate=0.5)

    pyplot.plot(market.random_sample(num=100, use_random=True).compare(s.y, quantity.value_list).tolist())
    pyplot.plot(market.random_sample(num=100, use_random=True).compare(ts.y, quantity.value_list).tolist())
    pyplot.axhline(0, color='black', linewidth=1)
    pyplot.show()

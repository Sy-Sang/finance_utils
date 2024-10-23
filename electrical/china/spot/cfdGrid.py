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
import itertools

# 项目模块
from easy_datetime.timestamp import TimeStamp
from data_utils.serial_utils.time_series import TimeSeries
from data_utils.serial_utils.series_trans_utils import MinMax
from data_utils.stochastic_utils.random_process import correlatedRandom
from data_utils.solve_utils.equationNSolve import newton_method

# 外部模块
import numpy
from matplotlib import pyplot


# 代码块


class DiscreteForecast:
    """离散预测"""

    def __init__(self, dayahead: dict, realtime: dict, quantity: dict):
        def standardization(xdic: dict[float, float]):
            """标准化"""
            d = list(xdic.items())
            a = numpy.array(d)
            p = 1 / numpy.sum(a[:, 1])
            return {i: xdic[i] * p for i in numpy.sort(a[:, 0])}

        self.dayahead = standardization(dayahead)
        self.realtime = standardization(realtime)
        self.quantity = standardization(quantity)

    def __repr__(self):
        return str({
            "dayahead": self.dayahead,
            "realtime": self.realtime,
            "quantity": self.quantity,
        })

    def forked_tree(self, position: float = 0.5):
        """多叉树"""
        trade_yield = []
        for q in self.quantity.items():
            dayahead_quantity = q[0] * position
            realtime_quantity = q[0] * (1 - position)
            dayahead_yield = []
            realtime_yield = []
            for dp in self.dayahead.items():
                dayahead_yield.append([dayahead_quantity * dp[0], dp[1]])
            for rp in self.realtime.items():
                realtime_yield.append([realtime_quantity * rp[0], rp[1]])
            yield_table = [[i[0][0] + i[1][0], (i[0][1] * i[1][1]) * q[1]] for i in
                           list(itertools.product(dayahead_yield, realtime_yield))]
            trade_yield += yield_table
        return numpy.array(trade_yield)

    def mean(self, position: float = 0.5):
        """平均收益"""
        data = self.forked_tree(position)
        return numpy.sum(data[:, 0] * data[:, 1])

    def std(self, position: float = 0.5):
        """收益标准差"""
        data = self.forked_tree(position)
        m = numpy.sum(data[:, 0] * data[:, 1])
        s = numpy.sum((data[:, 0] - m) ** 2 * data[:, 1])
        return s ** 0.5

    def sharpe(self, position: float = 0.5):
        return (self.mean(position) - self.mean(0.5)) / self.std(position)


if __name__ == "__main__":
    dpf = DiscreteForecast({100: 0.5, 101: 0.5},
                           {80: 0.5, 90: 0.3, 99: 0.3, 100: 0.4, 102: 0.2, 103: 0.3, 104: 0.2, 200: 0.1},
                           {199: 0.1, 200: 0.8, 201: 0.1})
    # print([float(dpf.forked_tree(i / 10)) for i in range(1, 10)])
    tree = dpf.forked_tree(0.1)
    print([float(dpf.mean(i / 10)) for i in range(1, 100)])

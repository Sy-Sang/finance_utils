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
from easy_datetime.temporal_utils import timer
from data_utils.serial_utils.time_series import TimeSeries
from data_utils.serial_utils.series_trans_utils import MinMax
from data_utils.stochastic_utils.random_process import correlatedRandom
from data_utils.solve_utils.equationNSolve import newton_method

# 外部模块
import numpy
from matplotlib import pyplot


# 代码块

class YieldTree:
    def __init__(self, data):
        self.tree = numpy.array(data).astype(float)

    def __repr__(self):
        return str(self.tree.tolist())

    @classmethod
    def product(cls, trees: list[Self]):
        data = []
        product = numpy.array(list(itertools.product(*[t.tree for t in trees])))
        for i, p in enumerate(product):
            discrete_yield = numpy.sum(p[:, 0])
            discrete_probability = numpy.prod(p[:, 1])
            temp_dayahead_quantity = numpy.sum(p[:, 4])
            temp_realtime_quantity = numpy.sum(p[:, 5])
            discrete_dayahead_quantity = temp_dayahead_quantity if temp_dayahead_quantity != 0 else 1
            discrete_realtime_quantity = temp_realtime_quantity if temp_realtime_quantity != 0 else 1
            discrete_dayahead_price = numpy.sum(p[:, 2] * p[:, 4]) / discrete_dayahead_quantity
            discrete_realtime_price = numpy.sum(p[:, 3] * p[:, 5]) / discrete_realtime_quantity
            discrete_submit = numpy.sum(p[:, 6])
            data.append([
                discrete_yield,
                discrete_probability,
                discrete_dayahead_price,
                discrete_realtime_price,
                discrete_submit
            ])

        return numpy.array(data)


class DiscreteForecast:
    """离散预测"""

    def __init__(self, dayahead: list, realtime: list, quantity: list):
        def standardization(xlist: list[list[float]]) -> numpy.ndarray:
            """标准化"""
            a = numpy.array(xlist).astype(float)
            p = 1 / numpy.sum(a[:, 1])
            a[:, 1] = a[:, 1] * p
            return a[numpy.argsort(a[:, 0])]

        self.dayahead = standardization(dayahead)
        self.realtime = standardization(realtime)
        self.quantity = standardization(quantity)

        default_index = numpy.argsort(self.quantity[:, 1])
        self.default_quantity = float(self.quantity[default_index[-1]][0])

        self.dayahead_mean = numpy.sum(self.dayahead[:, 0] * self.dayahead[:, 1])
        self.realtime_mean = numpy.sum(self.realtime[:, 0] * self.realtime[:, 1])
        self.quantity_mean = numpy.sum(self.quantity[:, 0] * self.quantity[:, 1])

    def __repr__(self):
        return str({
            "dayahead": self.dayahead.tolist(),
            "realtime": self.realtime.tolist(),
            "quantity": self.quantity.tolist(),
            "default_quantity": self.default_quantity
        })

    def forked_tree(self, submitted_quantity: float):
        """多叉树"""
        trade_yield = []
        for q in self.quantity:
            dayahead_quantity = submitted_quantity
            realtime_quantity = q[0] - submitted_quantity
            dayahead_yield = []
            realtime_yield = []
            for dp in self.dayahead:
                dayahead_yield.append([dayahead_quantity * dp[0], dp[1], dp[0]])
            for rp in self.realtime:
                realtime_yield.append([realtime_quantity * rp[0], rp[1], rp[0]])
            yield_table = [
                [
                    i[0][0] + i[1][0],
                    i[0][1] * i[1][1] * q[1],
                    i[0][2],
                    i[1][2],
                    dayahead_quantity,
                    realtime_quantity,
                    submitted_quantity
                ]
                for i in list(itertools.product(dayahead_yield, realtime_yield))]
            trade_yield += yield_table
        return YieldTree(trade_yield)

    def mean(self, submitted: float):
        """平均收益"""
        data = self.forked_tree(submitted).tree
        return numpy.sum(data[:, 0] * data[:, 1])

    def std(self, submitted: float):
        """收益标准差"""
        data = self.forked_tree(submitted).tree
        m = numpy.sum(data[:, 0] * data[:, 1])
        s = numpy.sum((data[:, 0] - m) ** 2 * data[:, 1])
        return s ** 0.5

    def bench_mark(self):
        """基准收益分布"""
        return self.mean(self.default_quantity), self.std(self.default_quantity)

    def sharpe(self, submitted: float):
        return (self.mean(submitted) - self.mean(self.default_quantity)) / self.std(submitted)


if __name__ == "__main__":
    dpf = DiscreteForecast(
        [[90, 0.3], [100, 5], [110, 0.3]],
        [[80, 0.2], [100, 0.5], [120, 0.2]],
        [[190, 0.3], [200, 0.5], [210, 0.1]]
    )

    # print(dpf.forked_tree(100))

    p = YieldTree.product(
        [dpf.forked_tree(100), dpf.forked_tree(200), dpf.forked_tree(300)]
    )

    with open("yieldtree.json", "w") as f:
        f.write(json.dumps(p[:, [0, 1]].tolist()))

    # print(dpf)
    # act = 200
    # print(dpf.realtime_mean)
    # print(dpf.forked_tree(100))
    # print([
    #     float(dpf.sharpe(i)) for i in range(100, 300, 10)
    # ])

    # print([float(dpf.forked_tree(i / 10)) for i in range(1, 10)])
    # tree = dpf.forked_tree(0.1)
    # print([float(dpf.mean(i / 10)) for i in range(1, 100)])

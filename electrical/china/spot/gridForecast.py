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
from sklearn.cluster import KMeans


# 代码块

class YieldTree:
    """收益树"""

    def __init__(self, data):
        self.tree = numpy.array(data).astype(float)

    def __repr__(self):
        return str(self.tree.tolist())


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
        self.dayahead_cdf = numpy.array([
            [self.dayahead[i][0], numpy.sum(self.dayahead[:, 1][:i + 1])] for i in range(len(self.dayahead))
        ])
        self.realtime_cdf = numpy.array([
            [self.realtime[i][0], numpy.sum(self.realtime[:, 1][:i + 1])] for i in range(len(self.realtime))
        ])
        self.quantity_cdf = numpy.array([
            [self.quantity[i][0], numpy.sum(self.quantity[:, 1][:i + 1])] for i in range(len(self.quantity))
        ])

    def __repr__(self):
        return str({
            "dayahead": self.dayahead.tolist(),
            "realtime": self.realtime.tolist(),
            "quantity": self.quantity.tolist(),
            "default_quantity": self.default_quantity
        })

    def rvf(self, n: int = 1):
        """随机路径"""

        def ppf(cdf_list, x):
            if 0 < x < 1:
                for i, cdf in enumerate(cdf_list):
                    if i != len(cdf_list) - 1:
                        if x < cdf[1]:
                            return cdf[0]
                        else:
                            pass
                    else:
                        return cdf[0]
            else:
                return numpy.nan

        seed = numpy.random.uniform(0, 1, 3 * n)
        if n <= 1:
            return numpy.array(
                [ppf(self.dayahead_cdf, seed[0]), ppf(self.realtime_cdf, seed[1]), ppf(self.quantity_cdf, seed[2])])
        else:
            return numpy.array([
                numpy.array([
                    ppf(self.dayahead_cdf, seed[0 + i * 3]),
                    ppf(self.realtime_cdf, seed[1 + i * 3]),
                    ppf(self.quantity_cdf, seed[2 + i * 3])
                ]) for i in range(n)
            ])

    def forked_tree(self, submitted_quantity: float):
        """多叉树"""
        trade_yield = []
        for q in self.quantity:
            actual_quantity = q[0]
            dayahead_quantity = submitted_quantity
            realtime_quantity = actual_quantity - submitted_quantity
            dayahead_yield = []
            realtime_yield = []
            for dp in self.dayahead:
                actual_dayahead_price = dp[0]
                actual_dayahead_price_probability = dp[1]
                dayahead_yield.append([
                    dayahead_quantity * actual_dayahead_price,
                    actual_dayahead_price_probability,
                    actual_dayahead_price
                ])
            for rp in self.realtime:
                actual_realtime_price = rp[0]
                actual_realtime_price_probability = rp[1]
                realtime_yield.append([
                    realtime_quantity * actual_realtime_price,
                    actual_realtime_price_probability,
                    actual_realtime_price
                ])
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

    def bench_mark_tree(self, bench_quantity: float = None):
        """基准收益分布"""
        if bench_quantity is None:
            return self.forked_tree(self.default_quantity)
        else:
            return self.forked_tree(bench_quantity)

    def sharpe(self, submitted: float, bench_quantity: float = None):
        if bench_quantity is None:
            return (self.mean(submitted) - self.mean(self.default_quantity)) / self.std(submitted)
        else:
            return (self.mean(submitted) - self.mean(bench_quantity)) / self.std(submitted)


if __name__ == "__main__":
    dpf = DiscreteForecast(
        [[80, 0.3], [90, 0.3], [100, 0.5], [110, 0.3], [130, 0.2]],
        [[50, 0.3], [80, 0.2], [100, 0.5], [120, 0.2]],
        [[190, 0.3], [200, 0.5], [210, 0.1]]
    )

    print(dpf.rvf(1000).tolist())

    # p = YieldTree.product(
    #     [dpf.forked_tree(100), dpf.forked_tree(200), dpf.forked_tree(300)]
    # )
    #
    # with open("yieldtree.json", "w") as f:
    #     f.write(json.dumps(p[:, [0, 1]].tolist()))

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

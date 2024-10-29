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
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution

# 外部模块
import numpy


# 代码块

class ProbabilisticPoint:
    """基于概率的现货曲线点"""

    def __init__(self, dist: ABCDistribution):
        self.value = dist.mean()
        self.dist = dist.clone()

    def __repr__(self):
        return f"value:{self.value}, dist:{self.dist}"

    def cdf(self, first: float = 0.01, end: float = 0.99, n: int = 10):
        """离散的概率密度分布"""

        def standardization(xlist: list) -> numpy.ndarray:
            """标准化"""
            a = numpy.array(xlist).astype(float)
            p = 1 / numpy.sum(a[:, 1])
            a[:, 1] = a[:, 1] * p
            return a[numpy.argsort(a[:, 0])]

        curve = self.dist.pdf(first=first, end=end, num=n)
        return standardization(numpy.column_stack((curve.x, curve.y)))


class ProbabilisticDiscreteCurve:
    """基于概率分布的离散曲线"""

    def __init__(self, data: list[ABCDistribution]):
        self.original = [ProbabilisticPoint(i) for i in data]

    def cdf(self, first: float = 0.01, end: float = 0.99, n: int = 10) -> numpy.ndarray:
        """离散的概率密度曲线"""
        cdf_curve = []
        for i, d in enumerate(self.original):
            cdf = d.cdf(first=first, end=end, n=n)
            cdf_curve.append(cdf)
        return numpy.array(cdf_curve)

    def geo_cdf(self, first: float = 0.01, end: float = 0.99, n: int = 10) -> numpy.ndarray:
        """离散的概率密度曲线(几何)"""
        cdf_curve = []
        for i, d in enumerate(self.original):
            cdf = d.cdf(first=first, end=end, n=n)
            if i == 0:
                cdf_curve.append(cdf)
            else:
                delta = (cdf[:, 0] / self.original[i - 1].value) - 1
                cdf_curve.append(
                    numpy.column_stack((delta, cdf[:, 1]))
                )
        return numpy.array(cdf_curve)


class DiscreteSpot:
    """离散现货曲线"""

    def __init__(
            self,
            dayahead: ProbabilisticDiscreteCurve,
            realtime: ProbabilisticDiscreteCurve,
            quantity: ProbabilisticDiscreteCurve
    ):
        self.dayahead = copy.deepcopy(dayahead)
        self.realtime = copy.deepcopy(realtime)
        self.quantity = copy.deepcopy(quantity)

    def cdf(self, first: float = 0.01, end: float = 0.99, n: int = 10) -> numpy.ndarray:
        dayahead_cdf = self.dayahead.cdf(first, end, n)
        realtime_cdf = self.realtime.cdf(first, end, n)
        quantity_cdf = self.quantity.cdf(first, end, n)
        return numpy.array([
            dayahead_cdf,
            realtime_cdf,
            quantity_cdf
        ])

    def geo_cdf(self, first: float = 0.01, end: float = 0.99, n: int = 10) -> numpy.ndarray:
        dayahead_cdf = self.dayahead.geo_cdf(first, end, n)
        realtime_cdf = self.realtime.geo_cdf(first, end, n)
        quantity_cdf = self.quantity.geo_cdf(first, end, n)
        return numpy.array([
            dayahead_cdf,
            realtime_cdf,
            quantity_cdf
        ])

    def random_sample(self, first: float = 0.01, end: float = 0.99, n: int = 10):
        """随机样本"""

        def sample(cdf_list, x):
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

        data = self.cdf(first, end, n)
        seed = numpy.random.uniform(0, 1, 3)
        dayahead_cdf = data[0]
        realtime_cdf = data[1]
        quantity_cdf = data[2]


# class SpotPointDist:
#     """现货点"""
#
#     def __init__(self, dayahead: ABCDistribution, realtime: ABCDistribution, quantity: ABCDistribution):
#         """
#
#         @param dayahead: 日前价格预测概率分布
#         @param realtime: 实时价格预测概率分布
#         @param quantity: 实际处理预测概率分布
#         """
#         self.dayahead = dayahead.clone()
#         self.realtime = realtime.clone()
#         self.quantity = quantity.clone()
#
#     def pdf(self, n: int = 10, first: float = 0.01, end: float = 0.99):
#         dayahead = self.dayahead.pdf(first=first, end=end, num=n)
#         dayahead_array = numpy.column_stack((dayahead.x, dayahead.y))
#         realtime = self.realtime.pdf(first=first, end=end, num=n)
#         realtime_array = numpy.column_stack((realtime.x, realtime.y))
#         quantity = self.quantity.pdf(first=first, end=end, num=n)
#         quantity_array = numpy.column_stack((quantity.x, quantity.y))
#         return numpy.array([
#             dayahead_array, realtime_array, quantity_array
#         ])


if __name__ == "__main__":
    # sp = SpotPointDist(
    #     NormalDistribution(100, 10),
    #     NormalDistribution(101, 11),
    #     NormalDistribution(10, 1)
    # )
    # print(sp.pdf(10, 0.0001, 0.9999).tolist()[2])
    dayahead = ProbabilisticDiscreteCurve([
        NormalDistribution(100, 10),
        NormalDistribution(101, 11),
        NormalDistribution(103, 12),
        NormalDistribution(102, 13),
    ])

    realtime = ProbabilisticDiscreteCurve([
        NormalDistribution(101, 10),
        NormalDistribution(101, 11),
        NormalDistribution(102, 12),
        NormalDistribution(105, 13),
    ])

    quantity = ProbabilisticDiscreteCurve([
        NormalDistribution(101, 10),
        NormalDistribution(101, 11),
        NormalDistribution(102, 12),
        NormalDistribution(105, 13),
    ])

    spot = DiscreteSpot(dayahead, realtime, quantity)

    print(spot.geo_cdf().tolist())

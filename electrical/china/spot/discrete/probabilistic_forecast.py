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

def get_sample(cdf_list, x):
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


class ProbabilisticPoint:
    """基于概率的现货曲线点"""

    def __init__(self, dist: ABCDistribution):
        self.value = dist.mean()
        self.dist = dist.clone()

    def __repr__(self):
        return f"value:{self.value}, dist:{self.dist}"

    def cdf(self, first: float = 0.01, end: float = 0.99, n: int = 10):
        """离散的概率密度分布"""
        curve = self.dist.cdf(first=first, end=end, num=n)
        return numpy.column_stack((curve.x, curve.y))


class ProbabilisticDiscreteCurve:
    """基于概率分布的离散曲线"""

    def __init__(self, data: list[ABCDistribution], min: float = 0, max: float = None):
        self.original = [ProbabilisticPoint(i) for i in data]
        self.len = len(self.original)
        self.min = min
        self.max = max

    def put_in_range(self, x: float):
        """将随机变量置于曲线可用范围内"""
        if self.min and self.max is None:
            return x
        elif self.min is None:
            return min(self.max, x)
        elif self.max is None:
            return max(self.min, x)
        else:
            return min(max(self.min, x), self.max)

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
            this_cdf = d.cdf(first=first, end=end, n=n)
            if i == 0:
                cdf_curve.append(this_cdf)
            else:
                delta = (this_cdf[:, 0] / self.original[i - 1].value) - 1
                cdf_curve.append(
                    numpy.column_stack((delta, this_cdf[:, 1]))
                )
        return numpy.array(cdf_curve)

    def random_sample(self, first: float = 0.01, end: float = 0.99, n: int = 10, eps: int = 1,
                      geo: bool = False) -> numpy.ndarray:
        """生成随机样本"""
        if geo is False:
            cdf_list = self.cdf(first, end, n)
        else:
            cdf_list = self.geo_cdf(first, end, n)
        sample_list = []
        for _ in range(eps):
            point_sample = []
            for l in range(self.len):
                seed = numpy.random.uniform(0, 1)
                if geo is False:
                    s = self.put_in_range(get_sample(cdf_list[l], seed))
                else:
                    s = get_sample(cdf_list[l], seed)
                point_sample.append(s)
            sample_list.append(point_sample)
        return numpy.array(sample_list)

    def geo_random_sample(self, first: float = 0.01, end: float = 0.99, n: int = 10, eps: int = 1) -> numpy.ndarray:
        samples = self.random_sample(first, end, n, eps, True)
        sample_list = []
        for i, sample_row in enumerate(samples):
            point_sample_list = []
            for j, sample_point in enumerate(sample_row):
                if j == 0:
                    s = self.put_in_range(sample_point)
                else:
                    s = self.put_in_range(
                        point_sample_list[-1] * (1 + sample_point)
                    )
                point_sample_list.append(s)
            sample_list.append(point_sample_list)
        return numpy.array(sample_list)


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

    def random_sample(self, first: float = 0.01, end: float = 0.99, n: int = 10, eps: int = 1,
                      geo: bool = False) -> numpy.ndarray:
        """随机样本"""
        # if geo is False:
        #     data = self.cdf(first, end, n)
        # else:
        #     data = self.geo_cdf(first, end, n)
        # dayahead_cdf = data[0]
        # realtime_cdf = data[1]
        # quantity_cdf = data[2]
        # sample_list = []
        # for i in range(eps):
        #     point_sample = []
        #     for l in range(len(dayahead_cdf)):
        #         seed = numpy.random.uniform(0, 1, 3)
        #         point_sample.append([
        #             get_sample(dayahead_cdf[l], seed[0]),
        #             get_sample(realtime_cdf[l], seed[1]),
        #             get_sample(quantity_cdf[l], seed[2])
        #         ])
        #     sample_list.append(point_sample)
        # return numpy.array(sample_list)
        sample_list = []
        dayahead_sample = self.dayahead.random_sample(first, end, n, eps, geo)
        realtime_sample = self.realtime.random_sample(first, end, n, eps, geo)
        quantity_sampe = self.quantity.random_sample(first, end, n, eps, geo)
        for i in range(eps):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sampe[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list)

    def geo_random_sample(self, first: float = 0.01, end: float = 0.99, n: int = 10, eps: int = 1) -> numpy.ndarray:
        """几何随机样本"""
        # data = self.random_sample(first, end, n, eps, True)
        # sample_list = []
        # for i, sample_row in enumerate(data):
        #     point_sample_list = []
        #     for j, sample_point in enumerate(sample_row):
        #         if j == 0:
        #             point_sample_list.append(sample_point)
        #         else:
        #             point_sample_list.append([
        #                 point_sample_list[-1][k] * (1 + sample_point[k]) for k in range(3)
        #             ])
        #     sample_list.append(point_sample_list)
        # return numpy.array(sample_list)
        sample_list = []
        dayahead_sample = self.dayahead.geo_random_sample(first, end, n, eps)
        realtime_sample = self.realtime.geo_random_sample(first, end, n, eps)
        quantity_sampe = self.quantity.geo_random_sample(first, end, n, eps)
        for i in range(eps):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sampe[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list)


if __name__ == "__main__":
    dayahead = ProbabilisticDiscreteCurve([
        NormalDistribution(100, 10 / 2),
        NormalDistribution(101, 11 / 2),
        NormalDistribution(103, 12 / 2),
        NormalDistribution(102, 13 / 2),
    ])

    realtime = ProbabilisticDiscreteCurve([
        NormalDistribution(101, 10),
        NormalDistribution(101, 11),
        NormalDistribution(102, 12),
        NormalDistribution(105, 13),
    ])

    quantity = ProbabilisticDiscreteCurve([
        NormalDistribution(10, 1),
        NormalDistribution(9, 3),
        NormalDistribution(8, 1),
        NormalDistribution(7, 2),
    ])

    spot = DiscreteSpot(dayahead, realtime, quantity)

    print(spot.random_sample(n=100, eps=100).tolist())
    # seeds = numpy.random.uniform(0, 1, 100)
    # print(seeds.tolist())
    # print([get_sample(spot.dayahead.cdf(n=10)[0], i) for i in seeds])
    # # print(spot.dayahead.cdf(n=10)[0].tolist())

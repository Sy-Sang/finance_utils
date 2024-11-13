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
from data_utils.stochastic_utils.random_process.correlatedRandom import correlated_series, random_correlated_series
from data_utils.solve_utils.equationNSolve import gradient_descent
from easy_utils.number_utils.number_utils import EasyFloat
from easy_utils.obj_utils.enumerable_utils import flatten

from finance_utils.electrical.china.spot.rule.recycle import Recycle, AnarchismRecycle, SampleRecycle
from finance_utils.electrical.china.spot.rule.settlement import province_new_energy_with_recycle

# 外部模块
import numpy
from scipy.optimize import differential_evolution

# 代码块

Epsilon = numpy.finfo(float).eps


class ForecastPoint:
    """预测点"""

    def __init__(self, dist: ABCDistribution):
        self.value = dist.mean()
        self.dist = dist.clone()

    def __repr__(self):
        return f"value:{self.value}, dist:{self.dist}"

    def cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10) -> numpy.ndarray:
        """离散的概率密度分布"""
        curve = self.dist.cdf(first=first, end=end, num=num)
        return numpy.column_stack((curve.x, curve.y))

    def rvf(self, num: int = 10):
        """生成随机数"""
        r = self.dist.rvf(num)
        curve = numpy.array([
            [
                x, self.dist.cdf(x)
            ] for x in r
        ]).astype(float)
        sort_index = numpy.argsort(curve[:, 0])
        return curve[sort_index]


class ForecastCurve:
    """预测曲线"""

    def __init__(self, data: list[ABCDistribution], domain_min: float = 0, domain_max: float = None):
        self.original = [ForecastPoint(i) for i in data]
        self.value_list = [i.value for i in self.original]
        self.len = len(self.original)
        self.domain_min = domain_min
        self.domain_max = domain_max

    def __repr__(self):
        return str(f"{self.original}")

    def cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10, use_random=False) -> numpy.ndarray:
        """离散的概率密度曲线"""
        cdf_curve = []
        for i, d in enumerate(self.original):
            if use_random is False:
                cdf = d.cdf(first=first, end=end, num=num)
            else:
                cdf = d.rvf(num=num)
            cdf_curve.append(cdf)
        return numpy.array(cdf_curve).astype(float)

    def random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10,
                      use_random=False) -> numpy.ndarray:
        """简单随机样本"""
        sample_list = []
        data = self.cdf(first, end, num, use_random)
        for i, d in enumerate(data):
            numpy.random.shuffle(d)
            sample_list.append(
                EasyFloat.put_in_range(self.domain_min, self.domain_max, *d[:, 0])
            )
        return numpy.array(sample_list).T.astype(float)

    def diff_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10,
                           use_random=False) -> numpy.ndarray:
        """差分随机样本"""
        sample_list = []
        diff_dist_list = [None]
        for i in range(1, self.len):
            diff = self.original[i].dist.cdf(first=first, end=end, num=num).x - self.original[i - 1].value
            diff_dist = HistogramDist(diff)
            diff_dist_list.append(diff_dist)

        for i in range(self.len):
            if i == 0:
                if use_random is True:
                    random_slice = self.original[i].dist.rvf(num)
                else:
                    random_slice = self.original[i].dist.cdf(num=num).x
                    numpy.random.shuffle(random_slice)
            else:
                if use_random is True:
                    diff_slice = diff_dist_list[i].rvf(num)
                else:
                    diff_slice = diff_dist_list[i].cdf(num=num).x
                    numpy.random.shuffle(diff_slice)
                random_slice = diff_slice + sample_list[-1]
            slice_array = numpy.array(
                EasyFloat.put_in_range(self.domain_min, self.domain_max, *random_slice)
            ).astype(float)
            sample_list.append(slice_array)
        return numpy.array(sample_list).T.astype(float)

    def geo_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, num: int = 10,
                          use_random=False) -> numpy.ndarray:
        """几何随机样本"""
        sample_list = []
        log_diff_dist_list = [None]
        for i in range(1, self.len):
            log_diff = self.original[i].dist.cdf(first=first, end=end, num=num).x / self.original[i - 1].value - 1
            log_diff_dist = HistogramDist(log_diff)
            log_diff_dist_list.append(log_diff_dist)

        for i in range(self.len):
            if i == 0:
                if use_random is True:
                    random_slice = self.original[i].dist.rvf(num)
                else:
                    random_slice = self.original[i].dist.cdf(num=num).x
                    numpy.random.shuffle(random_slice)
            else:
                if use_random is True:
                    diff_slice = log_diff_dist_list[i].rvf(num)
                else:
                    diff_slice = log_diff_dist_list[i].cdf(num=num).x
                    numpy.random.shuffle(diff_slice)
                random_slice = diff_slice * (1 + sample_list[-1])

            slice_array = numpy.array(
                EasyFloat.put_in_range(self.domain_min, self.domain_max, *random_slice)
            ).astype(float)
            sample_list.append(slice_array)
        return numpy.array(sample_list).T.astype(float)

    def self_related_random_sample(self, pearson: list[float], first: float = Epsilon, end: float = 1 - Epsilon,
                                   num: int = 10,
                                   use_random=False) -> numpy.ndarray:
        """自相关的随机样本"""
        data = random_correlated_series([p.dist for p in self.original], pearson, num=num)
        return data

    def noised(self, noise_list: list) -> Self:
        """叠加噪音"""
        dist_list = []
        for i, n in enumerate(noise_list):
            if isinstance(n, ABCDistribution):
                data = self.original[i].dist.ppf(num=100).y + n.ppf(num=100).y
            else:
                data = self.original[i].dist.rvf(len(n)) + numpy.array(n).astype(float)
            dist_list.append(
                HistogramDist(data)
            )
        return self.__class__(dist_list, domain_min=self.domain_min, domain_max=self.domain_max)


if __name__ == "__main__":
    fc = ForecastCurve([NormalDistribution(0, 1), NormalDistribution(0, 1), NormalDistribution(0, 1)],
                       domain_min=None, domain_max=None)
    # print(fc.diff_random_sample(num=100, use_random=True).tolist())
    # print(fc.random_sample(num=100, use_random=True).tolist())
    # print(fc.geo_random_sample(num=100, use_random=True).tolist())
    print(fc.self_related_random_sample([0,0.5,-0.5]))

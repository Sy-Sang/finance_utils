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

    def cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10):
        """离散的概率密度分布"""
        curve = self.dist.cdf(first=first, end=end, num=n)
        return numpy.column_stack((curve.x, curve.y))

    def random_cdf(self, n: int = 10):
        """生成随机变量的cdf曲线"""
        r = self.dist.rvf(n)
        curve = numpy.array([
            [
                x, self.dist.cdf(x)
            ] for x in r
        ]).astype(float)
        sort_index = numpy.argsort(curve[:, 0])
        return curve[sort_index]


class ProbabilisticDiscreteCurve:
    """基于概率分布的离散曲线"""

    def __init__(self, data: list[ABCDistribution], min: float = 0, max: float = None):
        self.original = [ProbabilisticPoint(i) for i in data]
        self.value_list = [i.value for i in self.original]
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

    def cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, use_random=False) -> numpy.ndarray:
        """离散的概率密度曲线"""
        cdf_curve = []
        for i, d in enumerate(self.original):
            if use_random is False:
                cdf = d.cdf(first=first, end=end, n=n)
            else:
                cdf = d.random_cdf(n=n)
            cdf_curve.append(cdf)
        return numpy.array(cdf_curve).astype(float)

    def geo_cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, use_random=False) -> numpy.ndarray:
        """离散的概率密度曲线(几何)"""
        cdf_curve = []
        for i, d in enumerate(self.original):
            if use_random is False:
                this_cdf = d.cdf(first=first, end=end, n=n)
            else:
                this_cdf = d.random_cdf(n=n)

            if i == 0:
                cdf_curve.append(this_cdf)
            else:
                delta = (this_cdf[:, 0] / self.original[i - 1].value) - 1
                cdf_curve.append(
                    numpy.column_stack((delta, this_cdf[:, 1]))
                )
        return numpy.array(cdf_curve).astype(float)

    def random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, epoch: int = 1,
                      geo: bool = False, use_random=False) -> numpy.ndarray:
        """生成随机样本"""
        if geo is False:
            cdf_list = self.cdf(first, end, n, use_random=use_random)
        else:
            cdf_list = self.geo_cdf(first, end, n, use_random=use_random)
        sample_list = []
        for _ in range(epoch):
            point_sample = []
            for l in range(self.len):
                seed = numpy.random.uniform(0, 1)
                if geo is False:
                    s = self.put_in_range(get_sample(cdf_list[l], seed))
                else:
                    s = get_sample(cdf_list[l], seed)
                point_sample.append(s)
            sample_list.append(point_sample)
        return numpy.array(sample_list).astype(float)

    def geo_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, epoch: int = 1,
                          use_random=False) -> numpy.ndarray:
        samples = self.random_sample(first, end, n, epoch, True, use_random=use_random)
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
        return numpy.array(sample_list).astype(float)


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

    def cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, use_random=False) -> numpy.ndarray:
        dayahead_cdf = self.dayahead.cdf(first, end, n, use_random=use_random)
        realtime_cdf = self.realtime.cdf(first, end, n, use_random=use_random)
        quantity_cdf = self.quantity.cdf(first, end, n, use_random=use_random)
        return numpy.array([
            dayahead_cdf,
            realtime_cdf,
            quantity_cdf
        ]).astype(float)

    def geo_cdf(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, use_random=False) -> numpy.ndarray:
        dayahead_cdf = self.dayahead.geo_cdf(first, end, n, use_random=use_random)
        realtime_cdf = self.realtime.geo_cdf(first, end, n, use_random=use_random)
        quantity_cdf = self.quantity.geo_cdf(first, end, n, use_random=use_random)
        return numpy.array([
            dayahead_cdf,
            realtime_cdf,
            quantity_cdf
        ]).astype(float)

    def random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, epoch: int = 1,
                      geo: bool = False, use_random=False) -> numpy.ndarray:
        """随机样本"""
        sample_list = []
        dayahead_sample = self.dayahead.random_sample(first, end, n, epoch, geo, use_random=use_random)
        realtime_sample = self.realtime.random_sample(first, end, n, epoch, geo, use_random=use_random)
        quantity_sampe = self.quantity.random_sample(first, end, n, epoch, geo, use_random=use_random)
        for i in range(epoch):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sampe[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list).astype(float)

    def geo_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, epoch: int = 1,
                          use_random=False) -> numpy.ndarray:
        """几何随机样本"""
        sample_list = []
        dayahead_sample = self.dayahead.geo_random_sample(first, end, n, epoch, use_random=use_random)
        realtime_sample = self.realtime.geo_random_sample(first, end, n, epoch, use_random=use_random)
        quantity_sampe = self.quantity.geo_random_sample(first, end, n, epoch, use_random=use_random)
        for i in range(epoch):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sampe[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list).astype(float)

    def differential_evolution__search(
            self, submitted_list: list[float], recycle: Type[Recycle] = None,
            delta_min: float = -10, delta_max: float = 10,
            submitted_min: float = 0, submitted_max: float = None,
            *args, **kwargs
    ):
        """差分进化搜索"""

        def target(xlist):
            xlist = EasyFloat.put_in_range(submitted_min, submitted_max, *xlist)
            trade_yield = province_new_energy_with_recycle(
                self.dayahead.value_list,
                self.realtime.value_list,
                self.quantity.value_list,
                xlist,
                recycle=recycle,
                *args, **kwargs
            )
            return -1 * trade_yield

        bounds = [
            [
                EasyFloat.put_in_range(submitted_min, submitted_max, i + delta_min),
                EasyFloat.put_in_range(submitted_min, submitted_max, i + delta_max)
            ] for i in submitted_list
        ]

        result = differential_evolution(target, bounds)
        return result.x, -result.fun

    def gradient_descent_search(
            self,
            submitted_list: list[float], target_yield: float, recycle: Type[Recycle] = None,
            submitted_min: float = 0, submitted_max: float = None,
            eps: float = 0.1, lr: float = 0.1, epoch: int = 200,
            *args, **kwargs
    ):
        """梯度下降搜索最优submitted quantity"""

        def target(xlist):
            xlist = EasyFloat.put_in_range(submitted_min, submitted_max, *xlist)
            return province_new_energy_with_recycle(
                self.dayahead.value_list,
                self.realtime.value_list,
                self.quantity.value_list,
                xlist,
                recycle=recycle,
                *args, **kwargs
            )

        gd = gradient_descent(
            f=target,
            x=submitted_list,
            y=[target_yield],
            eps=eps,
            lr=lr,
            epoch=epoch,
            print_loss=True
        )

        return EasyFloat.put_in_range(
            submitted_min, submitted_max,
            *gd[0]
        ), gd[1]


if __name__ == "__main__":
    dayahead = ProbabilisticDiscreteCurve([
        NormalDistribution(100, 10),
        NormalDistribution(101, 11),
        NormalDistribution(103, 12),
        NormalDistribution(102, 13),
    ])

    realtime = ProbabilisticDiscreteCurve([
        NormalDistribution(100 * 1.5, 20),
        NormalDistribution(101, 21),
        NormalDistribution(103 * 0.5, 22),
        NormalDistribution(102, 23),
    ])

    quantity = ProbabilisticDiscreteCurve([
        NormalDistribution(10, 1),
        NormalDistribution(9, 3),
        NormalDistribution(8, 1),
        NormalDistribution(7, 2),
    ])

    spot = DiscreteSpot(dayahead, realtime, quantity)
    print(spot.differential_evolution__search(
        spot.quantity.value_list,
        delta_min=-20, delta_max=20,
        submitted_min=0, submitted_max=20,
        recycle=SampleRecycle,
        trigger_rate=0.05,
        punishment_rate=0.5,
    ))

    print(
        province_new_energy_with_recycle(
            spot.dayahead.value_list,
            spot.realtime.value_list,
            spot.quantity.value_list,
            spot.quantity.value_list,
            recycle=SampleRecycle,
            trigger_rate=0.05,
            punishment_rate=0.5,
        )
    )

    # print(spot.random_sample(n=100, epoch=100).tolist())
    # seeds = numpy.random.uniform(0, 1, 100)
    # print(seeds.tolist())
    # print([get_sample(spot.dayahead.cdf(n=10)[0], i) for i in seeds])
    # # print(spot.dayahead.cdf(n=10)[0].tolist())

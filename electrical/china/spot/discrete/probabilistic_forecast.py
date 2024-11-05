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

    def __init__(self, data: list[ABCDistribution], domain_min: float = 0, domain_max: float = None):
        self.original = [ProbabilisticPoint(i) for i in data]
        self.value_list = [i.value for i in self.original]
        self.len = len(self.original)
        self.domain_min = domain_min
        self.domain_max = domain_max

    def __repr__(self):
        return str(f"{self.original}")

    def put_in_range(self, x: float):
        """将随机变量置于曲线可用范围内"""
        if self.domain_min and self.domain_max is None:
            return x
        elif self.domain_min is None:
            return min(self.domain_max, x)
        elif self.domain_max is None:
            return max(self.domain_min, x)
        else:
            return min(max(self.domain_min, x), self.domain_max)

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
        """几何随机样本"""
        samples = self.random_sample(first, end, n, epoch, True, use_random=use_random)
        sample_list = []
        for i, sample_row in enumerate(samples):
            point_sample_list = []
            for j, sample_point in enumerate(sample_row):
                if j == 0:
                    s = sample_point
                else:
                    s = point_sample_list[-1] * (1 + sample_point)
                point_sample_list.append(s)
            sample_list.append([self.put_in_range(p) for p in point_sample_list])
        return numpy.array(sample_list).astype(float)

    def diff_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10,
                           p: float = 0.5, epoch: int = 1) -> numpy.ndarray:
        """差分随机样本"""
        sample_list = []
        diff_dist_list = [None]
        for i in range(1, self.len):
            diff = (self.original[i].dist.cdf(first=first, end=end, num=n).x
                    - self.original[i - 1].dist.cdf(first=first, end=end, num=n).x)
            diff_dist = HistogramDist(diff)
            diff_dist_list.append(diff_dist)

        for _ in range(epoch):
            epoch_sample_list = []
            for i in range(self.len):
                rp = numpy.random.uniform(Epsilon, 1 - Epsilon, 3)
                if i == 0:
                    rv = self.original[i].dist.ppf(rp[0])
                else:
                    d = diff_dist_list[i].ppf(rp[0])
                    rv_s = [self.original[i].dist.ppf(rp[0]), 1 - p]
                    rv_d = [d + epoch_sample_list[-1], p]
                    rv_array = numpy.array([rv_s, rv_d])
                    rv_cdf = rv_array[numpy.argsort(rv_array[:, 0])]
                    rv = get_sample(rv_cdf, rp[2])
                epoch_sample_list.append(rv)
            sample_list.append([self.put_in_range(s) for s in epoch_sample_list])
        return numpy.array(sample_list).astype(float)


class SpotNoise:
    """现货市场噪音"""

    def __init__(
            self,
            dayahead: list[ABCDistribution],
            realtime: list[ABCDistribution],
            quantity: list[ABCDistribution]
    ):
        self.dayahead = dayahead
        self.realtime = realtime
        self.quantity = quantity

    def __call__(self, *args, **kwargs):
        return [
            [i.rvf() for i in self.dayahead],
            [i.rvf() for i in self.realtime],
            [i.rvf() for i in self.quantity]
        ]


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

    def __repr__(self):
        return str(f"{self.dayahead},{self.realtime},{self.quantity}")

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
        quantity_sample = self.quantity.random_sample(first, end, n, epoch, geo, use_random=use_random)
        for i in range(epoch):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sample[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list).astype(float)

    def geo_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, epoch: int = 1,
                          use_random=False) -> numpy.ndarray:
        """几何随机样本"""
        sample_list = []
        dayahead_sample = self.dayahead.geo_random_sample(first, end, n, epoch, use_random=use_random)
        realtime_sample = self.realtime.geo_random_sample(first, end, n, epoch, use_random=use_random)
        quantity_sample = self.quantity.geo_random_sample(first, end, n, epoch, use_random=use_random)
        for i in range(epoch):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sample[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list).astype(float)

    def diff_random_sample(self, first: float = Epsilon, end: float = 1 - Epsilon, n: int = 10, epoch: int = 1,
                           p=0.5, use_random=False) -> numpy.ndarray:
        """差分随机样本"""
        sample_list = []
        dayahead_sample = self.dayahead.diff_random_sample(first, end, n, p, epoch)
        realtime_sample = self.realtime.diff_random_sample(first, end, n, p, epoch)
        quantity_sample = self.quantity.diff_random_sample(first, end, n, p, epoch)
        for i in range(epoch):
            row = numpy.column_stack((
                dayahead_sample[i],
                realtime_sample[i],
                quantity_sample[i]
            ))
            sample_list.append(row)
        return numpy.array(sample_list).astype(float)

    def noised_random_sample(self, data, noise: SpotNoise) -> numpy.ndarray:
        """添加噪音"""
        sample_list = []
        for i, d in enumerate(data):
            n = noise()
            noised_dayahead = [self.dayahead.put_in_range(x) for x in d[:, 0] + n[0]]
            noised_realtime = [self.realtime.put_in_range(x) for x in d[:, 1] + n[1]]
            noised_quantity = [self.quantity.put_in_range(x) for x in d[:, 2] + n[2]]

            row = numpy.column_stack((
                noised_dayahead,
                noised_realtime,
                noised_quantity
            ))
            sample_list.append(row)
        return numpy.array(sample_list).astype(float)

    def noised_spot(self, data, noise: SpotNoise) -> Self:
        """添加噪音后的现货对象"""
        noised_sample = self.noised_random_sample(data, noise)
        dayahead_list = []
        realtime_list = []
        quantity_list = []
        dayahead_dist_list = []
        realtime_dist_list = []
        quantity_dist_list = []
        for i, s in enumerate(noised_sample):
            row_dayahead = s[:, 0]
            row_realtime = s[:, 1]
            row_quantity = s[:, 2]
            for j in range(len(row_dayahead)):
                if i == 0:
                    dayahead_list.append([row_dayahead[j]])
                    realtime_list.append([row_realtime[j]])
                    quantity_list.append([row_quantity[j]])
                else:
                    dayahead_list[j].append(row_dayahead[j])
                    realtime_list[j].append(row_realtime[j])
                    quantity_list[j].append(row_quantity[j])

        for i in range(len(dayahead_list)):
            dayahead_dist_list.append(HistogramDist(dayahead_list[i]))
            realtime_dist_list.append(HistogramDist(realtime_list[i]))
            quantity_dist_list.append(HistogramDist(quantity_list[i]))

        return type(self)(
            ProbabilisticDiscreteCurve(dayahead_dist_list, self.dayahead.domain_min, self.dayahead.domain_max),
            ProbabilisticDiscreteCurve(realtime_dist_list, self.realtime.domain_min, self.realtime.domain_max),
            ProbabilisticDiscreteCurve(quantity_dist_list, self.quantity.domain_min, self.quantity.domain_max)
        )

    def value_list_yield(self, xlist, recycle: Type[Recycle] = None, submitted_min: float = 0,
                         submitted_max: float = None,
                         *args, **kwargs):
        """不考虑随机性的收益"""
        xlist = EasyFloat.put_in_range(submitted_min, submitted_max, *xlist)
        trade_yield = province_new_energy_with_recycle(
            self.dayahead.value_list,
            self.realtime.value_list,
            self.quantity.value_list,
            xlist,
            recycle=recycle,
            *args, **kwargs
        )
        return trade_yield

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

    # s = quantity.diff_random_sample(epoch=10)
    # print(quantity.add_noise(s, [NormalDistribution(0, 10)] * 4))

    spot = DiscreteSpot(dayahead, realtime, quantity)
    samples = spot.diff_random_sample(n=100, epoch=100)
    noise = SpotNoise(
        [NormalDistribution(0, 1)] * 4,
        [NormalDistribution(0, 1)] * 4,
        [NormalDistribution(0, 1)] * 4
    )

    # print(spot.noised_random_sample(samples, noise))

    print(spot.noised_spot(samples, noise))
    #
    # print(
    #     province_new_energy_with_recycle(
    #         spot.dayahead.value_list,
    #         spot.realtime.value_list,
    #         spot.quantity.value_list,
    #         spot.quantity.value_list,
    #         recycle=SampleRecycle,
    #         trigger_rate=0.05,
    #         punishment_rate=0.5,
    #     )
    # )

    # print(spot.random_sample(n=100, epoch=100).tolist())
    # seeds = numpy.random.uniform(0, 1, 100)
    # print(seeds.tolist())
    # print([get_sample(spot.dayahead.cdf(n=10)[0], i) for i in seeds])
    # # print(spot.dayahead.cdf(n=10)[0].tolist())

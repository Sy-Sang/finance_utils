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
from abc import ABC, abstractmethod
from itertools import product

# 项目模块
from easy_utils.number_utils.number_utils import EasyFloat
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution

from finance_utils.electrical.china.spot.rule.recycle import Recycle, SampleRecycle
from finance_utils.electrical.china.spot.rule.settlement import province_new_energy, province_new_energy_with_recycle

# 外部模块
import numpy
from scipy.optimize import differential_evolution


# 代码块

class TestBackCurve:
    """回测曲线"""

    def __init__(
            self,
            curve_detail: numpy.ndarray,
            yield_curve: numpy.ndarray,
            punishment_curve: numpy.ndarray
    ):
        self.curve_detail = curve_detail
        self.yield_curve = yield_curve
        self.punishment_curve = punishment_curve

    def mean(self):
        return numpy.mean(self.yield_curve)

    def median(self):
        return numpy.median(self.yield_curve)

    def std(self):
        return numpy.std(self.yield_curve, ddof=1)


class TestBack:
    """回测"""

    def __init__(self, samples):
        self.samples = numpy.array(samples).astype(float)
        self.size = (len(self.samples), len(self.samples[0]))

    def testback_yield(self, submitted_list: list[float], recycle: Recycle = None, *args, **kwargs) -> TestBackCurve:
        """回测收益曲线"""
        testback_yield_list = []
        total_yield_list = []
        punishment_list = []
        for i, sample_list in enumerate(self.samples):
            trade_yield_list = []
            for j, sample_point in enumerate(sample_list):
                trade_yield = province_new_energy(
                    sample_point[0], sample_point[1], sample_point[2], submitted_list[j]
                )
                trade_yield_list.append(trade_yield)
            total_yield = numpy.sum(trade_yield_list)

            if recycle is None:
                recycle_f = SampleRecycle(sample_list, submitted_list, total_yield)
            else:
                recycle_f = recycle(sample_list, submitted_list, total_yield)
            punishment = recycle_f(*args, **kwargs)

            total_yield_list.append(
                total_yield - punishment
            )
            testback_yield_list.append(trade_yield_list)
            punishment_list.append(punishment)

        return TestBackCurve(
            numpy.array(testback_yield_list).astype(float),
            numpy.array(total_yield_list).astype(float),
            numpy.array(punishment_list).astype(float)
        )

    def __call__(self, submitted_list: list[float], recycle: Recycle = None, *args, **kwargs) -> numpy.ndarray:
        testback_yield_list = []
        for i, sample_list in enumerate(self.samples):
            sample_array = numpy.array(sample_list).astype(float)
            testback_yield_list.append(
                province_new_energy_with_recycle(
                    sample_array[:, 0],
                    sample_array[:, 1],
                    sample_array[:, 2],
                    submitted_list,
                    recycle,
                    *args, **kwargs
                )
            )
        return numpy.array(testback_yield_list).astype(float)

    def random_grid_search(
            self, submitted_list: list[float], recycle: Recycle = None, delta_min: float = -10,
            delta_max: float = 10, epoch: int = 100, submitted_min: float = 0, submitted_max: float = None,
            *args, **kwargs) -> tuple:
        """随机网格搜索"""
        search_yield = []
        search_list = []
        for s in range(epoch):
            submitted = [0] * len(submitted_list)
            for i in range(len(submitted_list)):
                if s == 0:
                    submitted[i] = submitted_list[i]
                else:
                    submitted[i] = submitted_list[i] + numpy.random.uniform(delta_min, delta_max)
                submitted[i] = EasyFloat.put_in_range(submitted_min, submitted_max, submitted[i])
            testback = self.testback_yield(submitted, recycle, *args, **kwargs)
            search_yield.append(testback.mean())
            search_list.append(submitted)
        sort_index = numpy.argsort(search_yield)
        return search_list[sort_index[-1]], search_yield[sort_index[-1]]

    def continuous_random_grid_search(
            self, submitted_list: list[float], recycle: Recycle = None, delta_min: float = -10,
            delta_max: float = 10, epoch: int = 100, rounds: int = 10, submitted_min: float = 0,
            submitted_max: float = None,
            *args, **kwargs
    ) -> list:
        """持续的随机格点搜索"""
        testback_list = []
        for _ in range(rounds):
            testback = self.random_grid_search(
                submitted_list, recycle, delta_min, delta_max, epoch, submitted_min, submitted_max, *args, **kwargs
            )
            submitted_list = testback[0]
            testback_list.append(testback)
        return testback_list

    def differential_evolution__search(
            self, recycle: Type[Recycle] = None,
            submitted_min: float = 0, submitted_max: float = None,
            *args, **kwargs
    ):
        """差分进化"""

        def target(xlist):
            """目标函数"""
            xlist = EasyFloat.put_in_range(submitted_min, submitted_max, *xlist)
            trade_yield = self.testback_yield(
                submitted_list=xlist,
                recycle=recycle,
                *args, **kwargs
            ).mean()
            return -1 * trade_yield

        bounds = [[submitted_min, submitted_max] for _ in range(self.size[1])]

        result = differential_evolution(target, bounds)
        return result.x, -result.fun


if __name__ == "__main__":
    print(province_new_energy(100, 180, 10, 5))

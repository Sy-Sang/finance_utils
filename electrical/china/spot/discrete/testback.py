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
from abc import ABC, abstractmethod
from itertools import product

# 项目模块
from easy_utils.number_utils.number_utils import EasyFloat
from data_utils.solve_utils.equationNSolve import gradient_descent

# 外部模块
import numpy


# 代码块


class Recycle(ABC):
    """回收机制(接口)"""

    def __init__(self, spot_list, submit_list, benefits, *args, **kwargs):
        self.spot_list = spot_list
        self.submit_list = submit_list
        self.benefits = benefits

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


class AnarchismRecycle(Recycle):
    def __call__(self, *args, **kwargs):
        return 0


class SampleRecycle(Recycle):
    def __call__(self, trigger_rate=0.05, punishment_rate: float = 0.5, *args, **kwargs):
        punishment = 0
        spot_array = numpy.array(self.spot_list).astype(float)
        power_deviation = abs(numpy.sum(self.submit_list) / numpy.sum(spot_array[:, 2]) - 1)
        if power_deviation >= trigger_rate:
            punishment = max(0, self.benefits * punishment_rate)
        else:
            pass
        return punishment


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

    @classmethod
    def trade(cls, dayahead, realtime, realtime_quantity, submitted_quantity):
        """交易函数"""
        return float(
            dayahead * submitted_quantity + (realtime_quantity - submitted_quantity) * realtime
        )

    def testback_yield(self, submitted_list: list[float], f: Recycle = None, *args, **kwargs) -> TestBackCurve:
        """回测收益曲线"""
        testback_yield_list = []
        total_yield_list = []
        punishment_list = []
        for i, sample_list in enumerate(self.samples):
            trade_yield_list = []
            for j, sample_point in enumerate(sample_list):
                trade_yield = self.trade(
                    sample_point[0], sample_point[1], sample_point[2], submitted_list[j]
                )
                trade_yield_list.append(trade_yield)
            total_yield = numpy.sum(trade_yield_list)

            if f is None:
                recycle = SampleRecycle(sample_list, submitted_list, total_yield)
            else:
                recycle = f(sample_list, submitted_list, total_yield)
            punishment = recycle(*args, **kwargs)

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

    def grid_search(self, submitted_list: list[float], f: Recycle = None, submit_min: float = -10,
                    submit_max: float = 10, interval=1, *args, **kwargs) -> tuple:
        """网格搜索"""
        grid_delta = numpy.array(EasyFloat.frange(submit_min, submit_max, interval, True)).astype(float)
        grid_list = [grid_delta + submitted_list[i] for i in range(len(submitted_list))]
        grid = product(*grid_list)
        search_yield = []
        search_list = []
        for g in grid:
            testback = self.testback_yield(g, f, *args, **kwargs)
            search_yield.append(testback.mean())
            search_list.append(g)
        sort_index = numpy.argsort(search_yield)
        return numpy.array(search_list[sort_index[-1]]), search_yield[sort_index[-1]]

    def random_grid_search(
            self, submitted_list: list[float], recycle: Recycle = None, submit_min: float = -10,
            submit_max: float = 10, eps: int = 100, nonnegative: bool = True, *args, **kwargs) -> tuple:
        """随机网格搜索"""
        search_yield = []
        search_list = []
        for s in range(eps):
            submitted = [0] * len(submitted_list)
            for i in range(len(submitted_list)):
                if s == 0:
                    submitted[i] = submitted_list[i]
                else:
                    submitted[i] = submitted_list[i] + numpy.random.uniform(submit_min, submit_max)
                if nonnegative is True:
                    submitted[i] = max(0, submitted[i])
                else:
                    pass
            testback = self.testback_yield(submitted, recycle, *args, **kwargs)
            search_yield.append(testback.mean())
            search_list.append(submitted)
        sort_index = numpy.argsort(search_yield)
        return search_list[sort_index[-1]], search_yield[sort_index[-1]]

    def continuous_random_grid_search(
            self, submitted_list: list[float], f: Recycle = None, submit_min: float = -10,
            submit_max: float = 10, eps: int = 100, rounds: int = 10, nonnegative: bool = True,
            *args, **kwargs
    ) -> list:
        """持续的随机格点搜索"""
        testback_list = []
        for _ in range(rounds):
            testback = self.random_grid_search(
                submitted_list, f, submit_min, submit_max, eps, nonnegative, *args, **kwargs
            )
            submitted_list = testback[0]
            testback_list.append(testback)
        return testback_list


if __name__ == "__main__":
    print(TestBack.trade(100, 180, 10, 5))

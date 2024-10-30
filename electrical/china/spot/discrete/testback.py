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

# 项目模块

# 外部模块
import numpy


# 代码块


class Recycle(ABC):
    def __init__(self, spot_list, submit_list, benefits, *args, **kwargs):
        self.spot_list = spot_list
        self.submit_list = submit_list
        self.benefits = benefits

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class AnarchismRecycle(Recycle):
    def __call__(self, *args, **kwargs):
        return 0


class SampleRecycle(Recycle):
    def __call__(self, *args, **kwargs):
        punishment = 0
        spot_array = numpy.array(self.spot_list).astype(float)
        power_deviation = abs(numpy.sum(self.submit_list) / numpy.sum(spot_array[:, 2]) - 1)
        if power_deviation >= 0.05:
            punishment = max(0, self.benefits * 0.5)
        else:
            pass
        return punishment


class TestBackCurve:
    def __init__(self, curve_detail: numpy.ndarray, yield_curve: numpy.ndarray):
        self.curve_detail = curve_detail
        self.yield_curve = yield_curve


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

    def testback_yield(self, submitted_list: list[float], f: Recycle = None, *args, **kwargs):
        """回测收益曲线"""
        # if f is None:
        #     recycle = self.__sample_recycle
        # else:
        #     recycle = f
        testback_yield_list = []
        total_yield_list = []
        for i, sample_list in enumerate(self.samples):
            trade_yield_list = []
            for j, sample_point in enumerate(sample_list):
                trade_yield = self.trade(
                    sample_point[0], sample_point[1], sample_point[2], submitted_list[j]
                )
                trade_yield_list.append(trade_yield)
            total_yield = numpy.sum(trade_yield_list)
            testback_yield_list.append(trade_yield_list)

            if f is None:
                recycle = SampleRecycle(sample_list, submitted_list, total_yield)
            else:
                recycle = f(sample_list, submitted_list, total_yield)

            total_yield_list.append(
                total_yield - recycle(*args, **kwargs)
            )

        return TestBackCurve(
            numpy.array(testback_yield_list).astype(float), numpy.array(total_yield_list).astype(float)
        )


if __name__ == "__main__":
    print(TestBack.trade(100, 180, 10, 5))

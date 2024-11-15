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
from finance_utils.electrical2.spot.rule.recycle import Recycle, AnarchismRecycle, SampleRecycle
from data_utils.stochastic_utils.distributions.nonParametricDistribution import HistogramDist

# 外部模块
import numpy


# 代码块
class SettlementResult:
    """交易结果"""

    def __init__(self):
        self.point_yield = []
        self.punishment = []
        self.trade_yield = []

    def new_trade(self, trade_result):
        """添加交易结果"""
        self.point_yield.append(trade_result[0])
        self.punishment.append(float(trade_result[1]))
        # self = numpy.append(self, trade_result[2]).astype(float)
        self.trade_yield.append(float(trade_result[2]))

    def to_array(self):
        return numpy.array(self.trade_yield).astype(float)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.trade_yield):
            result = self.trade_yield[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def point_matrix(self):
        """逐点交易收益矩阵"""
        return numpy.array(self.point_yield).astype(float)

    def point_slice(self, n: int = 0):
        """切片"""
        data = numpy.array(self.point_yield).astype(float)
        return data[:, n]

    def point_slice_dist(self, n: int = 0):
        """切片分布"""
        return HistogramDist(self.point_slice(n))

    def yield_dist(self):
        """收益分布"""
        return HistogramDist(self.trade_yield)


def province_new_energy(dayahead: float, realtime: float, realtime_quantity: float, submitted_quantity: float):
    """交易结算函数"""
    return float(
        dayahead * submitted_quantity + (realtime_quantity - submitted_quantity) * realtime
    )


def province_new_energy_with_recycle(
        dayahead: Union[list, numpy.ndarray], realtime: Union[list, numpy.ndarray],
        realtime_quantity: Union[list, numpy.ndarray], submitted_quantity: Union[list, numpy.ndarray],
        recycle: Type[Recycle] = None, show_detail=False, *args, **kwargs
):
    """包含省新能源回收的结算"""
    point_yield = []
    for i in range(len(dayahead)):
        point_yield.append(
            province_new_energy(dayahead[i], realtime[i], realtime_quantity[i], submitted_quantity[i])
        )

    total_yield = numpy.sum(point_yield)
    spot_array = numpy.column_stack((dayahead, realtime, realtime_quantity)).astype(float)

    if recycle is None:
        recycle_f = SampleRecycle(spot_array, submitted_quantity, total_yield)
    else:
        recycle_f = recycle(spot_array, submitted_quantity, total_yield)

    punishment = recycle_f(*args, **kwargs)

    if show_detail is False:
        return total_yield - punishment
    else:
        return point_yield, punishment, total_yield - punishment


if __name__ == "__main__":
    pass

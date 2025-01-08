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
from data_utils.serial_utils.time_series import TimeSeries
from easy_datetime.timestamp import TimeStamp

from finance_utils.asset.base import *
from finance_utils.asset.spot.base import Spot
from finance_utils.trader.base import Trader

# 外部模块
import numpy

# 代码块

PathValue = namedtuple("PathValue", ["value", "dic"])


class GBM:
    def __init__(self, asset: Asset, rv: Union[list, numpy.ndarray], s0: float, stdt: TimeStr,
                 temporal_expression: str, delta: RealNum):
        self.asset = asset.clone()
        self.timeline = [TimeStamp(stdt)]
        m = [s0]
        for i, r in enumerate(rv):
            self.timeline.append(
                self.timeline[-1] + [temporal_expression, delta]
            )
            m.append(
                m[-1] * (1 + r)
            )
        self.yield_rate_list = copy.deepcopy(rv)
        self.times_series = TimeSeries(timestamp=self.timeline, price=m)

    def __repr__(self):
        return str(self.times_series)

    def get_price(self, timestamp: TimeStr) -> PathValue:
        """获取序列价格"""
        array = self.times_series.get_array()
        index_array, *_ = numpy.where(array[:, 0] <= TimeStamp(timestamp).timestamp())
        if index_array.size > 0:
            index = index_array[-1]
            value = array[:, 1][index]
            return PathValue(value, {self.asset.name: {"price": value}})
        else:
            raise Exception(f"{self.timeline[0]} > {timestamp}, index: {index_array}")

    def purchased_to(self, trader: Trader, timestamp: TimeStr, *args, **kwargs):
        data = {
            "price": self.get_price(timestamp).value,
            "trader": trader,
            "timestamp": timestamp,
            "position": PositionType.long
        }
        self.asset.purchased_to(*args, **data, **kwargs)

    def sold_to(self, trader: Trader, timestamp: TimeStr, *args, **kwargs):
        data = {
            "price": self.get_price(timestamp).value,
            "trader": trader,
            "timestamp": timestamp,
            "position": PositionType.long
        }
        self.asset.sold_to(*args, **data, **kwargs)


if __name__ == "__main__":
    rv = NormalDistribution(0, 0.015).rvf(365)
    s = Spot("10001", 100)
    print(GBM(s, rv, 100, "2024-1-1", "day", 1).timeline)


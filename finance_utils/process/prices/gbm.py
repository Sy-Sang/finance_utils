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

from finance_utils.uniontypes import *
from finance_utils.asset.base import *
from finance_utils.asset.spot.base import Spot
from finance_utils.trader.base import Trader

from finance_utils.process.prices.base import PriceProcess, MultiPathing, PricePathValue

# 外部模块
import numpy


# 代码块


class RVDecoupledGBM(PriceProcess):
    """与收益率随机变量分离的GBM"""

    def __init__(self, name: str, rv: Union[list, numpy.ndarray], s0: float, stdt: TimeStr,
                 temporal_expression: str, delta: Rational):
        super().__init__(s0)
        self.name = name
        self.timeline = [TimeStamp(stdt)]
        self.price = [s0]
        for i, r in enumerate(rv):
            self.timeline.append(
                self.timeline[-1] + [temporal_expression, delta]
            )
            self.price.append(
                self.price[-1] * (1 + r)
            )
        self.yield_rate_list = copy.deepcopy(rv)
        self.times_series = TimeSeries(timestamp=self.timeline, price=self.price)
        self.constructor = {
            "name": name,
            "rv": self.yield_rate_list,
            "s0": s0,
            "stdt": stdt,
            "temporal_expression": temporal_expression,
            "delta": delta
        }

    def __repr__(self):
        return str(self.times_series)

    def get_price(self, timestamp: TimeStr):
        ts = TimeStamp(timestamp)
        if ts >= self.timeline[0]:
            raw_index = numpy.searchsorted(self.times_series.data["timestamp"], ts.timestamp())
            index = raw_index - 1 if raw_index > 0 else 0
            value = self.price[index]
            return PricePathValue(
                self.timeline[index],
                value,
                {self.name: {"price": value}}
            )
        else:
            return PricePathValue(
                numpy.nan,
                numpy.nan,
                {self.name: {"price": numpy.nan}}
            )

    def multi_pathing(self, base_trader: Trader, base_asset: Asset, rv_list: list):
        path_list = []
        trader_list = []
        for i, rv in enumerate(rv_list):
            constructor = self.constructor
            constructor["rv"] = rv
            path_list.append(
                type(self)(**constructor)
            )
            trader_list.append(
                base_trader.clone(f"{base_trader.name}_copy_{i}")
            )
        return MultiPathing(copy.deepcopy(self.timeline), path_list, trader_list)


class GBM(RVDecoupledGBM):
    """标准GBM价格过程"""

    def __init__(self, name: str, s0: float, mu: float, sigma: float, len: int, stdt: TimeStr, temporal_expression: str,
                 delta: Rational):
        rv = NormalDistribution(mu, sigma).rvf(len - 1)
        super().__init__(name, rv, s0, stdt, temporal_expression, delta)
        self.constructor = {
            "name": name,
            "s0": s0,
            "mu": mu,
            "sigma": sigma,
            "len": len,
            "stdt": stdt,
            "temporal_expression": temporal_expression,
            "delta": delta
        }

    def multi_pathing(self, base_trader: Trader, base_asset: Asset, num: int):
        path_list = []
        trader_list = []
        for i in range(num):
            constructor = self.constructor
            path_list.append(
                type(self)(**constructor)
            )
            trader_list.append(
                base_trader.clone(f"{base_trader.name}_copy_{i}")
            )
        return MultiPathing(copy.deepcopy(self.timeline), path_list, trader_list)


if __name__ == "__main__":
    rv = NormalDistribution(0, 0.015).rvf(365)
    s = Spot("10001", 100)
    print(RVDecoupledGBM(s.name, rv, 100, "2024-1-1", "day", 1).timeline)

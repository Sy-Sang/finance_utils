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
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution
from data_utils.serial_utils.time_series import TimeSeries
from easy_datetime.timestamp import TimeStamp

from finance_utils.asset.base import *
from finance_utils.asset.spot.base import Spot

# 外部模块
import numpy


# 代码块

def gbm(s0: float, mu: float, sigma: float, stdt: TimeStr, eddt: TimeStr, temporal_expression: str, delta: RealNum,
        include_last: bool = False) -> TimeSeries:
    timeline = TimeStamp.timestamp_range(stdt, eddt, temporal_expression, delta, include_last)
    r = NormalDistribution(mu, sigma).rvf(len(timeline))
    m = []
    for i in range(len(timeline)):
        m.append(s0)
        s0 *= 1 + r[i]

    ts = TimeSeries(timestamp=timeline, price=m)
    return ts


if __name__ == "__main__":
    print(gbm(100, 0, 0.015, "2024-1-1", "2025-1-1", "day", 1))

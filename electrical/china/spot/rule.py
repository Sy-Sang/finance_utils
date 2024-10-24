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
from easy_datetime.timestamp import TimeStamp
from data_utils.serial_utils.time_series import TimeSeries
from data_utils.stochastic_utils.distributions.baseclass import ABCDistribution
from data_utils.stochastic_utils.distributions.basic_distributions import NormalDistribution

# 外部模块
import numpy


# 代码块

def excess_return(
        submit_quantity,
        actual_quantity,
        price_baseline,
        price_day_ahead,
        price_real_time,
        max_deviation_ratio=0.4,
        government_price_factor=0.5
) -> float:
    """新能源超额获利回收"""
    allowed_energy = actual_quantity * (1 + max_deviation_ratio)
    energy_deviation = max(0, submit_quantity - allowed_energy)
    price_diff = government_price_factor * price_baseline + (
            1 - government_price_factor) * price_day_ahead - price_real_time
    recovery_amount = energy_deviation * price_diff

    return max(0, recovery_amount)


if __name__ == "__main__":
    tl = TimeStamp.timestamp_range("2024-1-1", "2024-1-2", "min", 15)
    print(len(tl))

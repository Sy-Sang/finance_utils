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
from finance_utils.electrical.china.spot.rule.recycle import Recycle, AnarchismRecycle, SampleRecycle

# 外部模块
import numpy


# 代码块

def province_new_energy(dayahead: float, realtime: float, realtime_quantity: float, submitted_quantity: float):
    """交易结算函数"""
    return float(
        dayahead * submitted_quantity + (realtime_quantity - submitted_quantity) * realtime
    )


def province_new_energy_with_recycle(
        dayahead: Union[list, numpy.ndarray], realtime: Union[list, numpy.ndarray],
        realtime_quantity: Union[list, numpy.ndarray], submitted_quantity: Union[list, numpy.ndarray],
        recycle: Type[Recycle] = None, *args, **kwargs
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

    return total_yield - punishment


if __name__ == "__main__":
    pass

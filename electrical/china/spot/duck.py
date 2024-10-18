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

# class Duck:
#     def __init__(self, *args):
#         self.base_list = numpy.array(base_list).astype(float)
#
#     def min_max(self, ):


if __name__ == "__main__":
    tl = TimeStamp.timestamp_range("2024-1-1", "2024-1-2", "min", 15)
    print(len(tl))

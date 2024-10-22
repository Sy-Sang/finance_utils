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
from data_utils.serial_utils.series_trans_utils import MinMax
from data_utils.stochastic_utils.random_process import correlatedRandom

# 外部模块
import numpy
from matplotlib import pyplot


# 代码块


class Spot(TimeSeries):
    """现货差价合约"""

    def __init__(self, timeline, dayahead, realtime):
        super().__init__(x=timeline, y=dayahead, y_hat=realtime)


if __name__ == "__main__":
    tl = TimeStamp.timestamp_range("2024-1-1", "2024-1-2", "min", 15)
    print(len(tl))

    # exp = SpotCFD()

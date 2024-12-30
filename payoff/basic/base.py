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

# 外部模块
import numpy


# 代码块

class Spot:
    def __init__(self, premium: float, strike_point: float, ):
        self.strike_point = strike_point

    def __call__(self, x, *args, **kwargs):
        return x - self.strike_point


if __name__ == "__main__":
    pass

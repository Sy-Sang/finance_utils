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
from typing import Union, Self, TypeAlias
from collections import namedtuple

# 项目模块
from easy_datetime.timestamp import TimeStamp

# 外部模块
import numpy

# 代码块
Rational: TypeAlias = Union[int, float]
TimeStr: TypeAlias = Union[TimeStamp, str, None]

if __name__ == "__main__":
    pass

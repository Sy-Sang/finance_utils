#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""financial_object基类"""

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
from abc import abstractmethod, ABC

# 项目模块

# 外部模块
import numpy


# 代码块

class FinancialObject(ABC):
    def __init__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    pass

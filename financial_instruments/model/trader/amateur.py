#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""amateur trader class"""

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
from typing import *
from collections import namedtuple

# 项目模块
from financial_instruments.model.fm import FinancialModule, ReactionEvent
from financial_instruments.model.entity.currency.cash import BasicCash

# 外部模块
import numpy


# 代码块

class TheMostAmateurTrader(FinancialModule):
    def __init__(self, name, init_amount: float, amount_symbol="CNY"):
        super().__init__(name)
        self.amount = BasicCash(amount_symbol, init_amount)
        self.position = FinancialModule("position")


if __name__ == "__main__":
    pass

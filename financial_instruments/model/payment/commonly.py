#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""commonly payment"""

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
from financial_instruments.mop.fm import *
from financial_instruments.model.currency.cash import BasicCash

# 外部模块
import numpy


# 代码块

class CashPay(FinancialModule):
    """立即产生负现金流"""

    def __init__(self, sym, amount):
        super().__init__(None)
        self.sym = sym
        self.amount = amount

    def reaction(self, *args, **kwargs) -> Union["ReactionEvent", None]:
        return ReactionEvent(
            None,
            "payment",
            False,
            None,
            -BasicCash(self.sym, self.amount)
        )


if __name__ == "__main__":
    pass

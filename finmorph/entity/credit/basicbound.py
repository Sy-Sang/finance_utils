#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""债券"""

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
from easy_datetime.timestamp import TimeStamp
from finmorph.mop.finmodule import *
from finmorph.mop.numeraire import *

# 外部模块
import numpy


# 代码块
class BasicBound(FinancialModule):
    def __init__(
            self,
            face_amount: Numeraire,
            price: Numeraire,
            quantity: int,
            maturity_date: Any,
            *args, **kwargs
    ):
        super().__init__(FinancialQuantity(quantity), *args, **kwargs)
        self.face_amount = face_amount
        self.price = price
        self.maturity_date = TimeStamp(maturity_date)

    def __repr__(self):
        return (
            f"{type(self).__name__}.{self.id}:{{"
            f"face={self.face_amount}; "
            f"price={self.price}; "
            f"quantity={self.quantity}; "
            f"maturity={self.maturity_date})}}"
        )

    def _value(self, *args, **kwargs):
        if TimeStamp(kwargs['date']) < self.maturity_date:
            pass


if __name__ == "__main__":
    bound1 = BasicBound(Numeraire(100), Numeraire(90), 1, '2025-12-31')
    bound2 = BasicBound(Numeraire(100), Numeraire(95), 1, '2025-8-31')

    print(bound1.value())

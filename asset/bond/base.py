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
from enum import Enum

# 项目模块
from finance_utils.asset.base import Asset

# 外部模块
import numpy


# 代码块

class Role(Enum):
    """角色"""
    borrower = -1
    lender = 1


class BaseBond(Asset):
    def __init__(self, amount: float, rate: float, term: float, role: Role):
        self.amount = amount
        self.rate = rate
        self.term = term
        self.role = role

    def settlement(self, *args, **kwargs) -> float:
        return self.amount * (-1 * self.role.value)

    def payoff(self, *args, **kwargs) -> float:
        return self.amount * (1 + self.rate) ** self.term * self.role.value


# class BondGroup:
#     def __init__(self):
#         self.bonds = []
#
#     def append(self, bond: BaseBond):
#         self.bonds.append(bond.clone())


if __name__ == "__main__":
    b = BaseBond(10000, 0.03, 2, Role.lender)
    print(b.payoff())

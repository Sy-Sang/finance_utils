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
from finance_utils.asset.base import Asset, LoanRole

# 外部模块
import numpy


# 代码块


class BaseBond(Asset):
    def __init__(self, amount: float, rate: float, role: LoanRole = LoanRole.borrower, shares: float = 1):
        super().__init__(shares)
        self.amount = amount
        self.rate = rate
        self.role = role

    @Asset.with_shares
    def initial_cost(self, *args, **kwargs):
        return self.amount * self.role.value

    @Asset.with_shares
    def payoff(self, x, *args, **kwargs) -> float:
        if x > 0:
            return self.amount * (1 + self.rate) * self.role.value * -1
        else:
            return 0


# class BondGroup:
#     def __init__(self):
#         self.bonds = []
#
#     def append(self, bond: BaseBond):
#         self.bonds.append(bond.clone())


if __name__ == "__main__":
    b = BaseBond(10000, 0.03, 2, LoanRole.lender)
    print(b.payoff())

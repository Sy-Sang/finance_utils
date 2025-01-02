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
from finance_utils.asset.base import Asset, PositionType

# 外部模块
import numpy


# 代码块


class Spot(Asset):
    """现货"""

    def __init__(
            self,
            name,
            price: float,
            # position: PositionType = PositionType.long,
            shares: float = 1,
            lot_size: Union[int, float] = None
    ):
        super().__init__(name, shares)
        self.price = price
        # self.position = position
        self.lot_size = lot_size

    def __repr__(self):
        return str({
            "name": self.name,
            "type": "Asset.Spot",
            "price": self.price,
            "shares": self.shares
        })

    @Asset.with_shares
    def purchase_cost(self, position: PositionType, *args, **kwargs):
        return self.price * position.value * -1

    @Asset.with_shares
    def payoff(self, x: float, position: PositionType, *args, **kwargs):
        """损益"""
        return (x - self.price) * position.value

    def max_purchaseable(self, capital: float):
        """最大购买量"""
        if self.lot_size is None:
            shares = capital / self.price
        else:
            shares = (capital // (self.price * self.lot_size)) * self.lot_size
        my_copy = self.clone()
        my_copy.shares = shares
        return my_copy


if __name__ == "__main__":
    pass

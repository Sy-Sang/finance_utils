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
from finance_utils.asset.base import Asset, PositionType, LoanRole, OptionType
from easy_datetime.timestamp import TimeStamp

# 外部模块
import numpy


# 代码块

class PositionUnit:
    def __init__(self, timestamp: Union[TimeStamp, str], asset: Asset, position_type: PositionType):
        self.timestamp = TimeStamp(timestamp)
        self.asset = asset
        self.type = position_type

    def __repr__(self):
        return str({
            "timestamp": self.timestamp,
            "asset": self.asset
        })


class Trader:
    def __init__(self, capital: float):
        self.capital = capital
        self.position = {}

    def __repr__(self):
        return str({
            "capital": self.capital,
            "position": self.position
        })

    def purchase(self, amount: float, asset: Asset, timestamp: Union[TimeStamp, str], *args, **kwargs):
        """买入"""
        amount = min(self.capital, amount)
        purchaseable_asset = asset.max_purchaseable(amount)
        if purchaseable_asset.shares > 0:
            self.capital += purchaseable_asset.purchase_cost(PositionType.long)
            trade_detail = PositionUnit(timestamp, purchaseable_asset, PositionType.long)
            if asset.name in self.position:
                self.position[asset.name].append(trade_detail)
            else:
                self.position[asset.name] = [trade_detail]

    # def sell(self, asset_name: Union[str, int], amount, asset_price, timestamp: Union[TimeStamp, str], *args, **kwargs):
    #     """卖出"""
    #     if asset_name in self.position:
    #         p: list[PositionUnit] = self.position[asset_name]
    #         bought_amount = sum([i.asset.shares for i in p if i.type == PositionType.long])
    #         sold_amount = sum([i.asset.shares for i in p if i.type == PositionType.short])
    #         sellable_amount = min(amount, bought_amount - sold_amount)
    #         sellable_asset =
    #         if sellable_amount > 0:
    #             trade_detail = PositionUnit(timestamp, purchaseable_asset, PositionType.short)
    #         else:
    #             pass
    #     else:
    #         pass


if __name__ == "__main__":
    from finance_utils.asset.spot.base import Spot, PositionType

    s = Spot(10001, 100, PositionType.long, lot_size=100)

    t = Trader(100000)
    t.purchase(10000, s, "2024-1-1")
    t.purchase(100, s, "2024-1-1")
    t.purchase(30000, s, "2024-3-1")
    print(t)

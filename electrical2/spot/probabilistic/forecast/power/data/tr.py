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
from easy_datetime.timestamp import TimeStamp
from private.db.tianrun.spot.fdw.new_energy_actual_power import market_hourly_actual_power
from private.db.tianrun.spot.fdw.xinzhi_new_energy_forecast_power import xinzhi_forecast_power

# 外部模块
import numpy

# 代码块

market_dic = {
    "shanxi": {
        "code": 14,
        "market": "PHBSX"
    }
}


class TRInterFace:

    @classmethod
    def hourly_actual_range(cls, market: str, st: str, ed: Union[int, str]):
        if isinstance(ed, int):
            ed = TimeStamp(st) + ["day", ed]
        else:
            pass

        return market_hourly_actual_power(
            market_dic[market]["market"], st, ed
        )

    @classmethod
    def hourly_actual_power(cls, market: str, target_day: str, known_days: Union[int, list]) -> numpy.ndarray:
        """实际出力"""
        p = numpy.array([])
        if isinstance(known_days, int):
            tr = TimeStamp.timestamp_range(
                TimeStamp(target_day) - ["day", known_days],
                TimeStamp(target_day) - ["day", 1],
                "day", 1, True
            ) + [TimeStamp(target_day)]
        else:
            tr = [TimeStamp(i) for i in known_days] + [TimeStamp(target_day)]
        for i, t in enumerate(tr):
            act = market_hourly_actual_power(market_dic[market]["market"], t, t)
            p = numpy.concatenate((p, act["power"]))
        return p

    @classmethod
    def control_group_forecast(cls, market: str, target_day: str, forecast_day: str):
        """对照组预测出力"""
        target_ts = TimeStamp(target_day)
        data = xinzhi_forecast_power(market_dic[market]["code"], forecast_day)
        return data.where(">=", target_ts.get_date()).where("<=", target_ts.get_date_with_last_sec()).aggregate(
            ["hour", 1],
            align=True,
            align_domain=[
                target_ts.get_date_string(),
                (target_ts + ["day", 1]).get_date_string()
            ]
        )


if __name__ == "__main__":
    print(TRInterFace.hourly_actual_range("shanxi", "2024-10-1", 0)["timestamp"])

    # print(control_group_forecast("shanxi", "2024-10-10", "2024-10-1"))

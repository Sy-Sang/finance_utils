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
from private.db.tianrun.spot.fdw.xinzhi_new_energy_forecast_power import xinzhi_daily_forecast_power

# 外部模块
import numpy

# 代码块

market_dic = {
    "shanxi": {
        "code": 14,
        "market": "PHBSX"
    }
}


def hourly_actual_power(market: str, target_day: str, *args) -> numpy.ndarray:
    p = numpy.array([])
    tr = [TimeStamp(i) for i in args] + [TimeStamp(target_day)]
    for i, t in enumerate(tr):
        act = market_hourly_actual_power(market_dic[market]["market"], t, t)
        p = numpy.concatenate((p, act["power"]))
    return p


def non_stop_hourly_forecast(market: str, target_day: str, counter: int = 10):
    td = TimeStamp(target_day)
    stdt = TimeStamp(td) - ["day", 10]
    eddt = TimeStamp(td) - ["day", 1]
    tr = TimeStamp.timestamp_range(stdt, eddt, "day", 1, True)
    f = numpy.array([])
    for i, t in enumerate(tr):
        f = numpy.concatenate((
            f,
            xinzhi_daily_forecast_power(market_dic[market]["code"], t.get_date_string(), td.get_date_string())
        ))
    return f


def day2day_hourly_forecast(market: str, target_day: str, counter: int = 10, delta: int = 0):
    td = TimeStamp(target_day)
    stdt = TimeStamp(td) - ["day", counter]
    eddt = TimeStamp(td) - ["day", 1]
    tr = TimeStamp.timestamp_range(stdt, eddt, "day", 1, True)
    f = numpy.array([])
    for i, t in enumerate(tr):
        ft = t - ["day", delta]
        tt = ft + ["day", 1 + delta]

        f = numpy.concatenate((
            f,
            xinzhi_daily_forecast_power(market_dic[market]["code"], ft.get_date_string(),
                                        tt.get_date_string())
        ))
    return f


if __name__ == "__main__":
    print(hourly_actual_power("shanxi", "2024-10-10",
                              *TimeStamp.timestamp_range("2024-10-1", "2024-10-9", "day", 1, True)).tolist())

    print(day2day_hourly_forecast("shanxi", "2024-10-10", delta=10).tolist())

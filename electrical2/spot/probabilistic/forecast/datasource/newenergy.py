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
import os

# 项目模块
from private.db.tianrun.spot.fdw.new_energy_actual_power import market_actual_power, market_name_dict
from private.db.tencnet.openmeteo.china_opemmeteo_forecast2 import ProvinceForecastData, forecast_slice
from easy_datetime.timestamp import TimeStamp
from data_utils.serial_utils.time_series import TimeSeries
from data_utils.serial_utils.series_trans_utils import MinMax, ZScore, RobustScaler

# 外部模块
import numpy


# 代码块

class PFDRoot:
    """预测文件集"""

    def __init__(self, path: str):
        self.path = path
        self.pfd_map = {}
        self.map_series = []

    def clr(self):
        self.pfd_map = {}
        self.map_series = []

    def get_forecast(self, province_name: str, forecast_date: Union[str, TimeStamp],
                     v_name: str = "", show_tqdm: bool = True):
        """获取预测对象"""
        forecast_timestamp = TimeStamp(forecast_date)
        file_name = f"{province_name}_{forecast_timestamp.get_date_string()}_{v_name}.pfd"
        full_name = f"{self.path}\\{file_name}"
        if os.path.exists(full_name):
            with open(f"{full_name}", "rb") as f:
                pfd = pickle.loads(f.read())
        else:
            pfd = ProvinceForecastData(province_name, forecast_timestamp.get_date_string())
            pfd.add_new_energy_dims(show_progress=show_tqdm)
            pfd.dumps_and_save(full_name)

        index = len(self.map_series)
        self.pfd_map[index] = pfd
        self.map_series.append([forecast_timestamp, index])

    def historical_forecast(self, start_offset: int = 1, end_offset: int = 1, item=None):
        """整合历史预测信息"""
        if item is None:
            item = ["cma_all", 0]
        cat_list = []
        dt, id = zip(*self.map_series)
        ts = TimeSeries(date=dt, id=id)
        for i in ts["id"]:
            pfd: ProvinceForecastData = self.pfd_map[int(i)]
            long_data = pfd[item]
            slice_data = forecast_slice(long_data, pfd, start_offset, end_offset)
            cat_list.append(slice_data)
        return numpy.concatenate(cat_list)

    # def actual_power(self, start_offset: int = 1, end_offset: int = 1):
    #     dt, id = zip(*self.map_series)
    #     ts = TimeSeries(date=dt, id=id)
    #     for i in ts["id"]:
    #         pfd: ProvinceForecastData = self.pfd_map[int(i)]
    #         province = pfd.


if __name__ == "__main__":
    root = PFDRoot(r"E:\code\github\private\private\db\tencnet\openmeteo\data")
    trange = TimeStamp.timestamp_range("2024-10-1", "2024-10-10", "day", 1, True)
    for t in trange:
        root.get_forecast("shanxi", t, "v7")

    d = root.historical_forecast(1, 2, item=["cma_all"])
    print(d[:, 0].tolist())

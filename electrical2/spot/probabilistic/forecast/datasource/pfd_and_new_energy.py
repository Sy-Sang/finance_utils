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
from typing import Union, Self, Type
from collections import namedtuple
import os

# 项目模块
from private.db.tianrun.spot.fdw.new_energy_actual_power import market_hourly_actual_power, market_name_dict
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import ProvinceForecastData
from easy_datetime.timestamp import TimeStamp
from data_utils.serial_utils.time_series import TimeSeries
# from data_utils.serial_utils.series_trans_utils import DataTransformator, MinMax, ZScore, RobustScaler
from finance_utils.electrical2.spot.probabilistic.forecast.torch.normalization.basic import Norm, MinMax

# 外部模块
import numpy
import torch


# 代码块

def torch_data_set(pfd: ProvinceForecastData, x_norm_method: Type[Norm], y_norm_method: Type[Norm],
                   train_start_date: str, forecast_date: str, slicer: list,
                   delete_column: Union[tuple, list] = None, province_name: str = None, *args, **kwargs):
    province_name = province_name if province_name is not None else market_name_dict[pfd.province_name]
    stdt_ts = TimeStamp(train_start_date)
    eddt_ts = TimeStamp(forecast_date)
    delta_days = int((eddt_ts - stdt_ts) / (24 * 60 * 60))
    trange = TimeStamp.timestamp_range(stdt_ts, eddt_ts, "day", 1, True)
    meteo_data = pfd[slicer[0], slicer[1], slicer[2], slicer[3], trange]
    actual_train = market_hourly_actual_power(
        province_name, stdt_ts.get_date_string(), (eddt_ts - ["day", 1]).get_date_string()
    )["power"]
    actual_test = market_hourly_actual_power(
        province_name, stdt_ts.get_date_string(), eddt_ts.get_date_string()
    )["power"]

    meteo_data_train_tensor = torch.Tensor(
        numpy.apply_along_axis(x_norm_method.f, axis=0, arr=meteo_data, *args, **kwargs)
    )[:delta_days * 24]

    meteo_data_test_tensor = torch.Tensor(
        numpy.apply_along_axis(x_norm_method.f, axis=0, arr=meteo_data, *args, **kwargs)
    )

    actual_train_tensor_norm_p = numpy.apply_along_axis(y_norm_method.params, axis=0, arr=actual_train, *args, **kwargs)

    actual_train_tensor = torch.Tensor(
        numpy.apply_along_axis(y_norm_method.f, axis=0, arr=actual_train, *args, **kwargs)
    )

    actual_test_tensor = torch.Tensor(actual_test)

    if delete_column is None:
        pass
    else:
        meteo_data_train_tensor = numpy.delete(meteo_data_train_tensor, delete_column, axis=1)
        meteo_data_test_tensor = numpy.delete(meteo_data_test_tensor, delete_column, axis=1)

    return meteo_data_train_tensor, meteo_data_test_tensor, actual_train_tensor, actual_test_tensor, actual_train_tensor_norm_p


if __name__ == "__main__":
    root = r"E:\code\github\private\private\db\tencnet\openmeteo\data\test_cma.pfd"
    with open(root, "rb") as f:
        pfd = pickle.loads(f.read())

    m, mt, t, tt, tp = torch_data_set(pfd, MinMax, MinMax, "2024-10-1", "2024-10-2", ["cma", "morning", 1, 1])
    print(m.shape)
    print(mt.shape)
    print(t.shape)
    print(tt.shape)
    print(tp)
    # print(t.numpy().tolist())
    print(MinMax.invert(t, tp).tolist())
    print(tt.numpy().tolist())

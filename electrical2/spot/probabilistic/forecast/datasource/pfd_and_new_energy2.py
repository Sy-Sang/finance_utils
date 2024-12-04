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
import tqdm

# 项目模块
from private.db.tianrun.spot.fdw.new_energy_actual_power import market_hourly_actual_power, market_name_dict
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import ProvinceForecastData, new_or_stack, \
    one_dim_new_or_cat
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import cma_new_energy_args, ecmwf_new_energy_args
from easy_datetime.timestamp import TimeStamp
from data_utils.serial_utils.time_series import TimeSeries
# from data_utils.serial_utils.series_trans_utils import DataTransformator, MinMax, ZScore, RobustScaler
from finance_utils.electrical2.spot.probabilistic.forecast.torch.normalization.basic import Norm, MinMax, ZScore

# 外部模块
import numpy
import torch


# 代码块

def get_reformed_dims(dims: list[str]) -> tuple[list, dict]:
    reformed_dims = []
    for d in dims:
        if "direction" in d:
            reformed_dims.append(f"{d}_sin")
            reformed_dims.append(f"{d}_cos")
        else:
            reformed_dims.append(f"{d}")
    p_dic = {k: numpy.array([]) for k in reformed_dims}
    return reformed_dims, p_dic


def grid_norm(m: numpy.ndarray, norm: Type[Norm], dims: list[str], time_dims: int = 7, show_tqdm: bool = True, *args,
              **kwargs):
    rm = numpy.array([])
    reformed_dims, dim_dic = get_reformed_dims(dims)
    p_dic = copy.deepcopy(dim_dic)
    cmd_dic = {}
    it = range(m.shape[1]) if show_tqdm is False else tqdm.trange(m.shape[1])

    grid_num = (m.shape[1] - time_dims) / len(reformed_dims)
    # print(grid_num)
    k_index = 0
    k_counter = 0

    for i in it:
        c = m[:, i]
        if i < time_dims:
            cmd_dic[i] = 0
        else:
            k = reformed_dims[k_index]
            cmd_dic[i] = k
            dim_dic[k] = one_dim_new_or_cat(dim_dic[k], c)

            k_counter += 1
            if k_counter == int(grid_num):
                k_counter = 0
                k_index += 1

    for k, c in dim_dic.items():
        p_dic[k] = norm.params(c, *args, **kwargs)

    for i, cmd in cmd_dic.items():
        c = m[:, i]
        if cmd == 0:
            rm = new_or_stack(rm, norm.f(c, *args, **kwargs))
        elif cmd == 1:
            rm = new_or_stack(rm, c)
        else:
            rm = new_or_stack(rm, norm.f_with_params(c, p_dic[cmd], *args, **kwargs))

    return rm


def torch_data_set(pfd: ProvinceForecastData, x_norm_method: Type[Norm], y_norm_method: Type[Norm],
                   train_start_date: str, forecast_date: str, slicer: list,
                   dims: list[str], time_dims: int = 7,
                   delete_column: Union[tuple, list] = None, province_name: str = None, show_tqdm: bool = True, *args,
                   **kwargs):
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

    meteo_data_test_tensor = torch.Tensor(
        grid_norm(meteo_data, x_norm_method, dims, time_dims, show_tqdm, *args, **kwargs)
    )

    meteo_data_train_tensor = meteo_data_test_tensor[:delta_days * 24]

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
    root = r"E:\code\github\private\private\db\tencnet\openmeteo\data\test_ecmwf.pfd"
    with open(root, "rb") as f:
        pfd = pickle.loads(f.read())

    m, mt, t, tt, tp = torch_data_set(
        pfd,
        MinMax,
        MinMax,
        "2024-10-1",
        "2024-10-2",
        ["ecmwf", "morning", 1, 1],
        ecmwf_new_energy_args,
    )

    print(m[0].tolist())
    # print(m.shape)
    # print(mt.shape)
    # print(t.shape)
    # print(tt.shape)
    # print(tp)
    # # print(t.numpy().tolist())
    # print(MinMax.invert(t, tp).tolist())
    # print(tt.numpy().tolist())

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
import tqdm

# 项目模块
from easy_datetime.timestamp import TimeStamp
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import ProvinceForecastData, new_or_stack, \
    one_dim_new_or_cat, cma_new_energy_args, ecmwf_new_energy_args

from finance_utils.electrical2.spot.probabilistic.forecast.torch.normalization.basic import Norm, MinMax, ZScore

# 外部模块
import numpy


# 代码块

def get_meteo_keys(dims: list[str]) -> tuple[list, dict]:
    meteo_keys = []
    for d in dims:
        if "direction" in d:
            meteo_keys.append(f"{d}_sin")
            meteo_keys.append(f"{d}_cos")
        else:
            meteo_keys.append(f"{d}")
    p_dic = {k: numpy.array([]) for k in meteo_keys}
    return meteo_keys, p_dic


def approaching_forecast(pfd: ProvinceForecastData, target_day: str, slicer: list, farthest: int = 10):
    """
    逐步逼近的天气预报
    """
    td = TimeStamp(target_day)
    stdt = TimeStamp(td) - ["day", farthest]
    eddt = TimeStamp(td) - ["day", 1]
    tr = TimeStamp.timestamp_range(stdt, eddt, "day", 1, True)
    forecast = []
    for i, t in enumerate(tr):
        forecast.append(
            pfd[slicer[0], slicer[1], 1, farthest - i, t]
        )
    return forecast


def directed_forecast(pfd: ProvinceForecastData, target_day: str, slicer: list, farthest: int = 10):
    """直接指向目标日的天气预报"""
    td = TimeStamp(target_day)
    stdt = TimeStamp(td) - ["day", farthest]
    eddt = TimeStamp(td) - ["day", 1]
    tr = TimeStamp.timestamp_range(stdt, eddt, "day", 1, True)
    forecast = []
    for i, t in enumerate(tr):
        forecast.append(
            pfd[slicer[0], slicer[1], farthest - i, farthest - i, t]
        )
    return forecast


def grid_separate(matrix: numpy.ndarray, dims: list[str], non_grid_dims: int = 7):
    """分离出每个格点的数据"""
    separated = []

    meteo_keys, dim_dic = get_meteo_keys(dims)

    meteo_len = len(meteo_keys)
    grid_len = int((matrix.shape[1] - non_grid_dims) / meteo_len)

    for l in range(grid_len):
        rm = numpy.array(matrix[:, :non_grid_dims])
        for i in range(non_grid_dims + l, matrix.shape[1], grid_len):
            c = matrix[:, i]
            rm = new_or_stack(rm, c)
        separated.append(rm)
    return separated


def norm_by_meteo_key(matrix: numpy.ndarray, norm: Type[Norm], dims: list[str], non_grid_dims: int = 7,
                      show_tqdm: bool = True, *args, **kwargs):
    """根据气象变量归一化"""
    rm = numpy.array([])
    meteo_keys, p_dic = get_meteo_keys(dims)

    it = range(matrix.shape[1]) if show_tqdm is False else tqdm.trange(matrix.shape[1])

    meteo_len = len(meteo_keys)
    grid_len = int((matrix.shape[1] - non_grid_dims) / meteo_len)

    grid_index = 0
    for i, k in enumerate(meteo_keys):
        column = numpy.array([])
        st = non_grid_dims + grid_index
        ed = non_grid_dims + grid_index + grid_len
        for g in range(st, ed):
            column = one_dim_new_or_cat(column, matrix[:, g])
        p_dic[k] = norm.params(column, *args, **kwargs)
        grid_index += grid_len

    print(p_dic)
    k_index = 0
    c_counter = 0
    for i in it:
        if i < non_grid_dims:
            rm = new_or_stack(rm, norm.f(matrix[:, i], *args, **kwargs))
        else:
            print(meteo_keys[k_index])
            rm = new_or_stack(rm, norm.f_with_params(matrix[:, i], p_dic[meteo_keys[k_index]]))
            c_counter += 1
            if c_counter == grid_len:
                c_counter = 0
                k_index += 1
    return rm


if __name__ == "__main__":
    from matplotlib import pyplot

    with open(r"E:\code\github\private\private\db\tencnet\openmeteo\data\shanxi_1.pfd", "rb") as f:
        pfd: ProvinceForecastData = pickle.loads(f.read())

    # print(pfd.cma_forecast_matrix[:,7].tolist())

    f = approaching_forecast(pfd, "2024-10-10", ["test", "morning"], 9)
    #
    #
    print(f)

    mm = grid_separate(f[0], ecmwf_new_energy_args)

    print(mm[0].shape)

    nmm = norm_by_meteo_key(mm[0], MinMax, ecmwf_new_energy_args)

    for i in range(nmm.shape[1]):
        # for i in range(7 + 3 * 25, 3 * 25 + 25):
        pyplot.plot(nmm[:, i])
    pyplot.show()
    # print([TimeStamp(i) for i in f[:, 0]])

    # print(pfd["cma", "evening", 1, 1, "2024-10-1"][:,7].tolist())

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


class OpenMeteoForecast:
    def __init__(self, pfd: ProvinceForecastData):
        self.pfd = pfd

    def approaching_forecast(self, target_day: str, slicer: list):
        td = TimeStamp(target_day)
        stdt = TimeStamp(td) - ["day", 10]
        eddt = TimeStamp(td) - ["day", 1]
        tr = TimeStamp.timestamp_range(stdt, eddt, "day", 1, True)
        forecast = []
        for i, t in enumerate(tr):
            forecast.append(
                self.pfd[slicer[0], slicer[1], 1, 10 - i, t]
            )
        return forecast

    def non_stop_forecast(self, target_day: str, slicer: list):
        td = TimeStamp(target_day)
        stdt = TimeStamp(td) - ["day", 10]
        eddt = TimeStamp(td) - ["day", 1]
        tr = TimeStamp.timestamp_range(stdt, eddt, "day", 1, True)
        forecast = []
        for i, t in enumerate(tr):
            forecast.append(
                pfd[slicer[0], slicer[1], 10 - i, 10 - i, t]
            )
        return forecast


class MeteoForecastMatrix:
    def __init__(self, m: numpy.array):
        self.matrix = m

    def grid_separate(self, dims: list[str], non_grid_dims: int = 7):
        separated = []

        meteo_keys, dim_dic = get_meteo_keys(dims)

        dim_len = len(meteo_keys)

        grid_index = 0
        for l in range(dim_len):
            rm = numpy.array(self.matrix[:, :non_grid_dims])
            for i in range(non_grid_dims + grid_index, self.matrix.shape[1], dim_len):
                c = self.matrix[:, i]
                rm = new_or_stack(rm, c)
            grid_index += 1
            separated.append(rm)

        return separated

    def norm_by_meteo_key(self, norm: Type[Norm], dims: list[str], non_grid_dims: int = 7,
                          show_tqdm: bool = True, *args, **kwargs):
        """根据气象变量归一化"""
        rm = numpy.array([])
        meteo_keys, p_dic = get_meteo_keys(dims)
        # dim_param_dic = copy.deepcopy(dim_dic)
        cmd_dic = {}
        it = range(self.matrix.shape[1]) if show_tqdm is False else tqdm.trange(self.matrix.shape[1])

        meteo_len = len(meteo_keys)
        grid_len = int((self.matrix.shape[1] - non_grid_dims) / meteo_len)

        grid_index = 0
        for i,k in enumerate(meteo_keys):
            column = numpy.array([])
            st = non_grid_dims + grid_index
            ed = non_grid_dims + grid_index + grid_len
            for g in range(st,ed):
                column = one_dim_new_or_cat(column, self.matrix[:,g])
            # if i == 4:
            #     print(k)
            #     print(column.tolist())
            #     break
            p_dic[k] = norm.params(column, *args, **kwargs)
            grid_index += grid_len

        return p_dic









        # for i in range(grid_num):
        #     column = numpy.array([])
        #     st = non_grid_dims + meteo_index
        #     ed = non_grid_dims + meteo_index + meteo_len
        #     for j in range(st, ed):
        #         # column = one_dim_new_or_cat(column, self.matrix[:, j])
        #         print(j)
        #     # print(column.tolist())
        #     meteo_index += meteo_len
        #     # break

        # k_index = 0
        # k_counter = 0
        #
        # for i in it:
        #     c = self.matrix[:, i]
        #     if i < non_grid_dims:
        #         cmd_dic[i] = 0
        #     else:
        #         k = meteo_keys[k_index]
        #         cmd_dic[i] = k
        #         dim_dic[k] = one_dim_new_or_cat(dim_dic[k], c)
        #
        #         k_counter += 1
        #         if k_counter == int(grid_num):
        #             k_counter = 0
        #             k_index += 1
        #
        # for k, c in dim_dic.items():
        #     dim_param_dic[k] = norm.params(c, *args, **kwargs)
        #
        # for i, cmd in cmd_dic.items():
        #     c = self.matrix[:, i]
        #     if cmd == 0:
        #         rm = new_or_stack(rm, norm.f(c, *args, **kwargs))
        #     elif cmd == 1:
        #         rm = new_or_stack(rm, c)
        #     else:
        #         rm = new_or_stack(rm, norm.f_with_params(c, dim_param_dic[cmd], *args, **kwargs))
        #
        # return rm


if __name__ == "__main__":
    from matplotlib import pyplot

    with open(r"E:\code\github\private\private\db\tencnet\openmeteo\data\test_ecmwf.pfd", "rb") as f:
        pfd: ProvinceForecastData = pickle.loads(f.read())

    # print(pfd.cma_forecast_matrix[:,7].tolist())

    omf = OpenMeteoForecast(pfd)
    f = omf.non_stop_forecast("2024-10-10", ["test", "morning"])[6]
    #
    m = MeteoForecastMatrix(f)
    print(m.matrix.shape)
    print(len(ecmwf_new_energy_args))
    #
    print(m.norm_by_meteo_key(MinMax, ecmwf_new_energy_args))
    # print([TimeStamp(i) for i in f[:, 0]])

    # print(pfd["cma", "evening", 1, 1, "2024-10-1"][:,7].tolist())

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
import math
import tqdm

# 项目模块
from finance_utils.electrical2.spot.probabilistic.forecast.torch.transformer.vanilla_encoder import (
    VanillaTransformerEncoder, vanilla_transformer_tester, vanilla_transformer_trainer, vanilla_transformer_trainer2)

from finance_utils.electrical2.spot.probabilistic.forecast.power.data.meteo import PFDInterFace, ecmwf_new_energy_args, \
    cma_new_energy_args
from finance_utils.electrical2.spot.probabilistic.forecast.power.data.tr import TRInterFace
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import ProvinceForecastData
from finance_utils.electrical2.spot.probabilistic.forecast.torch.normalization.basic import Norm, MinMax, ZScore, \
    RobustScaler
from easy_datetime.timestamp import TimeStamp

# 外部模块
import numpy
import torch.nn as nn
import torch


# 代码块
class GridTwiceCookedEncoder:
    def __init__(self, meteo_matrix: numpy.ndarray, act_power: numpy.ndarray, target_day: str, first_forecast_day: str):
        self.m = meteo_matrix
        self.grid_norm_m = None
        self.power = act_power
        self.train_y = None
        self.test_power = self.power[24:]

        self.y_norm_param = numpy.array([])
        self.model_list = []
        self.twice_cooked_module = None
        self.target_day = TimeStamp(target_day)
        self.first_forecast_day = TimeStamp(first_forecast_day)
        self.days = int(((self.target_day - ["day", 1]) - self.first_forecast_day) // 24 // 3600)

        self.xnorm = None
        self.ynorm = None
        self.batch_size = None

        self.time_dim_len = None

    def set_batch_size(self, bs: int = None):
        if bs is None:
            self.batch_size = self.days * 2
        else:
            self.batch_size = bs

    def grid_and_norm(self, xnormer: Type[Norm], ynormer: Type[Norm], meteo_arg: list = cma_new_energy_args):
        time_encoded, self.time_dim_len = PFDInterFace.time_periodic_encoding(self.m)
        self.grid_norm_m = PFDInterFace.grid_separate(
            # PFDInterFace.norm_by_meteo_key(time_encoded, xnormer, meteo_arg, self.time_dim_len),
            numpy.apply_along_axis(
                xnormer.f, axis=0, arr=time_encoded
            ),
            meteo_arg,
            self.time_dim_len
        )
        self.y_norm_param = ynormer.params(self.power[:-24])
        self.train_y = ynormer.f(self.power[:-24])
        self.xnorm = xnormer
        self.ynorm = ynormer

    def model_train(
            self,
            d_model: int = 128,
            dim_feedforward: int = 1024,
            nhead: int = 4,
            output_size: int = 1,
            num_layers: int = 8,
            dropout: float = 0.1,
            lr: float = 1e-3,
            epoch_num: int = 50,
            *args, **kwargs):
        self.model_list = []
        input_size = self.grid_norm_m[0].shape[1]

        for i, g in enumerate(self.grid_norm_m):
            temp_model = VanillaTransformerEncoder(
                input_size=input_size,
                output_size=output_size,
                num_layers=num_layers,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            trained_model = vanilla_transformer_trainer2(
                temp_model,
                torch.Tensor(g[:-24]),
                torch.Tensor(self.train_y),
                self.batch_size,
                epoch_num,
                lr=lr,
                loser=nn.L1Loss,
                shuffle=True,
            )
            self.model_list.append(trained_model)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self.model_list))

    def first_pot_data(self, model_filename: str = None, grid_data: list = None):
        if model_filename is None:
            model_list = self.model_list
        else:
            with open(model_filename, "rb") as f:
                model_list = pickle.loads(f.read())

        if grid_data is None:
            grid_data = [i[:-24] for i in self.grid_norm_m]
        else:
            pass

        first_pot_list = []
        for i, m in enumerate(model_list):
            first_pot = vanilla_transformer_tester(
                m, torch.Tensor(grid_data[i]), self.batch_size
            ).reshape(-1).cpu().numpy()
            #
            # first_pot = self.ynorm.invert(first_pot, self.p)
            # first_pot = self.ynorm.f(first_pot)

            first_pot_list.append(first_pot)

        first_pot_array = numpy.array(first_pot_list).T
        return first_pot_array

    def twice_cook(
            self,
            first_pot_matrix: numpy.ndarray,
            d_model: int = 128,
            dim_feedforward: int = 1024,
            nhead: int = 4,
            output_size: int = 1,
            num_layers: int = 8,
            batch_size: int = None,
            dropout: float = 0.1,
            lr: float = 1e-3,
            epoch_num: int = 50,
    ):
        input_size = first_pot_matrix.shape[1]
        if batch_size is None:
            batch_size = self.batch_size
        else:
            pass

        model = VanillaTransformerEncoder(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        twice_cooked_model = vanilla_transformer_trainer2(
            model,
            torch.Tensor(first_pot_matrix),
            torch.Tensor(self.train_y),
            # self.batch_size,
            batch_size,
            epoch_num,
            lr=lr,
            loser=nn.MSELoss,
            shuffle=True
        )

        return twice_cooked_model


if __name__ == "__main__":
    from matplotlib import pyplot

    root = r"E:\code\github\private\private\db\tencnet\openmeteo\data"

    with open(f"{root}\\shanxi_1.pfd", "rb") as f:
        pfd: ProvinceForecastData = pickle.loads(f.read())

    # with open("modellist1.ml", "rb") as f:
    #     ml = pickle.loads(f.read())

    gtce = GridTwiceCookedEncoder(
        PFDInterFace.moving_forecast(pfd, "2024-10-21", ["cma", "evening"], "2024-10-1"),
        TRInterFace.hourly_actual_range("shanxi", "2024-10-1", "2024-10-21")["power"],
        "2024-10-21",
        "2024-10-1"
    )
    # gtce.norm(ZScore, ZScore, cma_new_energy_args)
    # gtce.model_train()
    # gtce.save("modellist1.ml")

    gtce.set_batch_size(gtce.days * 4)
    gtce.grid_and_norm(ZScore, ZScore, cma_new_energy_args)
    # gtce.model_train()
    # gtce.save("modellist1.ml")
    # with open("modellist1.ml", "rb") as f:
    #     ml = pickle.loads(f.read())

    # gtce.first_pot_data("modellist1.ml")
    m = gtce.twice_cook(
        numpy.column_stack((
            gtce.grid_norm_m[0][:-24][:, :10], gtce.first_pot_data("modellist1.ml")
        ))
    )
    first_pot_predict = gtce.first_pot_data("modellist1.ml", [i[24:] for i in gtce.grid_norm_m])

    mp = vanilla_transformer_tester(
        m,
        torch.Tensor(
            numpy.column_stack((
                gtce.grid_norm_m[0][24:][:, :10], first_pot_predict
            ))
        ),
        gtce.batch_size
    )

    mp = ZScore.invert(mp.reshape(-1).cpu().numpy(), gtce.y_norm_param)
    pyplot.plot(gtce.test_power[-24 * 3:])
    pyplot.plot(mp[-24 * 3:])
    pyplot.show()

    for i in range(first_pot_predict.shape[1]):
        p = ZScore.invert(first_pot_predict[:, i], gtce.y_norm_param)
        pyplot.plot(p[-24 * 3:], color="red")
    pyplot.plot(gtce.test_power[-24 * 3:])
    pyplot.show()

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
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import ProvinceForecastData

# 外部模块
import numpy

# 代码块


if __name__ == "__main__":
    root = r"E:\code\github\private\private\db\tencnet\openmeteo\data"

    province = input("province")
    stdt = input("stdt")
    eddt = input("eddt")
    file_name = input("file_name")
    save_cma = input("save_cma")
    save_ecmwf = input("save_ecmwf")

    pfd = ProvinceForecastData(province, stdt, eddt)

    if save_cma == "y":
        pfd.add_new_energy_forecast("cma")

    if save_ecmwf == "y":
        pfd.add_new_energy_forecast("ecmwf")

    pfd.save(f"{root}\\{file_name}.pfd")

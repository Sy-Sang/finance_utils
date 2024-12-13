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

# 项目模块
from private.db.tencnet.openmeteo.china_openmeteo_forecast3 import ProvinceForecastData, new_or_cat, new_or_stack
from finance_utils.electrical2.spot.probabilistic.forecast.torch.normalization.basic import MinMax, ZScore, RobustScaler

from finance_utils.electrical2.spot.probabilistic.forecast.power.data.meteo import PFDInterFace, ecmwf_new_energy_args, \
    cma_new_energy_args
from finance_utils.electrical2.spot.probabilistic.forecast.power.data.tr import TRInterFace

# 外部模块
import numpy
import pandas

# 代码块


if __name__ == "__main__":
    pass

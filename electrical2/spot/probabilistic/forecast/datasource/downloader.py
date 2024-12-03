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

    pfd = ProvinceForecastData("china", "2024-10-1", "2024-10-31")
    if os.path.exists(f"{root}\\china_cma.pfd"):
        pass
    else:
        pfd.add_new_energy_forecast("cma")
        pfd.save(f"{root}\\china_cma.pfd")

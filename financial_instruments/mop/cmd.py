#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""reaction cmd 编译器"""

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
from typing import *
from collections import namedtuple

# 项目模块
from financial_instruments.mop.fm import *

# 外部模块
import numpy


# 代码块

class CommandCompiler:
    def __init__(self, cmd:str):
        self.cmd = cmd

    def __call__(self, *args, **kwargs):
        pass

class DeleteSubModule(CommandCompiler):
    def __init__(self):
        super().__init__("delete submodule")

    def __call__(self, act: ReactionEvent, dic: Dict, *args, **kwargs):
        pass



if __name__ == "__main__":
    pass

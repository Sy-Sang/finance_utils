#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""financial module"""

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
from abc import abstractmethod, ABC
import uuid
from collections.abc import Iterable

# 项目模块
from finmorph.mop.numeraire import *

# 外部模块
import numpy


# 代码块

class FinancialQuantity(float):
    def __new__(cls, value):
        return super().__new__(cls, value)


class FinancialModule(ABC):
    def __init__(self, *args, **kwargs):
        id = uuid.uuid4()
        object.__setattr__(self, "_id", str(id))
        object.__setattr__(self, "_submodules", {})
        object.__setattr__(self, "_quantity", FinancialQuantity(1))
        self._set_external_attr(*args, **kwargs)

    def __setattr__(self, name, value):
        # 1. 注册子 FinancialModule
        if isinstance(value, FinancialModule):
            self._submodules[name] = value

        object.__setattr__(self, name, value)

    def _set_external_attr(self, *args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, FinancialQuantity):
                self._quantity = float(arg)

        for k, v in kwargs.items():
            if isinstance(v, FinancialQuantity):
                self._quantity = float(v)

    def copy(self, keep_id=True):
        new_module = copy.deepcopy(self)
        if keep_id:
            pass
        else:
            new_module._id = str(uuid.uuid4())
        return new_module

    @property
    def id(self):
        return self._id

    @property
    def submodules(self) -> Dict:
        return self._submodules

    @property
    def quantity(self):
        return float(self._quantity)

    def __repr__(self):
        return f"{type(self).__name__}:{self.id}"

    @classmethod
    def composite(cls, *modules: "FinancialModule"):
        return PortfolioModule(*modules)

    def append(self, other):
        if isinstance(other, FinancialModule):
            setattr(self, f"{other.id}", other.copy())
        else:
            raise RuntimeError(f"Cannot append type of {type(other).__name__}")

    def appended(self, other):
        new_instance = self.copy()
        new_instance.append(other)
        return new_instance

    def value(self, numeraire_class, *args, **kwargs):
        values = []
        for m in self.submodules.values():
            sub_value = m.value(numeraire_class, *args, **kwargs)
            if not isinstance(sub_value, list):
                raise TypeError("reaction must return list[ReactionEvent]")
            for sv in sub_value:
                if sv.__class__ in numeraire_class:
                    values.append(sv)

        self_value = self._value(*args, **kwargs)
        if isinstance(self_value, list):
            for sv in self_value:
                if sv.__class__ in numeraire_class:
                    values.append(sv)

        return values

    def _value(self, *args, **kwargs):
        return [self.quantity]


class PortfolioModule(FinancialModule):
    def __init__(self, *args: FinancialModule):
        super().__init__()
        for i, m in enumerate(args):
            setattr(self, f"{m.id}", m)

    def _value(self, *args, **kwargs):
        return []


if __name__ == "__main__":
    fm1 = FinancialModule(FinancialQuantity(10))
    fm2 = FinancialModule().appended(FinancialModule())
    fm3 = FinancialModule()
    fm = FinancialModule.composite(fm1, fm2).appended(fm3)
    print(fm)
    print(fm.submodules)
    print(fm.value([float]))

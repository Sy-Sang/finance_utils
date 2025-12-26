#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""financial_object基类"""

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

# 外部模块
import numpy


# 代码块

class Leg(ABC):
    def __init__(self, *args, **kwargs):
        self.id = str(uuid.uuid4())

    @abstractmethod
    def forward(self, **kwargs):
        "在当前 context 下“当期发生的原子事件”"
        pass


class FinancialObject(ABC):
    """
    金融对象：可枚举、可组合的金融表达式节点
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def assemble(self, *args, **kwargs) -> Iterable[Leg]:
        pass


class FinancialInstance:
    def __init__(self, elements=None, financial_object: FinancialObject = None, *args, **kwargs):
        self.legs: list[Leg] = []

        if financial_object is not None and elements is not None:
            raise ValueError("Provide either financial_object or elements, not both")

        elif financial_object is not None:
            assembled = financial_object.assemble(*args, **kwargs)
            self._extend(assembled)

        elif elements is not None:
            self._extend(elements)

    def __repr__(self):
        return f"{self.legs}"

    def _extend(self, obj):
        if isinstance(obj, FinancialInstance):
            self.legs.extend(obj.legs)

        elif isinstance(obj, Leg):
            self.legs.append(obj)

        elif isinstance(obj, Iterable):
            for item in obj:
                self._extend(item)

        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

    def __add__(self, other: "FinancialInstance") -> "FinancialInstance":
        if not isinstance(other, FinancialInstance):
            return NotImplemented
        return FinancialInstance(elements=[self, other])

    @classmethod
    def cat(cls, *args):
        return FinancialInstance(elements=args)

    def forward(self, **kwargs):
        forward_result_set = []
        for i, leg in enumerate(self.legs):
            forward_result_set.append(leg.forward(**kwargs))
        return forward_result_set


if __name__ == "__main__":
    pass

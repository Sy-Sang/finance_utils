#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""金融模型"""

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

class Trade(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


class FinancialModule(ABC):
    def __init__(self, name: str, *args, **kwargs):
        object.__setattr__(self, "_id", uuid.uuid4())
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_submodules", {})
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_trades", {})

    def __setattr__(self, name, value):

        # 1. 注册子 FinancialModule
        if isinstance(value, FinancialModule):
            self._submodules[name] = value
            self._submodules[name].set_parent(self.name)

        elif isinstance(value, Trade):
            self._trades[name] = value

        object.__setattr__(self, name, value)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def submodules(self) -> Dict:
        return self._submodules

    @property
    def parent(self):
        return self._parent

    def set_parent(self, parent: "FinancialModule"):
        if self.parent is not None:
            raise RuntimeError("Parent already set")
        object.__setattr__(self, "_parent", parent)

    def __repr__(self):
        return f"{type(self).__name__}.{self.name}: {{submodules={self.submodules}}}"

    @classmethod
    def composite(cls, name, *modules: "FinancialModule"):
        return PortfolioModule(name, *modules)

    # def __add__(self, other: "FinancialModule") -> "FinancialModule":
    #     if not isinstance(other, FinancialModule):
    #         return NotImplemented
    #     return FinancialModule.composite(self, other)

    def append(self, other):
        if isinstance(other, FinancialModule):
            setattr(self, f"{other.name}", other.copy())
        else:
            raise RuntimeError(f"Cannot append type of {type(other).__name__}")

    def copy(self):
        new_module = copy.deepcopy(self)
        object.__setattr__(new_module, "_id", uuid.uuid4())
        return new_module

    def collect_reactions(self, *args, **kwargs) -> list["ReactionEvent"]:
        events = []

        for m in self.submodules.values():
            sub_events = m.collect_reactions(*args, **kwargs)
            if not isinstance(sub_events, list):
                raise TypeError("reaction must return list[ReactionEvent]")
            for e in sub_events:
                if not isinstance(e, ReactionEvent):
                    raise TypeError("All elements must be ReactionEvent")
            events.extend(sub_events)

        self_events = self.reaction(*args, **kwargs)
        if self_events:
            if not isinstance(self_events, ReactionEvent):
                raise TypeError("reaction must return ReactionEvent")
            events.append(self_events)

        return events

    def reaction(self, *args, **kwargs) -> Union["ReactionEvent", None]:
        """
        Override this in subclasses.
        Should return Event | List[Event] | None
        """
        return None

    def collect_trade(self, *args, **kwargs):
        events = []
        for m in self._submodules.values():
            sub_events = m.collect_trade(*args, **kwargs)
            if not isinstance(sub_events, list):
                raise TypeError("reaction must return list")
            events.extend(sub_events)

        self_events = self.trade(*args, **kwargs)
        if self_events:
            events.append(self_events)

        return events

    def trade(self, *args, **kwargs):
        return None


class PortfolioModule(FinancialModule):
    def __init__(self, name, *args: FinancialModule):
        super().__init__(name)
        for i, m in enumerate(args):
            setattr(self, f"{m.name}", m)


class ReactionEvent:
    def __init__(self, source=None, tag=None, back_to_market: bool = False, market_tag=None, data=None):
        self.source = source
        self.tag = tag
        self.back_to_market = back_to_market
        self.market_tag = market_tag
        self.data = data

    def __repr__(self):
        return (f"source={self.source}; "
                f"tag={self.tag}; "
                f"market_tag={self.market_tag}; "
                f"data={self.data}")


if __name__ == "__main__":
    fm1 = FinancialModule("fm1")
    fm2 = FinancialModule("fm2")
    fm = FinancialModule.composite("fm", fm1, fm2)
    print(fm)
    print(fm.id)
    print(fm.copy().collect_reactions())

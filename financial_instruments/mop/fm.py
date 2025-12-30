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

class TradeRequest:
    def __init__(self, trade_type: str, *args, **kwargs):
        self.trade_type = trade_type
        self.args = args
        self.kwargs = kwargs


class FinancialModule(ABC):
    def __init__(self, *args, **kwargs):
        id = uuid.uuid4()
        object.__setattr__(self, "_id", str(id))
        object.__setattr__(self, "_submodules", {})
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_trades", {})

    def __setattr__(self, name, value):

        # 1. 注册子 FinancialModule
        if isinstance(value, FinancialModule):
            self._submodules[name] = value
            # self._submodules[name].set_parent(self.name)

        elif isinstance(value, Trade):
            self._trades[name] = value

        object.__setattr__(self, name, value)

    @property
    def id(self):
        return self._id

    @property
    def submodules(self) -> Dict:
        return self._submodules

    @property
    def parent(self):
        return self._parent

    @property
    def trades(self):
        return self._trades

    def set_parent(self, parent: "FinancialModule"):
        if self.parent is not None:
            raise RuntimeError("Parent already set")
        object.__setattr__(self, "_parent", parent)

    def __repr__(self):
        return (f"{type(self).__name__}.{self.id}: "
                f"{{submodules={self.submodules}}}")

    @classmethod
    def composite(cls, *modules: "FinancialModule"):
        return PortfolioModule(*modules)

    def append(self, other):
        if isinstance(other, FinancialModule):
            setattr(self, f"{other.id}", other.copy())
        else:
            raise RuntimeError(f"Cannot append type of {type(other).__name__}")

    def copy(self):
        new_module = copy.deepcopy(self)
        return new_module

    def collect_reactions(self, *args, **kwargs) -> list["ReactionEvent"]:
        events = []

        for m in self.submodules.values():
            sub_events = m.collect_reactions(*args, **kwargs)
            if not isinstance(sub_events, list):
                raise TypeError("reaction must return list[ReactionEvent]")
            for e in sub_events:
                if isinstance(e, ReactionEvent):
                    events.append(e)

        self_events = self.reaction(*args, **kwargs)
        if self_events:
            if isinstance(self_events, ReactionEvent):
                events.append(self_events)
            elif isinstance(self_events, list):
                for e in self_events:
                    if isinstance(e, ReactionEvent):
                        events.append(e)
            else:
                raise TypeError("reaction must return ReactionEvent")

        return events

    def reaction(self, *args, **kwargs) -> Union["ReactionEvent", None]:
        """
        Override this in subclasses.
        Should return Event | List[Event] | None
        """
        return None

    def collect_trades(self, f: TradeRequest):
        events = []
        for m in self._submodules.values():
            sub_events = m.collect_trades(f)
            if not isinstance(sub_events, list):
                raise TypeError("reaction must return list")
            events.extend(sub_events)

        if f.trade_type in self.trades.keys():
            self_events = self.trades[f.trade_type](*f.args, **f.kwargs)
            if self_events:
                events.append(self_events)
        else:
            pass

        return events


class PortfolioModule(FinancialModule):
    def __init__(self, *args: FinancialModule):
        super().__init__()
        for i, m in enumerate(args):
            setattr(self, f"{m.id}", m)


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


class Trade:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    fm1 = FinancialModule()
    fm2 = FinancialModule()
    fm = FinancialModule.composite(fm1, fm2)
    print(fm)
    print(fm.id)
    print(fm.copy().id)

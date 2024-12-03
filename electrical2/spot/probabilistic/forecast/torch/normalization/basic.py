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
from abc import ABC, abstractmethod

# 项目模块

# 外部模块
import numpy


# 代码块

class Norm:
    """归一化接口"""

    @classmethod
    def f(cls, xlist, *args, **kwargs) -> numpy.ndarray:
        pass

    @classmethod
    def f_with_params(cls, xlist, p: Union[list, tuple, numpy.ndarray], *args, **kwargs):
        pass

    @classmethod
    def params(cls, xlist, *args, **kwargs) -> numpy.ndarray:
        pass

    @classmethod
    def invert(cls, xlist, p, *args, **kwargs) -> numpy.ndarray:
        pass


class MinMax(Norm):
    @classmethod
    def f(cls, xlist: Union[list, tuple, numpy.ndarray],
          a: Union[float, int, numpy.floating] = 0,
          b: Union[float, int, numpy.floating] = 1,
          eps: Union[float, int, numpy.floating] = 0,
          *args, **kwargs):
        xarray = numpy.array(xlist).astype(float)
        if max(xarray) == min(xarray):
            yarray = numpy.zeros(len(xlist))
            return yarray
        else:
            yarray = (xarray - min(xarray)) * (b - a - 2 * eps) / (max(xarray) - min(xarray)) + a + eps
            return yarray

    @classmethod
    def f_with_params(
            cls, xlist: Union[list, tuple, numpy.ndarray],
            p,
            a: Union[float, int, numpy.floating] = 0,
            b: Union[float, int, numpy.floating] = 1,
            eps: Union[float, int, numpy.floating] = 0,
            *args, **kwargs):
        xarray = numpy.array(xlist).astype(float)
        if max(xarray) == min(xarray):
            yarray = numpy.zeros(len(xlist))
            return yarray
        else:
            p_min = p[0]
            p_max = p[1]
            yarray = (xarray - p_min) * (b - a - 2 * eps) / (p_max - p_min) + a + eps
            return yarray

    @classmethod
    def params(cls, xlist: Union[list, tuple, numpy.ndarray],
               a: Union[float, int, numpy.floating] = 0,
               b: Union[float, int, numpy.floating] = 1,
               eps: Union[float, int, numpy.floating] = 0,
               *args, **kwargs):
        xarray = numpy.array(xlist).astype(float)
        if max(xarray) == min(xarray):
            numpy.array([])
        else:
            return numpy.array([min(xarray), max(xarray), a, b, eps]).astype(float)

    @classmethod
    def invert(cls, ylist, p, *args, **kwargs):
        yarray = numpy.array(ylist)
        xmin = p[0]
        xmax = p[1]
        a = p[2]
        b = p[3]
        eps = p[4]
        return (yarray - a - eps) * (xmax - xmin) / (a + b + 2 * eps) + xmin


class ZScore(Norm):
    @classmethod
    def f(cls, x: Union[list, tuple, numpy.ndarray],
          miu: float = 0,
          sigma: float = 1,
          *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            y = numpy.array([miu] * len(x))
            return y
        else:
            z = (numpy.array(x) - numpy.mean(x)) / numpy.std(x, ddof=1)
            y = z * sigma + miu
            return y

    @classmethod
    def f_with_params(cls, x: Union[list, tuple, numpy.ndarray], p,
                      miu: float = 0,
                      sigma: float = 1,
                      *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            y = numpy.array([miu] * len(x))
            return y
        else:
            p_mean = p[0]
            p_std = p[1]
            z = (numpy.array(x) - p_mean) / p_std
            y = z * sigma + miu
            return y

    @classmethod
    def params(cls, x: Union[list, tuple, numpy.ndarray],
               miu: float = 0,
               sigma: float = 1,
               *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            numpy.array([])
        else:
            return numpy.array([numpy.mean(x), numpy.std(x, ddof=1), miu, sigma]).astype(float)

    @classmethod
    def invert(cls, y: Union[list, tuple, numpy.ndarray], p, *args, **kwargs) -> numpy.ndarray:
        x_miu = p[0]
        x_std = p[1]
        miu = p[2]
        sigma = p[3]
        z = (numpy.array(y) - miu) / sigma
        x = z * x_std + x_miu
        return x


class RobustScaler(Norm):
    @classmethod
    def f(cls, x: Union[list, tuple, numpy.ndarray], q0=25, *args, **kwargs):
        median = numpy.median(x)
        q1 = numpy.percentile(x, q0)
        q3 = numpy.percentile(x, 100 - q0)
        iqr = q3 - q1
        if iqr == 0:
            y = numpy.zeros(len(x))
            return y
        else:
            y = (numpy.array(x) - median) / iqr
            return y

    @classmethod
    def f_with_params(cls, x: Union[list, tuple, numpy.ndarray], p, q0=25, *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            y = numpy.zeros(len(x))
            return y
        else:
            median = p[0]
            q1 = p[1]
            q3 = p[2]
            iqr = p[3]
            y = (numpy.array(x) - median) / iqr
            return y

    @classmethod
    def params(cls, x: Union[list, tuple, numpy.ndarray], q0=25, *args, **kwargs):
        if numpy.std(x, ddof=1) == 0:
            return numpy.array([])
        else:
            median = numpy.median(x)
            q1 = numpy.percentile(x, q0)
            q3 = numpy.percentile(x, 100 - q0)
            iqr = q3 - q1
            return [median, q1, q3, iqr]

    @classmethod
    def invert(cls, y: Union[list, tuple, numpy.ndarray], p, *args, **kwargs):
        median, q1, q3, iqr = p
        return numpy.array(y) * iqr + median


if __name__ == "__main__":
    pass

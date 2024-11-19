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

# 外部模块
import numpy
import torch.nn as nn
import torch


# 代码块

class abc_lstm(nn.Module):
    """
    二次封装的lstm基类
    """

    def __init__(self, input_size: int, output_size: int, hidden_size, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple:
        """
        向前传播
        :param x:
        :param h:
        :param c:
        :return:
        """
        return x, h, c

    def hc_init(self, batch_size: int) -> tuple:
        """
        初始化h,c
        :param batch_size:
        :return:
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.cuda_device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.cuda_device)
        )


class vanilla_lstm(abc_lstm):
    """香草lstm"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int,
                 acf: Type[nn.Module] = nn.ReLU):
        super(vanilla_lstm, self).__init__(input_size, output_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.acf = acf()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple:
        output, (hidden, cell) = self.lstm(x, (h, c))
        output = self.linear(output)
        output = self.acf(output)
        return output, hidden.detach(), cell.detach()

class vanilla_trainer:
    def __init__(self, model:nn.Module, lf:Type[nn.Module]):

    def train(self, lr:float, batch_size:int, epoch_size:int):


if __name__ == "__main__":
    import tqdm
    from matplotlib import pyplot

    batch_size = 2

    vl = vanilla_lstm(1, 32, 1, 4, nn.Tanh)
    x = torch.Tensor(numpy.sin(numpy.arange(-3, 3, 0.01))).reshape(-1, batch_size, 1).to(vl.cuda_device)
    y = torch.Tensor(numpy.cos(numpy.arange(-3, 3, 0.01))).reshape(-1, batch_size, 1).to(vl.cuda_device)
    tx = torch.Tensor(numpy.sin(numpy.arange(0, 6, 0.01))).reshape(-1, batch_size, 1).to(vl.cuda_device)
    ty = torch.Tensor(numpy.cos(numpy.arange(0, 6, 0.01))).reshape(-1, batch_size, 1).to(vl.cuda_device)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(vl.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    h, c = vl.hc_init(batch_size)
    for epoch in tqdm.trange(200):
        optimizer.zero_grad()
        # h, c = vl.hc_init(batch_size)

        outputs, h, c = vl(x, h, c)
        loss = loss_fn(outputs, y)

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_tensor, _0, _1 = vl(tx, h, c)

    pyplot.plot(y_tensor[:, 0, 0].cpu().numpy(), label='Predicted')  # 取第一个样本
    pyplot.plot(ty[:, 0, 0].cpu().numpy(), label='Actual')  # 取第一个样本
    pyplot.legend()
    pyplot.show()

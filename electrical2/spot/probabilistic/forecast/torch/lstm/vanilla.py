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
print(torch.__version__)


class VanillaLstm(nn.Module):
    """香草lstm"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int,
                 acf: Type[nn.Module] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.acf = acf() if acf is not None else None
        self.cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.cuda_device)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple:
        output, (hidden, cell) = self.lstm(x, (h, c))
        output = self.linear(output)
        if self.acf is not None:
            output = self.acf(output)
        else:
            pass
        return output, hidden, cell

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


def vanilla_lstm_trainer(model: VanillaLstm, x: torch.Tensor, y: torch.Tensor, batch_size: int, lr: float = 1e-6,
                         lf: Type[nn.Module] = nn.MSELoss, epoch_size: int = 200, show_tqdm: bool = False):
    """香草lstm训练"""

    x = x.reshape(-1, batch_size, model.input_size).to(model.cuda_device)
    y = y.reshape(-1, batch_size, model.output_size).to(model.cuda_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = lf()

    h, c = model.hc_init(batch_size)
    iterator = tqdm.trange(epoch_size) if show_tqdm else range(epoch_size)
    for _ in iterator:
        optimizer.zero_grad()
        outputs, h, c = model(x, h.detach(), c.detach())
        loss = loss_f(outputs, y)
        loss.backward()
        optimizer.step()

    return model, h, c


def vanilla_tester(trained_model: VanillaLstm, test_x: torch.Tensor, test_y: torch.Tensor, batch_size: int, *args):
    """香草lstm测试"""
    test_x = test_x.reshape(-1, batch_size, trained_model.input_size).to(trained_model.cuda_device)
    test_y = test_y.reshape(-1, batch_size, trained_model.output_size).to(trained_model.cuda_device)
    with torch.no_grad():
        y_tensor, _0, _1 = trained_vl(test_x, args[0], args[1])

    return y_tensor.reshape(-1), test_y.reshape(-1)


if __name__ == "__main__":
    import tqdm
    from matplotlib import pyplot

    batch_size = 10

    vl = VanillaLstm(2, 64, 1, 4, None)

    x1 = torch.Tensor(numpy.sin(numpy.arange(-6, 6, 0.01)))
    x2 = torch.Tensor(numpy.arange(-6, 6, 0.01))
    x = torch.stack((x2, x1))

    y = torch.Tensor(numpy.cos(numpy.arange(-6, 6, 0.01)))

    tx1 = torch.Tensor(numpy.sin(numpy.arange(0, 12, 0.01)))
    tx2 = torch.Tensor(numpy.arange(0, 12, 0.01))
    tx = torch.stack((tx2, tx1))

    ty = torch.Tensor(numpy.cos(numpy.arange(0, 12, 0.01)))
    trained_vl, _0, _1 = vanilla_lstm_trainer(vl, x, y, batch_size, show_tqdm=True, lr=0.01)

    y, y_hat = vanilla_tester(trained_vl, tx, ty, batch_size, _0, _1)

    pyplot.plot(y, label='Predicted')
    pyplot.plot(y_hat, label='Actual')

    pyplot.legend()
    pyplot.show()

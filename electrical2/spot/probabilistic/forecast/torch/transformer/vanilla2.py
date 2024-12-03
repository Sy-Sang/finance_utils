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
import math
import tqdm

# 项目模块

# 外部模块
import numpy
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset


# 代码块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(1)  # (max_len, 1, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.encoding[:seq_len, :].to(x.device)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建一个位置偏置矩阵
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        seq_len, batch_size, _ = x.size()

        # 计算相对位置偏置 (i - j)
        position_bias = (torch.arange(seq_len, device=x.device).unsqueeze(0) -
                         torch.arange(seq_len, device=x.device).unsqueeze(1))

        # 位置偏置大小: [seq_len, seq_len]
        position_bias_expanded = position_bias.unsqueeze(1).expand(-1, batch_size, -1)
        position_bias_expanded = position_bias_expanded.to(torch.float32)

        result = x.unsqueeze(3).expand(-1, -1, -1, position_bias_expanded.shape[2]) + position_bias_expanded.unsqueeze(
            2)

        return result.mean(dim=-1)


class VanillaTransformer(nn.Module):
    """香草(多维输入, 一维输出)transformer"""

    def __init__(self, input_size: int, output_size: int, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        self.input_fc = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.relative_positional_encoding = RelativePositionalEncoding(d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, output_size)
        self.cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        seq_len, batch_size, _ = src.size()

        src = self.input_fc(src)
        tgt = self.input_fc(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        src = src + self.relative_positional_encoding(src)
        tgt = tgt + self.relative_positional_encoding(tgt)

        src_mask = src_mask if src_mask is not None else self.generate_square_subsequent_mask(src.size(0)).to(
            src.device)
        tgt_mask = tgt_mask if tgt_mask is not None else self.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device)

        transformer_output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        transformer_output = transformer_output + src
        transformer_output = self.norm(transformer_output)

        output = self.fc_out(transformer_output)
        return output


def vanilla_transformer_trainer(
        model: VanillaTransformer,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        batch_size: int,
        epochs: int,
        lr: float = 0.001,
        loser: Type[nn.Module] = nn.MSELoss,
        show_tqdm: bool = True
) -> VanillaTransformer:
    """香草transformer训练器"""
    model_t = model.to(model.cuda_device)
    x = x_tensor.reshape(-1, batch_size, model.input_size).to(model.cuda_device)
    y = y_tensor.reshape(-1, batch_size, model.output_size).to(model.cuda_device)

    optimizer = torch.optim.Adam(model_t.parameters(), lr=lr)
    loss_fn = loser()

    iterator = tqdm.trange(epochs) if show_tqdm else range(epochs)
    for _ in iterator:
        model_t.train()
        optimizer.zero_grad()

        tgt = torch.cat([
            torch.zeros(1, batch_size, model.input_size).to(model.cuda_device),
            x[:-1]
        ], dim=0).to(model.cuda_device)

        output = model_t(x, tgt)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    return model_t


def vanilla_transformer_trainer2(
        model: VanillaTransformer,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        batch_size: int,
        epochs: int,
        lr: float = 0.001,
        loser: Type[nn.Module] = nn.MSELoss,
        show_tqdm: bool = True
) -> VanillaTransformer:
    """香草transformer训练器"""
    model_t = model.to(model.cuda_device)

    x = x_tensor.reshape(-1, model.input_size).to(model.cuda_device)
    y = y_tensor.reshape(-1, model.output_size).to(model.cuda_device)

    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model_t.parameters(), lr=lr)
    loss_fn = loser()

    iterator = tqdm.trange(epochs) if show_tqdm else range(epochs)
    for _ in iterator:
        model_t.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            seq_len, batch_size_actual = x_batch.size()
            print(x_batch.shape)
            tgt = torch.cat([
                torch.zeros(1, model.input_size).to(model.cuda_device),
                x_batch[:-1]
            ], dim=0).to(model.cuda_device)
            output = model_t(x_batch, tgt)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

    return model_t


def vanilla_transformer_tester(
        model: VanillaTransformer,
        test_x_tensor: torch.Tensor,
        batch_size: int,
) -> torch.Tensor:
    """输出预测数据"""
    reshaped_test_x_tensor = test_x_tensor.reshape(-1, batch_size, model.input_size).to(model.cuda_device)
    model.eval()
    with torch.no_grad():
        test_src_input = reshaped_test_x_tensor.to(model.cuda_device)

        # # 初始的目标序列是全零的
        # test_tgt_input = torch.zeros(1, batch_size, model.input_size).to(model.cuda_device)

        test_tgt_input = torch.cat(
            [
                torch.zeros(1, batch_size, model.input_size).to(model.cuda_device),
                reshaped_test_x_tensor[:-1]
            ], dim=0
        ).to(model.cuda_device)

        predicted = model(test_src_input, test_tgt_input)
    return predicted


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import trange

    input_size = 2
    d_model = 8
    nhead = 2
    output_size = 1
    num_layers = 2

    batch_size = 6

    model = VanillaTransformer(input_size, output_size, num_layers, d_model, nhead, d_model * 4)

    x1 = torch.Tensor(numpy.sin(numpy.arange(-6, 6, 0.01)))
    x2 = torch.Tensor(numpy.arange(-6, 6, 0.01))
    # x = torch.stack((x1, x2)).reshape(-1, batch_size, input_size).to(model.cuda_device)
    # tgt = torch.cat([torch.zeros(1, batch_size, input_size), x[:-1]], dim=0).to(model.cuda_device)

    test_x1 = torch.Tensor(numpy.sin(numpy.arange(0, 12, 0.01)))
    test_x2 = torch.Tensor(numpy.arange(-6, 6, 0.01))
    # test_x = torch.stack((test_x1, test_x2)).reshape(-1, batch_size, input_size)

    y = torch.Tensor(numpy.cos(numpy.arange(-6, 6, 0.01)))
    test_y = numpy.cos(numpy.arange(0, 12, 0.01))

    trained_model = vanilla_transformer_trainer2(model, torch.stack((x1, x2)), y, batch_size, 100, 1e-2, nn.HuberLoss)
    predicted = vanilla_transformer_tester(trained_model, torch.stack((test_x1, test_x2)), batch_size)

    # Initialize model

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # loss_fn = nn.MSELoss()
    #
    # # Training loop
    # epochs = 100
    # for epoch in trange(epochs):
    #     model.train()
    #     optimizer.zero_grad()
    #
    #     output = model(x, tgt)
    #     loss = loss_fn(output, y)
    #     loss.backward()
    #     optimizer.step()
    #     # scheduler.step()
    #
    # # Evaluation
    # model.eval()
    # with torch.no_grad():
    #     test_src_input = test_x.to(model.cuda_device)
    #     test_tgt_input = torch.cat([torch.zeros(1, batch_size, input_size), test_x[:-1]], dim=0).to(model.cuda_device)
    #
    #     predicted = model(test_src_input, test_tgt_input)
    #
    # # Plot results
    plt.plot(test_y.reshape(-1), label='Actual')
    plt.plot(predicted.reshape(-1), label='Predicted')
    plt.legend()
    plt.show()

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
import math

# 项目模块

# 外部模块
import numpy
import torch.nn as nn
import torch


# 代码块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码张量，形状为 (max_len, d_model)
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 使用正弦和余弦函数填充位置编码张量
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        # 增加一个维度，使其适用于 batch_first=True 的输入，形状变为 (1, max_len, d_model)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # 获取输入序列的长度，并裁剪位置编码，使其与输入匹配
        seq_len = x.size(1)
        encoding = self.encoding[:, :seq_len, :].to(x.device)  # 确保位置编码与输入张量在同一设备上
        return x + encoding


class VanillaTransformer(nn.Module):
    """香草transformer"""

    def __init__(self, input_size: int, output_size: int, num_layers: int, d_model: int, nhead: int,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead

        # 输入全连接层，将输入维度转换为 d_model
        self.input_fc = nn.Linear(input_size, d_model)
        # 使用位置编码模块
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        # LayerNorm 层用于标准化 Transformer 输出
        self.norm = nn.LayerNorm(d_model)

        # 初始化 Transformer 模型，设置 batch_first=True
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True  # 设置 batch_first=True，使批次维度为第一个维度
        )
        # 输出全连接层，将 d_model 映射到输出维度
        self.fc_out = nn.Linear(d_model, output_size)
        # 检查是否可用 GPU，并将模型移动到相应的设备上
        self.cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.cuda_device)

    def generate_square_subsequent_mask(self, sz):
        # 生成掩码矩阵，用于屏蔽未来时间步
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src, tgt):
        # 将输入通过全连接层
        src = self.input_fc(src)
        tgt = self.input_fc(tgt)
        # 添加位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 生成源序列和目标序列的掩码，并移动到相应的设备上
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Transformer 的前向传播
        transformer_output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # 残差连接
        transformer_output = transformer_output + src
        # LayerNorm 标准化
        transformer_output = self.norm(transformer_output)

        # 通过全连接层得到最终输出
        output = self.fc_out(transformer_output)
        return output


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import trange

    input_size = 2
    d_model = 10
    nhead = 2
    output_size = 1
    num_layers = 2

    batch_size = 10

    # 初始化模型
    model = VanillaTransformer(input_size, output_size, num_layers, d_model, nhead)

    # 构造训练数据，确保数据形状符合 batch_first=True 的格式
    x1 = torch.Tensor(numpy.sin(numpy.arange(-6, 6, 0.01)))
    x2 = torch.Tensor(numpy.arange(-6, 6, 0.01))
    x = torch.stack((x1, x2)).reshape(batch_size, -1, input_size).to(model.cuda_device)
    tgt = torch.cat([torch.zeros(batch_size, 1, input_size), x[:, :-1, :]], dim=1).to(model.cuda_device)

    # 构造测试数据
    test_x1 = torch.Tensor(numpy.sin(numpy.arange(0, 12, 0.01)))
    test_x2 = torch.Tensor(numpy.arange(-6, 6, 0.01))
    test_x = torch.stack((test_x1, test_x2)).reshape(batch_size, -1, input_size)

    y = torch.Tensor(numpy.cos(numpy.arange(-6, 6, 0.01))).reshape(batch_size, -1, output_size).to(model.cuda_device)
    test_y = torch.Tensor(numpy.cos(numpy.arange(0, 12, 0.01))).reshape(batch_size, -1, output_size).to(
        model.cuda_device)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.MSELoss()

    # 训练循环
    epochs = 100
    for epoch in trange(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播，计算损失并反向传播
        output = model(x, tgt)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        # scheduler.step()

    # 评估模型
    model.eval()
    with torch.no_grad():
        # 构造测试输入，确保形状符合 batch_first=True 的格式
        test_src_input = test_x.to(model.cuda_device)
        test_tgt_input = torch.cat([torch.zeros(batch_size, 1, input_size), test_x[:, :-1, :]], dim=1).to(
            model.cuda_device)

        # 前向传播以获得预测结果
        predicted = model(test_src_input, test_tgt_input)
    #
    # 绘制预测结果与真实结果的对比
    plt.plot(test_y.reshape(-1).cpu(), label='Actual')
    plt.plot(predicted.reshape(-1).cpu(), label='Predicted')
    plt.legend()
    plt.show()

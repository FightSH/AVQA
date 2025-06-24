import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from torch.distributions.normal import Normal
from einops import rearrange, repeat
from audio_layers import Transformer_Layer
from audio_others import SparseDispatcher, series_decomp_multi, MLP

class AMS(nn.Module):

    """
    自适应多尺度稀疏混合专家模型(Adaptive Multi-scale Sparsity)
    实现了一种基于多尺度特征的混合专家模型，通过噪声门控机制选择合适的专家组合
    """


    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1, batch_norm=False):
        """
        初始化AMS模型

        参数:
            input_size: 输入特征维度
            output_size: 输出特征维度
            num_experts: 专家模型数量
            device: 计算设备
            num_nodes: 节点数量，默认为1
            d_model: Transformer模型维度，默认32
            d_ff: 前馈网络维度，默认64
            dynamic: 是否使用动态机制，默认False
            patch_size: 多尺度补丁大小列表，默认[8,6,4,2]
            noisy_gating: 是否使用噪声门控，默认True
            k: 每次选择的专家数量，默认4
            layer_number: 层数，默认1
            residual_connection: 是否使用残差连接，默认1
            batch_norm: 是否使用批归一化，默认False
        """
        super(AMS, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        # 初始线性变换，将节点特征压缩
        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        # 时间序列分解模型，用于提取趋势信息
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        # 初始化专家模型列表
        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        # 为每种patch大小创建对应的Transformer专家
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff,
                                                  dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                                                  patch_size=patch, factorized=True, layer_number=layer_number,
                                                  batch_norm=batch_norm))

        # 门控和噪声网络，用于专家选择
        # 注释掉的是参数化版本
        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Linear(input_size, num_experts)  # 噪声生成网络
        self.w_gate = nn.Linear(input_size, num_experts)  # 门控网络

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        # 噪声门控相关参数
        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()  # 用于生成正值噪声标准差
        self.softmax = nn.Softmax(1)  # 用于归一化门控权重
        self.register_buffer("mean", torch.tensor([0.0]))  # 标准正态分布参数
        self.register_buffer("std", torch.tensor([1.0]))  # 标准正态分布参数
        assert (self.k <= self.num_experts)  # 确保选择的专家数不超过总专家数


    def cv_squared(self, x):
        """
        计算变异系数的平方(coefficient of variation squared)
        用于衡量专家负载的平衡性，值越小表示负载越均衡

        参数:
            x: 输入张量
        返回:
            变异系数的平方
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)


    def _gates_to_load(self, gates):
        """
        计算每个专家的负载(被选中的次数)

        参数:
            gates: 门控权重矩阵
        返回:
            每个专家的负载
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        计算每个样本选择每个专家的概率
        用于计算专家的期望负载，以进行负载均衡

        参数:
            clean_values: 无噪声的门控值
            noisy_values: 加噪声后的门控值
            noise_stddev: 噪声标准差
            noisy_top_values: 噪声门控值中的top-k+1值
        返回:
            选择概率
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        # 计算阈值位置和阈值
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        # 使用正态分布计算概率
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def trend_decompose(self, x):
        """
        进行时间序列趋势分解
        输入处理：
            接收一个四维张量x作为输入特征
            通过x[:, :, :, 0]操作去除最后一个维度，将四维张量转换为三维张量
        趋势分解：
            调用self.trend_model(x)方法进行时间序列分解
            这个方法返回两个值，使用_忽略第一个返回值（可能是季节性成分），只保留第二个返回值trend（趋势成分）
            self.trend_model是在类初始化时创建的series_decomp_multi实例，使用多个不同的卷积核大小[4, 8, 12]进行多尺度分解
            特征增强：
        返回x + trend，将原始特征与提取出的趋势成分相加
        这种操作增强了原始数据中的趋势信息，有助于模型更好地捕捉长期变化
        参数:
            x: 输入特征
        返回:
            分解后的特征
        """
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        return x + trend


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        带噪声的Top-K门控机制，用于专家选择

        参数:
            x: 输入特征
            train: 是否处于训练模式
            noise_epsilon: 噪声项最小值
        返回:
            gates: 门控权重
            load: 专家负载
        """
        x = self.start_linear(x).squeeze(-1)

        # 生成门控逻辑值
        # clean_logits = x @ self.w_gate  # 参数化版本
        clean_logits = self.w_gate(x)

        # 训练时添加噪声以增加多样性
        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise  # 参数化版本
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 选择Top-K+1个专家（用于计算阈值）
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        # 保留前K个专家
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)  # 归一化门控权重

        # 构建稀疏门控矩阵
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 计算专家负载
        if self.noisy_gating and self.k < self.num_experts and train:
            # 训练时使用概率方法计算期望负载
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            # 测试时直接计算实际负载
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, v_q, loss_coef=1e-2):
        """
        前向传播函数

        参数:
            x: 输入特征  (音频特征)
            v_q: 视觉特征查询
            loss_coef: 负载均衡损失系数
        返回:
            output: 输出特征
            balance_loss: 负载均衡损失
        """
        # 时间序列分解
        new_x = self.trend_decompose(x)

        # 多尺度路由器
        gates, load = self.noisy_top_k_gating(new_x, self.training)

        # 计算负载均衡损失
        importance = gates.sum(0)  # 每个专家的总重要性
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)  # 计算负载变异系数
        balance_loss *= loss_coef  # 应用损失系数

        # 分发输入到各专家
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        # 收集各专家输出
        expert_outputs = [self.experts[i](expert_inputs[i], v_q)[0] for i in range(self.num_experts)]

        # 合并各专家输出
        output = dispatcher.combine(expert_outputs)

        # 添加残差连接
        if self.residual_connection:
            output = output + x
        return output, balance_loss


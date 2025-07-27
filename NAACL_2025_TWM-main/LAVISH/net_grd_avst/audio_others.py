import torch
import torch.nn as nn
import numpy as np
import math

import torch.fft as fft
from einops import rearrange, reduce, repeat


class SparseDispatcher(object):
    """
    稀疏分发器类，用于将输入数据分发给选中的专家网络，并合并专家的输出结果
    主要用于MoE（Mixture of Experts）架构中
    """

    def __init__(self, num_experts, gates):
        """
        初始化稀疏分发器

        参数:
            num_experts (int): 专家网络的总数
            gates (Tensor): 门控权重矩阵，形状为[batch_size, num_experts]
                           每行表示对应样本分配给各专家的权重
        """
        self._gates = gates
        self._num_experts = num_experts

        # 对专家进行排序
        # 找到所有非零门控值的位置
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # 分离专家索引和对应的行索引
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # 获取每个专家对应的批次索引
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # 计算每个专家需要处理的样本数量
        self._part_sizes = (gates > 0).sum(0).tolist()
        # 获取非零门控值
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        将输入数据分发给对应的专家网络

        参数:
            inp (Tensor): 输入数据，形状为[batch_size, ...]

        返回:
            tuple: 按照每个专家需要处理的样本数量分割的输入数据元组
        """
        # 根据批次索引扩展输入数据
        inp_exp = inp[self._batch_index].squeeze(1)
        # 按照每个专家的样本数量进行分割
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
        合并所有专家网络的输出结果

        参数:
            expert_out (list): 各专家网络的输出结果列表
            multiply_by_gates (bool): 是否使用门控权重进行加权

        返回:
            Tensor: 合并后的输出结果
        """
        # 将所有专家输出连接起来并进行指数运算（从log空间回到原始空间）
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            # 使用门控权重对专家输出进行加权
            stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)

        # 创建零张量用于存储合并结果
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        # 根据批次索引将处理结果合并回原始顺序
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # 为零值添加极小值避免log运算时出现NaN
        combined[combined == 0] = np.finfo(float).eps
        # 转换回log空间
        return combined.log()

    def expert_to_gates(self):
        """
        将非零门控值按照专家进行分割

        返回:
            tuple: 按专家分割的门控权重元组
        """
        # 按照每个专家的样本数量分割非零门控值
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    """
    多层感知机类，使用1x1卷积实现
    """

    def __init__(self, input_size, output_size):
        """
        初始化MLP

        参数:
            input_size (int): 输入通道数
            output_size (int): 输出通道数
        """
        super(MLP, self).__init__()
        # 使用1x1卷积作为全连接层
        self.fc = nn.Conv2d(in_channels=input_size,
                            out_channels=output_size,
                            kernel_size=(1, 1),
                            bias=True)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 输出张量
        """
        out = self.fc(x)
        return out


class moving_avg(nn.Module):
    """
    移动平均模块，用于提取时间序列的趋势成分
    """

    def __init__(self, kernel_size, stride):
        """
        初始化移动平均模块

        参数:
            kernel_size (int): 平均池化窗口大小
            stride (int): 步长
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 1D平均池化层
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        前向传播，计算移动平均

        参数:
            x (Tensor): 输入时间序列，形状为[batch_size, sequence_length, features]

        返回:
            Tensor: 移动平均结果
        """
        # 在时间序列两端进行填充以保持长度不变
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        # 转换维度以适应AvgPool1d (batch, features, sequence)
        x = self.avg(x.permute(0, 2, 1))
        # 转换回原始维度 (batch, sequence, features)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    时间序列分解模块，将序列分解为残差和趋势两部分
    """

    def __init__(self, kernel_size):
        """
        初始化序列分解模块

        参数:
            kernel_size (int): 移动平均窗口大小
        """
        super(series_decomp, self).__init__()
        # 移动平均模块用于提取趋势
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        前向传播，执行序列分解

        参数:
            x (Tensor): 输入时间序列

        返回:
            tuple: (residual, trend) 残差和趋势成分
        """
        moving_mean = self.moving_avg(x)  # 计算趋势成分
        res = x - moving_mean  # 计算残差成分
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    多尺度时间序列分解模块，使用多个不同窗口大小的移动平均
    """

    def __init__(self, kernel_size):
        """
        初始化多尺度序列分解模块

        参数:
            kernel_size (list): 多个移动平均窗口大小的列表
        """
        super(series_decomp_multi, self).__init__()
        # 创建多个不同窗口大小的移动平均模块
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        # 线性层用于融合不同尺度的结果
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        """
        前向传播，执行多尺度序列分解

        参数:
            x (Tensor): 输入时间序列

        返回:
            tuple: (residual, moving_mean) 残差和融合后的趋势成分
        """
        moving_mean = []
        # 计算多个不同尺度的移动平均
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))

        # 将多个移动平均结果连接起来
        moving_mean = torch.cat(moving_mean, dim=-1)
        # 使用线性层和softmax对不同尺度进行加权融合
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean  # 计算残差成分
        return res, moving_mean


class FourierLayer(nn.Module):
    """
    傅里叶变换层，用于时间序列的频域分析和预测
    """

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        """
        初始化傅里叶层

        参数:
            pred_len (int): 预测长度
            k (int): 保留的频率分量数量
            low_freq (int): 最低频率索引（去除直流分量）
            output_attention (bool): 是否输出注意力权重
        """
        super().__init__()
        self.pred_len = pred_len  # 预测长度
        self.k = k  # 保留的频率分量数
        self.low_freq = low_freq  # 最低频率
        self.output_attention = output_attention  # 是否输出注意力

    def forward(self, x):
        """
        前向传播，执行傅里叶变换和频域预测

        参数:
            x (Tensor): 输入时间序列，形状为[batch_size, time_steps, features]

        返回:
            tuple: (extrapolated_signal, attention_weights) 外推信号和注意力权重
        """
        # 如果需要输出注意力，则使用DFT前向传播
        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape  # 获取输入维度
        # 执行快速傅里叶变换
        x_freq = fft.rfft(x, dim=1)

        # 根据序列长度的奇偶性选择频率分量
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]  # 偶数长度
            f = fft.rfftfreq(t)[self.low_freq:-1]  # 获取对应的频率
        else:
            x_freq = x_freq[:, self.low_freq:]  # 奇数长度
            f = fft.rfftfreq(t)[self.low_freq:]  # 获取对应的频率

        # 选择最重要的k个频率分量
        x_freq, index_tuple = self.topk_freq(x_freq)
        # 构造频率张量
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        # 外推到预测长度
        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        """
        基于频率分量外推时间序列

        参数:
            x_freq (Tensor): 频率分量
            f (Tensor): 频率值
            t (int): 原始时间序列长度

        返回:
            Tensor: 外推后的时间序列
        """
        # 添加共轭频率分量以保证实数输出
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        # 构造时间点张量
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        # 计算幅度和相位
        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        # 重构时间序列信号
        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        # 对所有频率分量求和
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        """
        选择幅度最大的k个频率分量

        参数:
            x_freq (Tensor): 所有频率分量

        返回:
            tuple: (selected_freq, index_tuple) 选中的频率分量和索引
        """
        # 选择幅度最大的k个频率分量
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        # 构造网格索引
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        """
        使用离散傅里叶变换的前向传播（用于注意力计算）

        参数:
            x (Tensor): 输入时间序列

        返回:
            tuple: (output, attention_weights) 输出和注意力权重
        """
        T = x.size(1)  # 时间序列长度

        # 构造DFT和IDFT矩阵
        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        # 执行傅里叶变换
        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        # 根据序列长度奇偶性选择频率分量
        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        # 选择最重要的k个频率分量
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        # 构造DFT和IDFT矩阵并应用掩码
        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        # 应用频率选择掩码
        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        # 计算注意力权重并应用
        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')
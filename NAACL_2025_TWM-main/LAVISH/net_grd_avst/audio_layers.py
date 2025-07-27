import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math

import torch
import torch.nn as nn


# 注意：导入部分省略，假设 torch.nn.functional as F, numpy as np, math 已正确导入

# --- 位置编码类 ---

class PositionalEmbedding(nn.Module):
    """
    正弦/余弦位置编码 (Sinusoidal Positional Encoding)。
    为序列中的每个位置生成唯一的、固定的向量表示，以注入序列顺序信息。
    这种编码方式不包含可学习参数。
    """

    def __init__(self, d_model, n_position=1024):
        """
        初始化位置编码层。
        :param d_model: 模型的维度大小，必须与词嵌入维度一致。
        :param n_position: 预先计算的最大序列长度。
        """
        super(PositionalEmbedding, self).__init__()

        # register_buffer 将 pos_table 注册为模型的缓冲区（非参数）。
        # 缓冲区在模型保存/加载时会被包含，但不会被优化器更新。
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model))

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        """
        生成正弦/余弦位置编码表。
        :param n_position: 序列的最大长度。
        :param d_model: 模型维度。
        :return: 形状为 [1, n_position, d_model] 的位置编码表。
        """
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            """
            计算给定位置对应的所有维度的角度值。
            公式: pos / 10000^(2i/d_model)
            """
            # 对于每个维度 hid_j，计算其对应的角度
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        # 为每个位置计算角度向量
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # 偶数索引的维度使用正弦函数
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        # 奇数索引的维度使用余弦函数
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        # 转换为 PyTorch 张量，并在最前面增加一个维度 (批次维度)
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        获取输入序列对应的位置编码。
        :param x: 输入序列张量，形状为 [batch_size, seq_len, ...]。
        :return: 对应的位置编码，形状为 [1, seq_len, d_model]。
        """
        # 截取与输入序列长度相匹配的部分，并进行克隆以避免影响原始缓冲区
        return self.pos_table[:, :x.size(1)].clone().detach()


class TokenEmbedding(nn.Module):
    """
    Token 嵌入层，将原始输入（如时间序列值）通过一维卷积映射到高维向量空间。
    通常用于将连续值或离散 token 转换为模型可以处理的稠密向量。
    """

    def __init__(self, c_in, d_model):
        """
        初始化 Token 嵌入层。
        :param c_in: 输入通道数（例如，时间序列的特征维度）。
        :param d_model: 输出的嵌入维度。
        """
        super(TokenEmbedding, self).__init__()
        # 根据 PyTorch 版本设置合适的 padding
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 使用一维卷积进行嵌入，kernel_size=3, circular padding 有助于处理序列边界
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        执行 Token 嵌入。
        :param x: 输入张量，形状为 [batch_size, seq_len, c_in]。
        :return: 嵌入后的张量，形状为 [batch_size, seq_len, d_model]。
        """
        # 卷积操作需要 [batch_size, c_in, seq_len] 的输入格式
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    固定的正弦/余弦嵌入层。
    与 PositionalEmbedding 类似，但实现为一个可复用的 nn.Embedding 层。
    常用于类别型时间特征（如月份、星期几）的嵌入，权重固定不更新。
    """

    def __init__(self, c_in, d_model):
        """
        初始化固定嵌入层。
        :param c_in: 词汇表大小（例如，小时数24）。
        :param d_model: 嵌入维度。
        """
        super(FixedEmbedding, self).__init__()

        # 创建一个空的权重矩阵
        w = torch.zeros(c_in, d_model).float()
        # 设置为不需要梯度（固定）
        w.require_grad = False

        # 生成位置索引和频率分母
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 填充权重矩阵：偶数列用 sin，奇数列用 cos
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 创建 Embedding 层并替换其权重
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        获取嵌入向量。
        :param x: 输入索引张量，形状为 [...,]。
        :return: 嵌入向量，形状为 [..., d_model]。
        """
        # 使用嵌入层并分离梯度（确保权重不被更新）
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    时间特征嵌入层。
    将时间戳分解为多个组成部分（如月、日、星期、小时、分钟），
    并为每个部分生成独立的嵌入向量，最后将它们相加。
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        初始化时间嵌入层。
        :param d_model: 每个时间部分的嵌入维度。
        :param embed_type: 嵌入类型 ('fixed' 或 'learned')。
        :param freq: 时间频率 ('t'=分钟, 'h'=小时, 'd'=天, 'w'=周, 'm'=月等)。
        """
        super(TemporalEmbedding, self).__init__()

        # 定义各种时间单位的类别数
        minute_size = 4  # 示例值
        hour_size = 24
        weekday_size = 7
        day_size = 32  # 日期范围 1-31，+1 作为缓冲
        month_size = 13  # 月份范围 1-12，+1 作为缓冲

        # 根据 embed_type 选择嵌入方式
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # 根据频率决定需要哪些时间部分的嵌入
        if freq == 't':  # 分钟级别
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        生成时间嵌入。
        :param x: 时间戳特征张量，形状为 [batch_size, seq_len, 5]。
                  最后一维按顺序为: [month, day, weekday, hour, minute]。
        :return: 融合后的时间嵌入，形状为 [batch_size, seq_len, d_model]。
        """
        x = x.long()  # 确保输入为长整型索引

        # 根据是否有对应的嵌入层来获取嵌入向量，并求和
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    时间特征连续值嵌入层。
    将经过预处理的连续时间特征（如 sin/cos 编码的月份、小时等）通过线性变换映射到 d_model 空间。
    """

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        初始化时间特征嵌入层。
        :param d_model: 输出嵌入维度。
        :param embed_type: 嵌入类型（此处固定为 'timeF'）。
        :param freq: 时间频率，决定输入特征的维度。
        """
        super(TimeFeatureEmbedding, self).__init__()

        # 根据频率映射到对应的输入特征维度
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        # 使用线性层进行映射
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        执行时间特征嵌入。
        :param x: 预处理后的连续时间特征，形状为 [batch_size, seq_len, d_inp]。
        :return: 嵌入后的特征，形状为 [batch_size, seq_len, d_model]。
        """
        return self.embed(x)


# --- 数据嵌入组合类 ---

class DataEmbedding(nn.Module):
    """
    完整的数据嵌入层。
    组合了 Token 嵌入、位置嵌入和时间嵌入，为模型提供丰富的输入表示。
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        初始化数据嵌入层。
        :param c_in: 输入特征维度。
        :param d_model: 嵌入后的统一维度。
        :param embed_type: 时间嵌入类型。
        :param freq: 时间频率。
        :param dropout: Dropout 比例。
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 根据 embed_type 选择时间嵌入方式
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        执行完整的数据嵌入。
        :param x: 原始输入数据，形状为 [batch_size, seq_len, c_in]。
        :param x_mark: 时间戳特征，形状为 [batch_size, seq_len, date_feat_dim]。
        :return: 嵌入后的数据，形状为 [batch_size, seq_len, d_model]。
        """
        # 将三种嵌入相加
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    """
    不包含时间嵌入的数据嵌入层 (wo_temp = without temporal)。
    仅组合 Token 嵌入和位置嵌入。
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        """
        初始化数据嵌入层（无时间嵌入）。
        :param c_in: 输入特征维度。
        :param d_model: 嵌入后的统一维度。
        :param dropout: Dropout 比例。
        """
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        执行数据嵌入（无时间嵌入）。
        :param x: 原始输入数据，形状为 [batch_size, seq_len, c_in]。
        :param x_mark: 时间戳特征（未使用）。
        :return: 嵌入后的数据，形状为 [batch_size, seq_len, d_model]。
        """
        # 仅将 Token 嵌入和位置嵌入相加
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# --- 位置编码生成函数 ---

def PositionalEncoding(q_len, d_model, normalize=True):
    """
    生成正弦/余弦位置编码的函数。
    与 PositionalEmbedding 类中的 `_get_sinusoid_encoding_table` 方法功能相似。
    :param q_len: 序列长度。
    :param d_model: 模型维度。
    :param normalize: 是否对编码进行归一化。
    :return: 形状为 [q_len, d_model] 的位置编码张量。
    """
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


# 别名
SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    """
    生成二维坐标风格的位置编码。
    :param q_len: 序列长度。
    :param d_model: 模型维度。
    :param exponential: 是否使用指数缩放。
    :param normalize: 是否归一化。
    :param eps: 均值接近零的阈值。
    :param verbose: 是否打印调试信息。
    :return: 形状为 [q_len, d_model] 的位置编码。
    """
    # 设置缩放因子
    x = .5 if exponential else 1
    i = 0
    # 迭代调整 x 以使编码均值接近 0
    for i in range(100):
        # 生成二维坐标编码
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    """
    生成一维坐标风格的位置编码。
    :param q_len: 序列长度。
    :param exponential: 是否使用指数缩放。
    :param normalize: 是否归一化。
    :return: 形状为 [q_len, 1] 的位置编码。
    """
    # 生成一维坐标编码
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    """
    通用的位置编码生成函数，根据参数返回不同类型的编码。
    :param pe: 位置编码类型字符串。
    :param learn_pe: 编码是否可学习。
    :param q_len: 序列长度。
    :param d_model: 模型维度。
    :return: 形状为 [q_len, d_model] 或 [q_len, 1] 的位置编码 (nn.Parameter)。
    """
    # Positional encoding
    if pe == None:
        # 无位置编码或用于测量影响
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        # 零填充 (可能用于特殊用途)
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        # 零初始化的位置编码
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        # 高斯分布初始化
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        # 均匀分布初始化
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        # 一维线性坐标编码
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        # 一维指数坐标编码
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        # 二维线性坐标编码
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        # 二维指数坐标编码
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        # 正弦/余弦位置编码
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    # 将编码包装为可学习参数 (如果 learn_pe 为 True)
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# --- 核心 Transformer Layer 类 ---

class Transformer_Layer(nn.Module):
    """
    自定义的 Transformer 层，包含 intra-patch (块内) 和 inter-patch (块间) 注意力机制。
    设计用于处理具有特定 patch 结构的时间序列或多变量数据。
    """

    def __init__(self, device, d_model, d_ff, num_nodes, patch_nums, patch_size, dynamic, factorized, layer_number, batch_norm):
        """
        初始化 Transformer_Layer。

        :param device: 计算设备 (e.g., 'cuda:0', 'cpu')。
        :param d_model: 模型的隐藏层维度。
        :param d_ff: 前馈网络 (FFN) 的中间层维度。
        :param num_nodes: 数据中的节点/变量数量 (例如，传感器数量)。
        :param patch_nums: 输入序列被分割成的 patch 数量。
        :param patch_size: 每个 patch 的大小（时间步长）。
        :param dynamic: (参数保留，但在此实现中未使用)。
        :param factorized: 是否使用权重分解机制 (影响注意力和权重生成)。
        :param layer_number: (参数保留，但在此实现中未使用)。
        :param batch_norm: 是否在注意力和 FFN 后使用批归一化。
        """
        super(Transformer_Layer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dynamic = dynamic # 未在forward中使用
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number # 未在forward中使用
        self.batch_norm = batch_norm

        # --- Intra-Patch Attention (块内注意力) ---
        # 为每个 patch 生成嵌入表示。注意：这里的 1536 是硬编码的输入维度。
        self.embeddings_generator = nn.ModuleList([
            nn.Sequential(*[nn.Linear(1536, self.d_model)]) for _ in range(self.patch_nums)
        ])

        self.intra_d_model = self.d_model
        # 定义 intra-patch 注意力模块
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=factorized)

        # 权重生成器：为 intra-patch 注意力生成特定和共享的权重/偏置
        # distinct: 为每个节点生成不同的权重
        self.weights_generator_distinct = WeightGenerator(
            self.intra_d_model, self.intra_d_model, mem_dim=16, num_nodes=num_nodes,
            factorized=factorized, number_of_weights=2
        )
        # shared: 为所有节点生成共享的权重
        self.weights_generator_shared = WeightGenerator(
            self.intra_d_model, self.intra_d_model, mem_dim=None, num_nodes=num_nodes,
            factorized=False, number_of_weights=2
        )

        # 线性层：用于调整 intra-patch 输出的序列长度，使其与原始输入对齐
        self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)

        # --- Inter-Patch Attention (块间注意力) ---
        self.stride = patch_size # 步长通常等于 patch 大小
        # inter-patch 注意力的维度是 intra 维度乘以 patch 大小
        self.inter_d_model = self.d_model * self.patch_size

        # inter-patch 的嵌入层
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)

        # 位置编码：为 patch 序列提供位置信息
        # positional_encoding 是一个辅助函数，返回一个 nn.Parameter
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_nums, d_model=self.inter_d_model)

        # Inter-patch 注意力的配置
        n_heads = self.d_model # 注意力头数
        d_k = self.inter_d_model // n_heads # 每个头的 Key 维度
        d_v = self.inter_d_model // n_heads # 每个头的 Value 维度
        self.inter_patch_attention = Inter_Patch_Attention(
            self.inter_d_model, self.inter_d_model, n_heads, d_k, d_v,
            attn_dropout=0, proj_dropout=0.1, res_attention=False
        )

        # --- 归一化层 ---
        # 注意：这里的维度 (self.d_model) 是针对 intra-patch 输出的
        self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2))
        self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2))

        # --- 前馈网络 (FFN) ---
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=True),
            nn.GELU(), # 激活函数
            nn.Dropout(0.2), # FFN 内部的 dropout
            nn.Linear(self.d_ff, self.d_model, bias=True)
        )

    def forward(self, x, query):
        """
        Transformer Layer 的前向传播。

        :param x: 输入张量，形状通常为 [batch_size, total_time_steps, num_nodes, d_model]。
                  其中 total_time_steps = patch_nums * patch_size。
        :param query: 查询向量，用于 intra-patch 注意力中的 Q。
        :return: out: 输出张量，形状与输入 x 相同。
                 attention: 注意力权重 (主要来自 inter-patch attention)。
        """
        new_x = x # 保存原始输入用于残差连接
        batch_size = x.size(0)
        intra_out_concat = None # 用于拼接所有 patch 的 intra 输出

        # 生成 intra-patch 注意力所需的权重和偏置
        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        # #### Intra-Patch Attention #####
        # 遍历每一个 patch
        for i in range(self.patch_nums):
            # 从输入 x 中提取第 i 个 patch 的数据
            # x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :] 的形状是
            # [batch_size, patch_size, num_nodes, d_model]
            t = x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :]

            # 为当前 patch 生成嵌入，并扩展到 batch 维度
            # query 通常是文本特征，通过 embeddings_generator[i] 映射到 d_model 空间
            intra_emb = self.embeddings_generator[i](query).expand(batch_size, -1, -1, -1)
            # 将 patch 数据 t 和嵌入 intra_emb 在时间维度上拼接
            # 形状变为 [batch_size, patch_size + 1, num_nodes, d_model]
            # intra_emb 作为额外的 token (类似 CLS token) 参与注意力计算
            t = torch.cat([intra_emb, t], dim=1)

            # 执行 intra-patch 注意力计算
            # Q 是 intra_emb, K 和 V 是拼接后的 t
            # 使用预生成的权重和偏置
            out, attention = self.intra_patch_attention(
                intra_emb, t, t,
                weights_distinct, biases_distinct,
                weights_shared, biases_shared
            )
            # out 的形状通常是 [batch_size, 1, num_nodes, d_model] (因为 Q 只有 intra_emb)

            # 将每个 patch 的输出拼接起来
            if intra_out_concat is None:
                intra_out_concat = out
            else:
                intra_out_concat = torch.cat([intra_out_concat, out], dim=1)
        # intra_out_concat 形状: [batch_size, patch_nums, num_nodes, d_model]

        # 调整 intra 输出的序列长度，使其与原始输入对齐
        # permute 和 Linear 操作是为了在 patch_nums 维度上进行线性变换
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)  # [B, D, N, P]
        intra_out_concat = self.intra_Linear(intra_out_concat)   # [B, D, N, P*S]
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)  # [B, P*S, N, D]
        # 此时 intra_out_concat 的时间步长 (P*S) 与原始输入 x 相同

        # #### Inter-Patch Attention ######
        # 使用 unfold 将输入 x 重新组织成 patch 格式
        # unfold(dimension=1, size=patch_size, step=stride) 在第1维(时间维)上滑动
        # x.unfold 后形状: [batch_size, patch_nums, num_nodes, d_model, patch_size]
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        # 调整维度顺序: [batch_size, num_nodes, patch_nums, d_model, patch_size]
        x = x.permute(0, 2, 1, 3, 4)

        b, nvar, patch_num, dim, patch_len = x.shape
        # 将最后两个维度 (d_model, patch_size) 合并
        # reshape 后形状: [batch_size * num_nodes, patch_nums, d_model * patch_size]
        x = torch.reshape(x, (b * nvar, patch_num, dim * patch_len))

        # 通过线性层将合并后的特征映射到 inter_d_model 空间
        x = self.emb_linear(x)
        # 添加位置编码并应用 dropout
        x = self.dropout(x + self.W_pos) # W_pos 已广播到 [batch_size * num_nodes, patch_nums, inter_d_model]

        # 执行 inter-patch 注意力 (Q=K=V=x)
        inter_out, attention = self.inter_patch_attention(Q=x, K=x, V=x)
        # inter_out 形状: [batch_size * num_nodes, patch_nums, inter_d_model]

        # 将 inter_out 重塑回原始的多维结构
        # [batch_size, num_nodes, patch_nums, inter_d_model]
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], inter_out.shape[-1]))
        # 进一步重塑，将 inter_d_model 分解回 (patch_size, d_model)
        # [batch_size, num_nodes, patch_nums, patch_size, d_model]
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], self.patch_size, self.d_model))
        # 最终调整维度顺序，得到与输入 x 相同的形状
        # [batch_size, patch_nums * patch_size, num_nodes, d_model] 即 [B, T, N, D]
        inter_out = torch.reshape(inter_out, (b, self.patch_size * self.patch_nums, nvar, self.d_model))

        # --- 残差连接与归一化 ---
        # 将原始输入、intra 输出和 inter 输出相加
        out = new_x + intra_out_concat + inter_out

        # 如果启用批归一化，则对注意力输出部分进行归一化
        if self.batch_norm:
            # reshape 以便 BatchNorm1d 处理
            out = self.norm_attn(out.reshape(b * nvar, self.patch_size * self.patch_nums, self.d_model))
            # reshape 回去
            out = out.reshape(b, self.patch_size * self.patch_nums, nvar, self.d_model)

        # --- FFN (前馈网络) ---
        out = self.dropout(out) # 应用 dropout
        # FFN 操作 + 残差连接
        out = self.ff(out) + out

        # 如果启用批归一化，则对 FFN 输出部分进行归一化
        if self.batch_norm:
            out = self.norm_ffn(out.reshape(b * nvar, self.patch_size * self.patch_nums, self.d_model))
            out = out.reshape(b, self.patch_size * self.patch_nums, nvar, self.d_model)

        return out, attention



class CustomLinear(nn.Module):
    """
    自定义线性层，支持标准线性变换和因子分解（factorized）线性变换。
    因子分解版本适用于特定的高维张量操作。
    """

    def __init__(self, factorized):
        """
        初始化自定义线性层。
        :param factorized (bool): 是否使用因子分解的线性变换。
                              如果为 True，则权重形状和计算方式不同。
        """
        super(CustomLinear, self).__init__()
        self.factorized = factorized # 存储因子分解标志

    def forward(self, input, weights, biases):
        """
        执行前向传播。
        :param input (Tensor): 输入张量。
        :param weights (Tensor): 权重张量。
        :param biases (Tensor): 偏置张量。
        :return: 线性变换后的输出张量。
        """
        if self.factorized:
            # 因子分解线性变换：
            # 1. 在输入张量的第4维（索引3）增加一个维度，使其变为 [..., *, in_features, 1]
            # 2. 使用 torch.matmul 执行批量矩阵乘法，结果形状为 [..., *, out_features, 1]
            #    (因为 weights 形状为 [..., in_features, out_features])
            # 3. 使用 squeeze(3) 去掉新增的第4维，得到 [..., *, out_features]
            # 4. 加上偏置 biases。
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        else:
            # 标准线性变换：matmul(input, weights) + biases
            return torch.matmul(input, weights) + biases


class Intra_Patch_Attention(nn.Module):
    """
    块内注意力机制 (Intra-Patch Attention)。
    该层计算单个 patch 内部不同时间步或元素之间的注意力。
    使用了特殊的权重生成和自定义线性层。
    """

    def __init__(self, d_model, factorized):
        """
        初始化块内注意力层。
        :param d_model (int): 模型的隐藏层维度。
        :param factorized (bool): 是否使用权重因子分解。
        """
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2  # 注意力头数
        # 检查隐藏层维度是否能被头数整除
        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')
        self.head_size = int(d_model // self.head)  # 每个头的维度
        # 使用自定义线性层处理 K, V 和最终输出
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        """
        执行块内注意力计算。
        :param query (Tensor): 查询张量，形状 [batch_size, seq_len_q, num_nodes, d_model]。
        :param key (Tensor): 键张量，形状 [batch_size, seq_len_k, num_nodes, d_model]。
        :param value (Tensor): 值张量，形状 [batch_size, seq_len_v, num_nodes, d_model]。
                             (通常 seq_len_k == seq_len_v)。
        :param weights_distinct (list of Tensors): 为 K 和 V 生成的特定权重。
        :param biases_distinct (list of Tensors): 为 K 和 V 生成的特定偏置。
        :param weights_shared (list of Tensors): 为最终输出生成的共享权重。
        :param biases_shared (list of Tensors): 为最终输出生成的共享偏置。
        :return: output (Tensor): 注意力输出，形状 [batch_size, seq_len_q, num_nodes, d_model]。
                 attention (Tensor): 注意力权重，形状 [batch_size * head, num_nodes, seq_len_q, seq_len_k]。
        """
        batch_size = query.shape[0] # 获取批次大小

        # 1. 使用特定权重和自定义线性层处理 Key 和 Value
        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])

        # 2. 多头分割和维度重排
        # 将 Q, K, V 按 head_size 分割成多个头，并拼接批次维度
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)  # [B*head, L_q, N, head_size]
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)      # [B*head, L_k, N, head_size]
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)  # [B*head, L_v, N, head_size]

        # 3. 维度重排以适应矩阵乘法
        query = query.permute((0, 2, 1, 3))  # [B*head, N, L_q, head_size]
        key = key.permute((0, 2, 3, 1))      # [B*head, N, head_size, L_k]
        value = value.permute((0, 2, 1, 3))  # [B*head, N, L_v, head_size]

        # 4. 计算注意力分数 (QK^T)
        attention = torch.matmul(query, key)  # [B*head, N, L_q, L_k]
        # 缩放
        attention /= (self.head_size ** 0.5)
        # Softmax 归一化
        attention = torch.softmax(attention, dim=-1)  # [B*head, N, L_q, L_k]

        # 5. 应用注意力权重到 Value (AV)
        x = torch.matmul(attention, value)  # [B*head, N, L_q, head_size]

        # 6. 恢复维度顺序
        x = x.permute((0, 2, 1, 3))  # [B*head, L_q, N, head_size]
        # 将多头结果拼接回原始维度
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)  # [B, L_q, N, d_model]

        # 7. 处理边界情况（如果 batch_size 为 0）
        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        # 8. 使用共享权重进行两次线性变换和激活
        # 第一次线性变换 + ReLU
        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        # 第二次线性变换
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])

        return x, attention


class Inter_Patch_Attention(nn.Module):
    """
    块间注意力机制 (Inter-Patch Attention)。
    该层计算不同 patch 之间的注意力。
    基于标准的多头自注意力机制实现。
    """

    def __init__(self, d_model, out_dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        """
        初始化块间注意力层。
        :param d_model (int): 输入特征维度。
        :param out_dim (int): 输出特征维度。
        :param n_heads (int): 注意力头数。
        :param d_k (int, optional): 每个头的 Key 维度。默认为 d_model // n_heads。
        :param d_v (int, optional): 每个头的 Value 维度。默认为 d_model // n_heads。
        :param res_attention (bool): 是否使用残差注意力 (来自前一层的注意力分数)。
        :param attn_dropout (float): 注意力权重的 Dropout 比例。
        :param proj_dropout (float): 输出投影层的 Dropout 比例。
        :param qkv_bias (bool): 线性层是否使用偏置。
        :param lsa (bool): 是否使用可学习的缩放因子 (Locality Self Attention)。
        """
        super().__init__()
        # 设置 Key 和 Value 的维度
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        # 定义用于生成 Q, K, V 的线性层
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # 残差注意力标志
        self.res_attention = res_attention
        # 实例化缩放点积注意力模块
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)
        # 输出投影层
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, out_dim), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        """
        执行块间注意力计算。
        :param Q (Tensor): 查询张量，形状 [batch_size * num_nodes, patch_num, d_model]。
        :param K (Tensor, optional): 键张量，形状同 Q。如果为 None，则等于 Q。
        :param V (Tensor, optional): 值张量，形状同 Q。如果为 None，则等于 Q。
        :param prev (Tensor, optional): 前一层的注意力分数 (用于残差注意力)。
        :param key_padding_mask (Tensor, optional): Key 的填充掩码，形状 [batch_size * num_nodes, patch_num]。
        :param attn_mask (Tensor, optional): 注意力掩码，形状 [patch_num, patch_num]。
        :return: output (Tensor): 注意力输出，形状 [batch_size * num_nodes, patch_num, out_dim]。
                 attn_weights (Tensor): 注意力权重，形状 [batch_size * num_nodes, n_heads, patch_num, patch_num]。
        """
        bs = Q.size(0) # 获取批次大小 (batch_size * num_nodes)
        # 如果 K 或 V 未提供，则默认使用 Q
        if K is None: K = Q
        if V is None: V = Q

        # 1. 线性变换并分割成多头
        # Q: [bs, patch_num, d_model] -> [bs, patch_num, n_heads * d_k] -> view -> [bs, patch_num, n_heads, d_k] -> transpose -> [bs, n_heads, patch_num, d_k]
        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        # K: [bs, patch_num, d_model] -> [bs, patch_num, n_heads * d_k] -> view -> [bs, patch_num, n_heads, d_k] -> permute -> [bs, n_heads, d_k, patch_num]
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3, 1)
        # V: [bs, patch_num, d_model] -> [bs, patch_num, n_heads * d_v] -> view -> [bs, patch_num, n_heads, d_v] -> transpose -> [bs, n_heads, patch_num, d_v]
        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1, 2)

        # 2. 应用缩放点积注意力
        if self.res_attention:
            # 如果使用残差注意力，传递 prev 参数
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # 3. 合并多头输出并进行投影
        # output: [bs, n_heads, patch_num, d_v] -> transpose -> [bs, patch_num, n_heads, d_v] -> contiguous -> view -> [bs, patch_num, n_heads * d_v]
        output = output.transpose(1, 2).contiguous().view(bs, Q.shape[1], self.n_heads * self.d_v)
        # 通过输出投影层
        output = self.to_out(output) # [bs, patch_num, out_dim]

        if self.res_attention:
            return output, attn_weights
        else:
            return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""
    缩放点积注意力模块 (Scaled Dot-Product Attention)。
    这是 Transformer 中的核心注意力计算单元 (Vaswani et al., 2017)。
    支持可选的残差注意力 (Realformer) 和局部自注意力 (Vision Transformer)。
    """

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        """
        初始化缩放点积注意力。
        :param d_model (int): 模型维度。
        :param n_heads (int): 注意力头数。
        :param attn_dropout (float): 注意力权重的 Dropout 比例。
        :param res_attention (bool): 是否使用残差注意力。
        :param lsa (bool): 是否使用可学习的缩放因子。
        """
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout) # 注意力 Dropout 层
        self.res_attention = res_attention          # 残差注意力标志
        head_dim = d_model // n_heads               # 每个头的维度
        # 可学习的缩放因子 (如果 lsa=True)
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa # 可学习缩放标志

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        """
        执行缩放点积注意力计算。
        :param q (Tensor): 查询，形状 [batch_size, n_heads, q_len, d_k]。
        :param k (Tensor): 键，形状 [batch_size, n_heads, d_k, kv_len] (注意维度顺序)。
        :param v (Tensor): 值，形状 [batch_size, n_heads, kv_len, d_v]。
        :param prev (Tensor, optional): 前一层的注意力分数 (用于残差注意力)。
        :param key_padding_mask (Tensor, optional): Key 填充掩码，形状 [batch_size, kv_len]。
        :param attn_mask (Tensor, optional): 注意力掩码，形状 [q_len, kv_len] 或 [batch_size, n_heads, q_len, kv_len]。
        :return: output (Tensor): 注意力输出，形状 [batch_size, n_heads, q_len, d_v]。
                 attn_weights (Tensor): 注意力权重，形状 [batch_size, n_heads, q_len, kv_len]。
                 attn_scores (Tensor, optional): 未归一化的注意力分数 (如果 res_attention=True)。
        """
        # 1. 计算 QK^T 并缩放
        # q: [bs, n_heads, q_len, d_k], k: [bs, n_heads, d_k, kv_len]
        # matmul 结果: [bs, n_heads, q_len, kv_len]
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x q_len x kv_len]

        # 2. 添加前一层的注意力分数（可选，用于残差注意力）
        if prev is not None: attn_scores = attn_scores + prev

        # 3. 应用注意力掩码（可选）
        # attn_mask 通常用于防止未来信息泄露（因果掩码）或屏蔽某些位置
        if attn_mask is not None:  # attn_mask with shape [q_len x kv_len] or [bs x n_heads x q_len x kv_len]
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf) # 将掩码为 True 的位置设为 -inf
            else:
                attn_scores += attn_mask # 直接加到分数上

        # 4. 应用 Key 填充掩码（可选）
        # key_padding_mask 用于屏蔽序列中的填充部分
        if key_padding_mask is not None:  # mask with shape [bs x kv_len]
            # unsqueeze(1).unsqueeze(2) 将其扩展为 [bs, 1, 1, kv_len] 以广播到 [bs, n_heads, q_len, kv_len]
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # 5. 对注意力分数进行 Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights : [bs x n_heads x q_len x kv_len]
        # 应用注意力 Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 使用注意力权重对 Value 进行加权求和
        # attn_weights: [bs, n_heads, q_len, kv_len], v: [bs, n_heads, kv_len, d_v]
        # matmul 结果: [bs, n_heads, q_len, d_v]
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class WeightGenerator(nn.Module):
    """
    权重生成器。
    根据是否启用因子分解 (factorized)，生成用于 CustomLinear 的权重和偏置。
    因子分解版本通过一个小的记忆模块和生成网络来生成多个权重矩阵，可能用于参数共享或减少参数量。
    """

    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        """
        初始化权重生成器。
        :param in_dim (int): 输入维度。
        :param out_dim (int): 输出维度。
        :param mem_dim (int or None): 记忆模块的维度。如果 factorized=False，则此参数被忽略。
        :param num_nodes (int): 节点/变量数量。
        :param factorized (bool): 是否使用权重因子分解机制。
        :param number_of_weights (int): 要生成的权重矩阵/偏置向量的数量。
        """
        super(WeightGenerator, self).__init__()
        # print('FACTORIZED {}'.format(factorized)) # 调试打印
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim

        if self.factorized:
            # --- 因子分解模式 ---
            # 为每个节点创建一个记忆向量
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True) # .to('cpu') # 原注释
            # 生成网络：将记忆向量映射到更高维度的中间表示
            self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100) # 输出维度硬编码为100
            ])
            # 更新 mem_dim 为生成网络输出维度的一部分
            self.mem_dim = 10

            # 定义因子分解所需的参数矩阵 P, Q, B
            # P: [in_dim, mem_dim]
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            # Q: [mem_dim, out_dim]
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            # B: [mem_dim^2, out_dim] (用于偏置生成)
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            # --- 标准模式 ---
            # 直接定义权重 P 和偏置 B
            # P: [in_dim, out_dim]
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            # B: [1, out_dim]
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True) for _ in range(number_of_weights)])

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化模型参数。
        """
        # 根据是否因子分解选择要初始化的参数列表
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        # 对权重矩阵使用 Kaiming 均匀初始化
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        # 对标准模式下的偏置向量进行均匀初始化
        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        """
        生成权重和偏置。
        :return: (weights, biases)
                 - weights: 权重列表。
                 - biases: 偏置列表。
        """
        if self.factorized:
            # --- 因子分解模式的前向传播 ---
            # 1. 通过生成网络处理记忆向量
            # memory: [num_nodes, mem_dim] -> unsqueeze(1) -> [num_nodes, 1, mem_dim]
            # generator 输出: [num_nodes, 1, 100]
            memory = self.generator(self.memory.unsqueeze(1))

            # 2. 生成偏置：matmul(memory, B[i])
            # memory: [num_nodes, 1, 100], B[i]: [100, out_dim]
            # 结果: [num_nodes, 1, out_dim] -> squeeze(1) -> [num_nodes, out_dim]
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]

            # 3. 生成权重：P[i] @ memory @ Q[i]
            # memory 需要 reshape 成方阵: [num_nodes, mem_dim, mem_dim] (10x10)
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            # P[i]: [in_dim, mem_dim], memory: [num_nodes, mem_dim, mem_dim], Q[i]: [mem_dim, out_dim]
            # torch.matmul(P[i], memory): [num_nodes, in_dim, mem_dim]
            # torch.matmul(..., Q[i]): [num_nodes, in_dim, out_dim]
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]

            return weights, bias
        else:
            # --- 标准模式的前向传播 ---
            # 直接返回参数列表
            return self.P, self.B


class Transpose(nn.Module):
    """
    简单的张量转置模块。
    方便在 nn.Sequential 中使用转置操作。
    """

    def __init__(self, *dims, contiguous=False):
        """
        初始化转置模块。
        :param dims: 要交换的维度元组，例如 (1, 2) 表示交换第1和第2维。
        :param contiguous: 是否在转置后调用 .contiguous() 以确保内存连续。
        """
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        """
        执行转置操作。
        :param x: 输入张量。
        :return: 转置后的张量。
        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)
import sys
from pathlib import Path

# 获取当前脚本文件的绝对路径
FILE = Path(__file__).resolve()
# 获取当前文件所在目录的父目录的父目录 (通常是项目的根目录)
ROOT = FILE.parents[2]
# 将项目根目录添加到 sys.path，以便 Python解释器可以找到项目中的模块
sys.path.append(ROOT.as_posix())

from torch import Tensor
from typing import List, Union, Optional  # 从 typing 模块导入类型提示工具

import torch
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch神经网络函数


class Projection(nn.Module):
    """
    投影层模块。
    这个模块的作用是将输入张量投影到一个指定维度的空间。
    通常用于匹配不同层或不同模态特征的维度。
    """

    def __init__(self,
                 inp_dim: int = 512,  # 输入特征的维度，默认为 512
                 d_model: int = 512,  # 输出特征的维度 (即模型的隐藏层维度)，默认为 512
                 ):
        super(Projection, self).__init__()
        # 定义一个线性变换层 (全连接层)
        # inp_dim: 输入特征数, d_model: 输出特征数
        self.proj = nn.Linear(inp_dim, d_model)

    def forward(self, inp: Tensor) -> Tensor:
        """
        前向传播函数。
        Args:
            inp (Tensor): 输入张量，形状可以是 (batch_size, ..., inp_dim)。
        Returns:
            Tensor: 经过线性投影后的张量，形状为 (batch_size, ..., d_model)。
        """
        return self.proj(inp)


class AVCrossAttn(nn.Module):
    """
    音视频交叉注意力模块 (Audio-Visual Cross-Attention)。
    该模块实现了音频和视频特征之间的相互关注，允许它们交换和融合信息。
    它包含自注意力机制和交叉注意力机制。
    """

    def __init__(self,
                 d_model: int = 512,  # 模型的隐藏层维度，默认为 512
                 nhead: int = 8,  # 多头注意力机制中的头数，默认为 8
                 dropout: float = 0.1  # Dropout 的概率，默认为 0.1
                 ):
        super(AVCrossAttn, self).__init__()

        # 交叉注意力层：一个模态的特征作为查询(query)，另一个模态的特征作为键(key)和值(value)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 自注意力层：模态内部的特征进行自关注
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络 (Feed-Forward Network) 的第一个线性层
        self.linear1 = nn.Linear(d_model, d_model)
        # 前馈网络的第二个线性层
        self.linear2 = nn.Linear(d_model, d_model)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 第一个层归一化 (Layer Normalization)
        self.norm1 = nn.LayerNorm(d_model)
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 注意力掩码 (在此模块中未使用，设为 None)
        self.attn_mask = None

        # 初始化前馈网络中线性层的权重和偏置
        # 使用 Kaiming 正态分布初始化权重，有助于缓解梯度消失/爆炸问题
        nn.init.kaiming_normal_(self.linear1.weight)
        # 将偏置初始化为 0
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def sub_forward(self,
                    src_q: Tensor,  # 作为查询 (query) 的源特征张量
                    src_v: Tensor,  # 作为键 (key) 和值 (value) 的源特征张量
                    query: Optional[Tensor] = None,  # 可选的查询张量，在此函数中未使用
                    ) -> Tensor:
        """
        子前向传播函数，执行单向的注意力操作 (例如，音频关注视频，或视频关注音频)。
        这是 Transformer Encoder Layer 的典型结构。
        Args:
            src_q (Tensor): 查询序列，形状 (Batch, Seq_q, Dim)。
            src_v (Tensor): 键/值序列，形状 (Batch, Seq_v, Dim)。
            query (Optional[Tensor]): 此参数在此处未使用。
        Returns:
            Tensor: 处理后的查询序列，形状 (Batch, Seq_q, Dim)。
        """
        # PyTorch 的 MultiheadAttention 期望输入的形状是 (Seq_len, Batch, Dim)。
        # 因此，需要将输入的 (Batch, Seq_len, Dim) 转换为 (Seq_len, Batch, Dim)。
        src_q = src_q.permute(1, 0, 2)  # (Seq_q, Batch, Dim)
        src_v = src_v.permute(1, 0, 2)  # (Seq_v, Batch, Dim)

        # 1. 多头注意力部分 (自注意力和交叉注意力)
        # 对 src_q 进行自注意力 (src_q 作为 query, key, value)
        slf_attn_out = self.slf_attn(src_q, src_q, src_q)[0]  # [0] 取的是注意力输出，忽略权重
        # 对 src_q 和 src_v 进行交叉注意力 (src_q 作为 query, src_v 作为 key 和 value)
        crs_attn_out = self.crs_attn(src_q, src_v, src_v)[0]
        # 残差连接：原始 src_q 加上 dropout 后的自注意力和交叉注意力的输出
        src_q = src_q + \
                self.dropout(slf_attn_out) + \
                self.dropout(crs_attn_out)
        # 第一个层归一化
        src_q = self.norm1(src_q)

        # 2. 前馈网络 (Feed-Forward Network) 部分
        # FFN: Linear -> ReLU -> Dropout -> Linear
        ffn_output = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        # 残差连接：加上 dropout 后的 FFN 输出
        src_q = src_q + self.dropout(ffn_output)
        # 第二个层归一化
        src_q = self.norm2(src_q)
        # 将维度转换回 (Batch, Seq_q, Dim)
        return src_q.permute(1, 0, 2)

    def forward(self,
                src_q: Tensor,  # 第一个模态的特征，例如音频 (Batch, Seq_audio, Dim)
                src_v: Tensor,  # 第二个模态的特征，例如视频 (Batch, Seq_video, Dim)
                query: Optional[Tensor] = None,  # 可选查询张量，在此函数中未使用
                visualize: bool = False  # 是否返回可视化信息 (例如注意力权重)，此处默认为 False
                ) -> List[Tensor]:
        """
        完整的前向传播函数。
        它会进行双向的交叉注意力：
        - src_q (如音频) 关注 src_v (如视频) 并更新 src_q。
        - src_v (如视频) 关注 src_q (如音频) 并更新 src_v。
        Args:
            src_q (Tensor): 第一个模态的特征。
            src_v (Tensor): 第二个模态的特征。
            query (Optional[Tensor]): 未使用。
            visualize (bool): 若为 True，则返回额外的可视化信息 (当前版本返回 None)。
        Returns:
            List[Tensor]: 包含两个张量，分别是更新后的 src_q 和 src_v。
                          如果 visualize 为 True，则列表第三个元素为 None。
        """
        # src_q (查询) 与 src_v (键/值) 交互，得到更新后的 src1 (即更新后的 src_q)
        src1 = self.sub_forward(src_q, src_v)
        # src_v (查询) 与 src_q (键/值) 交互，得到更新后的 src2 (即更新后的 src_v)
        src2 = self.sub_forward(src_v, src_q)

        if visualize:
            # 如果需要可视化，返回更新后的两个模态特征以及一个 None (占位符，可扩展为注意力权重)
            return src1, src2, None
        # 默认只返回更新后的两个模态特征
        return src1, src2


class AVQCrossAttn(nn.Module):
    """
    音视频问句交叉注意力模块 (Audio-Visual-Question Cross-Attention)。
    此模块在音视频交叉注意力的基础上，额外引入了问句 (Question) 特征，
    使得音视频特征的交互也受到问句的引导。
    """

    def __init__(self,
                 d_model: int = 512,  # 模型维度
                 nhead: int = 8,  # 多头注意力头数
                 dropout: float = 0.1  # Dropout 比率
                 ):
        super(AVQCrossAttn, self).__init__()

        # 问句注意力层：模态特征作为 query，问句特征作为 key 和 value
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 交叉注意力层 (音视频之间)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 自注意力层 (模态内部)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 初始化权重
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def sub_forward(self,
                    src_q: Tensor,  # 作为 query 的源模态特征 (例如音频)
                    src_v: Tensor,  # 作为 key/value 的源模态特征 (例如视频)
                    query: Tensor,  # 问句特征
                    visualize: bool = False  # 是否返回注意力权重 (此参数在此处不直接影响逻辑，但外部调用时可能有关)
                    ) -> Tensor:
        """
        子前向传播过程，处理一个方向的注意力，并融合问句信息。
        Args:
            src_q (Tensor): 查询序列 (Batch, Seq_q, Dim)。
            src_v (Tensor): 键/值序列 (Batch, Seq_v, Dim)。
            query (Tensor): 问句序列 (Batch, Seq_query, Dim)。
            visualize (bool): 未在此函数中直接使用其布尔值来改变行为。
        Returns:
            Tuple[Tensor, Tensor]: 处理后的查询序列 (Batch, Seq_q, Dim) 和问句注意力权重。
        """
        # 维度置换 (Batch, Seq, Dim) -> (Seq, Batch, Dim) 以适应 MultiheadAttention
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        query = query.permute(1, 0, 2)  # 问句特征也进行置换

        # 1. 多种注意力融合
        # 问句注意力：src_q 对 query (问句) 进行注意力 (src_q 作为 query, query 作为 key 和 value)
        qst_attn_out, weight = self.qst_attn(src_q, query, query)  # weight 是问句注意力权重
        # 自注意力：src_q 对自身进行自注意力
        slf_attn_out = self.slf_attn(src_q, src_q, src_q)[0]
        # 交叉注意力：src_q 对 src_v 进行交叉注意力
        crs_attn_out = self.crs_attn(src_q, src_v, src_v)[0]
        # 残差连接：原始 src_q 加上三种注意力（自、交叉、问句）的 dropout 输出
        src_q = src_q + \
                self.dropout(slf_attn_out) + \
                self.dropout(crs_attn_out) + \
                self.dropout(qst_attn_out)
        # 第一个层归一化
        src_q = self.norm1(src_q)

        # 2. 前馈网络
        ffn_out = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        # 残差连接
        src_q = src_q + self.dropout(ffn_out)
        # 第二个层归一化
        src_q = self.norm2(src_q)
        # 维度置换回 (Batch, Seq_q, Dim)
        return src_q.permute(1, 0, 2), weight  # 返回更新后的 src_q 和问句注意力权重

    def forward(self,
                src_q: Tensor,  # 第一个模态特征 (例如音频)
                src_v: Tensor,  # 第二个模态特征 (例如视频)
                query: Tensor,  # 问句特征
                visualize: bool = False  # 是否返回注意力权重
                ) -> List[Tensor]:
        """
        完整的前向传播。双向计算注意力，并融合问句信息。
        Args:
            src_q (Tensor): 第一个模态特征 (Batch, Seq1, Dim)。
            src_v (Tensor): 第二个模态特征 (Batch, Seq2, Dim)。
            query (Tensor): 问句特征 (Batch, Seq_query, Dim)。
            visualize (bool): 如果为 True，则返回注意力权重。
        Returns:
            List[Tensor]: 包含两个更新后的模态特征。如果 visualize 为 True，
                          则第三个元素是一个包含两个注意力权重张量的列表 [a_weight, v_weight]。
        """
        # 计算 src_q 在问句和 src_v 指导下的更新，以及对应的问句注意力权重 a_weight
        src1, a_weight = self.sub_forward(src_q, src_v, query, visualize)
        # 计算 src_v 在问句和 src_q 指导下的更新，以及对应的问句注意力权重 v_weight
        src2, v_weight = self.sub_forward(src_v, src_q, query, visualize)

        if visualize:
            # 返回更新后的特征和两个方向的问句注意力权重
            return src1, src2, [a_weight, v_weight]
        # 否则只返回更新后的特征
        return src1, src2


class QstGrounding(nn.Module):
    """
    问句定位 (Question Grounding) 模块。
    该模块使用问句特征来关注 (attend to) 音频/视频数据中的相关部分，
    并生成一个融合了问句信息的特征表示。
    """

    def __init__(self,
                 d_model: int = 512,  # 模型维度
                 nhead: int = 8,  # 多头注意力头数
                 dropout: float = 0.1):  # Dropout 比率
        super(QstGrounding, self).__init__()

        self.act = nn.ReLU()  # ReLU 激活函数
        self.norm = nn.LayerNorm(d_model)  # 层归一化
        # 多头注意力层：问句作为 query，融合后的音视频数据作为 key 和 value
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        # MLP (多层感知机)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 第一个线性层，维度减半
            nn.ReLU(),  # ReLU 激活
            nn.Linear(d_model // 2, d_model)  # 第二个线性层，维度恢复
        )
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        # 初始化 MLP 的权重
        self.mlp.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """自定义权重初始化函数"""
        if isinstance(m, nn.Linear):
            # 对线性层的权重使用 Kaiming 正态初始化
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                # 将偏置初始化为 0
                nn.init.constant_(m.bias, 0)

    def forward(self,
                qst: Tensor,  # 问句特征, 形状 (Batch, Dim) 或 (Batch, 1, Dim)
                data: Union[Tensor, List[Tensor]],  # 音视频数据。
                # 可以是单个张量 (Batch, Seq, Dim)
                # 或包含多个此类张量的列表 (例如 [音频特征, 视频特征])
                ) -> Tensor:
        """
        前向传播。
        Args:
            qst (Tensor): 问句特征。
            data (Union[Tensor, List[Tensor]]): 一个或多个模态的数据。
        Returns:
            Tensor: 经过问句定位和融合后的特征, 形状 (Batch, Dim)。
        """
        if isinstance(data, list):
            # 如果 data 是一个列表 (例如，包含音频和视频特征)
            # 1. 将每个元素的维度从 (Batch, Seq, Dim) 转换为 (Seq, Batch, Dim)
            data = [d.permute(1, 0, 2) for d in data]
            # 2. 沿着序列维度 (dim=0) 拼接这些特征，形成一个统一的序列
            data = torch.cat(data, dim=0)  # 形状 (Seq_total, Batch, Dim)
        else:
            # 如果 data 是单个张量，同样进行维度转换
            data = data.permute(1, 0, 2)  # 形状 (Seq, Batch, Dim)

        # 问句特征 qst 通常是 (Batch, Dim) 或 (Batch, 1, Dim)。
        # MultiheadAttention 的 query 输入期望是 (Query_Seq_len, Batch, Dim)。
        # 如果 qst 是 (Batch, Dim)，则 unsqueeze(0) 变为 (1, Batch, Dim)。
        # 如果 qst 是 (Batch, 1, Dim)，则 permute(1,0,2) 变为 (1, Batch, Dim)。
        # 这里假设 qst 是 (Batch, Dim) 或已经处理成类似全局特征的形式。
        qst = qst.unsqueeze(0)  # 形状 (1, Batch, Dim)

        # 注意力计算：qst 作为 query，data (融合后的音视频序列) 作为 key 和 value。
        # attn_output 形状为 (1, Batch, Dim)，squeeze(0) 后为 (Batch, Dim)。
        # 这是问句关注音视频数据后得到的上下文向量。
        attn_output = self.attn(qst, data, data)[0].squeeze(0)

        # 特征融合：
        # 1. data.mean(dim=0): 计算 data 在序列维度上的平均池化，得到 (Batch, Dim)，作为全局上下文。
        # 2. self.mlp(attn_output): 将问句关注的上下文向量通过 MLP 进行非线性变换。
        # 3. 两者相加，再通过 dropout。
        feat = data.mean(dim=0) + self.dropout(self.mlp(attn_output))
        # 层归一化
        feat = self.norm(feat)
        return feat


class TempMoE(nn.Module):
    """
    时间混合专家 (Temporal Mixture of Experts) 模块。
    该模块利用问句信息动态地选择和组合多个“专家”网络（通常是MLP）的输出来处理时间序列数据。
    它还使用高斯函数来对时间维度进行加权，实现时间上的软选择和聚焦。
    """

    def __init__(self,
                 d_model: int = 512,  # 模型维度
                 nhead: int = 8,  # 多头注意力头数 (用于问句注意力)
                 topK: int = 5,  # 选择 top-K 个专家进行组合
                 n_experts: int = 10,  # 专家网络总数
                 sigma: int = 9,  # 高斯函数宽度调整参数 (影响高斯权重的集中程度)
                 dropout: float = 0.1,  # Dropout 比率
                 vis_branch: bool = False,  # 指示是否为视觉特定分支 (例如，若为True，可能使用不同的归一化层)
                 ):
        super(TempMoE, self).__init__()

        self.sigma = sigma
        self.topK = topK
        self.n_experts = n_experts

        if vis_branch:
            # 如果是为视觉分支设计的 (或需要区分音视频处理)，使用独立的归一化层
            self.anorm = nn.LayerNorm(d_model)  # 音频输出归一化
            self.vnorm = nn.LayerNorm(d_model)  # 视频输出归一化
        else:
            # 否则，使用单个归一化层
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)  # Dropout 层
        # 问句注意力层：用于根据问句生成时间相关的权重/特征 (temp_w)
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        # 高斯参数预测器：一个线性层，输出每个专家的中心 (center_offset) 和宽度 (width_scale) 参数
        # 每个专家需要2个参数，所以输出维度是 2 * n_experts
        self.gauss_pred = nn.Sequential(
            nn.Linear(d_model, 2 * n_experts)
        )
        # 路由网络 (Router)：一个线性层，用于预测每个专家的选择概率 (gating weights)
        self.router = nn.Sequential(
            nn.Linear(d_model, n_experts)
        )
        # 专家网络列表：每个专家是一个简单的 MLP (Linear -> ReLU -> Linear)
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(d_model, int(d_model // 2)),
                nn.ReLU(),
                nn.Linear(int(d_model // 2), d_model)
            ])
            for _ in range(n_experts)
        ])
        # 初始化专家网络的权重
        self.experts.apply(self._init_weights)

        # 边距 (margin)，用于确保高斯中心的基础位置在调整后不会过于重叠或超出[0,1]范围
        self.margin = (1 / (n_experts * 2))
        # 预定义的高斯中心的基础位置，均匀分布在 [margin, 1-margin] 之间
        self.center = torch.linspace(self.margin, 1 - self.margin, self.n_experts)
        self.center.requires_grad_(False)  # 中心基础位置是固定的，不需要梯度
        # 初始化路由网络和高斯参数预测器的权重
        self.router.apply(self._init_weights)
        self.gauss_pred.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """自定义权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_gaussian(self,
                          pred: torch.Tensor,  # 预测的高斯参数 (Batch, N_experts, 2)，2 代表 (center_offset, width_scale)
                          topk_inds: torch.Tensor,  # top-K 专家的索引 (Batch, TopK)
                          T: int = 60  # 时间序列的长度
                          ) -> Tensor:
        """
        为选中的 top-K 专家生成高斯时间权重。
        参考实现: https://github.com/minghangz/cpl
        Args:
            pred (torch.Tensor): 所有专家预测的高斯参数 (中心偏移和宽度缩放)。
            topk_inds (torch.Tensor): 被选中的 top-K 专家的索引。
            T (int): 时间序列的长度。
        Returns:
            Tensor: 高斯时间权重 (Batch, TopK, T)。每个选中的专家对应一行时间权重。
        """
        weights = []
        # 基础中心位置 (1, N_experts)，复制为 (Batch, N_experts) 并移动到设备
        base_centers = self.center.unsqueeze(0).repeat(pred.size(0), 1).to(pred.device)
        # 实际中心 = 基础中心 + 预测的中心偏移 (pred[:, :, 0])
        adjusted_centers = base_centers + pred[:, :, 0]
        # 根据 topk_inds 选择对应专家的调整后中心
        selected_centers = torch.gather(adjusted_centers, 1, topk_inds)  # (Batch, TopK)
        # 根据 topk_inds 选择对应专家的宽度缩放因子 (pred[:, :, 1])
        selected_widths_scale = torch.gather(pred[:, :, 1], 1, topk_inds)  # (Batch, TopK)

        for i in range(self.topK):  # 对每个选中的专家
            center_i = selected_centers[:, i]  # 当前专家的中心 (Batch,)
            width_scale_i = selected_widths_scale[:, i]  # 当前专家的宽度缩放因子 (Batch,)

            # 生成时间轴上的点，范围 [0, 1]，长度为 T
            t_axis = torch.linspace(0, 1, T)
            t_axis = t_axis.view(1, -1).expand(center_i.size(0), -1).to(center_i.device)  # (Batch, T)

            # 限制中心在 [0, 1] 范围内
            center_i_clamped = torch.clamp(center_i.unsqueeze(-1), min=0, max=1)  # (Batch, 1)
            # 计算实际宽度：限制宽度缩放因子大于等于0.09，然后除以 sigma 进行调整
            # width_scale 越大，高斯分布越宽；sigma 越大，高斯分布越窄
            actual_width_i = torch.clamp(width_scale_i.unsqueeze(-1), min=0.09) / self.sigma  # (Batch, 1)

            # 计算高斯权重: N(t | mu, sigma^2) = (1 / (sigma * sqrt(2*pi))) * exp(-(t-mu)^2 / (2*sigma^2))
            # 这里的 0.3989... 是 1/sqrt(2*pi)
            # w_val 是高斯分布的归一化常数的一部分
            w_val = 0.3989422804014327
            # 计算未归一化的高斯权重
            gaussian_w = w_val / actual_width_i * torch.exp(
                -(t_axis - center_i_clamped) ** 2 / (2 * actual_width_i ** 2))  # (Batch, T)
            # 对每个batch内的时间权重进行归一化，使其最大值为1 (峰值归一化)
            weights.append(
                gaussian_w / gaussian_w.max(dim=-1, keepdim=True)[0]
            )
        # 将所有 topK 专家的权重堆叠起来
        return torch.stack(weights, dim=1)  # (Batch, TopK, T)

    def get_output(self,
                   experts_logits: Tensor,  # 所有专家的输出 (T, Batch, N_experts, C)
                   gauss_weight: Tensor,  # 高斯时间权重 (Batch, TopK, T)
                   topk_inds: Tensor,  # top-K 专家的索引 (Batch, TopK)
                   topk_probs: Tensor,  # top-K 专家的选择概率 (Batch, TopK) (gating_weights)
                   shape: tuple,  # 原始数据形状 (B, T, C)
                   ) -> Tensor:
        """
        根据高斯时间权重和专家选择概率，聚合专家输出。
        Args:
            experts_logits (Tensor): 所有专家的输出。
            gauss_weight (Tensor): 高斯时间权重。
            topk_inds (Tensor): Top-K 专家索引。
            topk_probs (Tensor): Top-K 专家概率 (门控权重)。
            shape (tuple): 原始输入数据的形状 (B, T, C)，用于获取 B, T, C。
        Returns:
            Tensor: 聚合后的输出 (B, 1, C)，代表一个加权融合的全局特征。
        """
        B, T, C = shape  # Batch_size, Time_steps, Channels

        # experts_logits: (T, B, N_experts, C) -> (B*T, N_experts, C)
        # 先 permute 再 reshape，是为了后续 gather 操作方便
        experts_logits_reshaped = experts_logits.permute(1, 0, 2, 3).reshape(B * T, self.n_experts, C)

        # topk_inds: (B, TopK)
        # 扩展 topk_inds 以匹配 experts_logits_reshaped 的维度，用于 gather
        # 1. topk_inds.repeat(T, 1): (B*T, TopK) - T个时间步共享相同的专家选择
        # 2. .unsqueeze(-1): (B*T, TopK, 1)
        # 3. .repeat(1, 1, C): (B*T, TopK, C) - 在特征维度上复制索引
        topk_inds_expanded = topk_inds.repeat(T, 1).unsqueeze(-1).repeat(1, 1, C)

        # 从所有专家输出中，根据 topk_inds 选择 top-K 个专家的输出
        # torch.gather(input, dim, index)
        # selected_experts_logits: (B*T, TopK, C)
        selected_experts_logits = torch.gather(
            experts_logits_reshaped, 1,  # 沿着第1维 (N_experts 维度) gather
            topk_inds_expanded
        )

        # (B*T, TopK, C) -> (B, T, TopK, C)
        selected_experts_logits = selected_experts_logits.reshape(B, T, self.topK, C).contiguous()

        # 对每个选中的专家，用其对应的高斯时间权重对专家在时间维度上的输出进行加权平均
        output_per_expert_list = []
        for i in range(self.topK):  # 遍历 topK 个选中的专家
            # gauss_weight[:, i, :]: (B, T) -> (B, 1, T) (当前专家的时间权重)
            # selected_experts_logits[:, :, i, :]: (B, T, C) (当前专家的输出)
            # 矩阵乘法 (B, 1, T) @ (B, T, C) -> (B, 1, C)
            # 得到每个专家经过时间加权后的输出
            weighted_expert_output = gauss_weight[:, i, :].unsqueeze(1) @ selected_experts_logits[:, :, i, :]
            output_per_expert_list.append(weighted_expert_output)

        # output_per_expert_list 是一个包含 TopK 个张量的列表, 每个张量形状 (B, 1, C)
        # 沿着新的维度 (dim=1) 拼接它们 -> (B, TopK, C)
        output_all_selected_experts = torch.cat(output_per_expert_list, dim=1)

        # 用路由器的 top-K 概率 (门控权重) 对加权后的专家输出进行最终的加权平均
        # topk_probs: (B, TopK) -> (B, 1, TopK)
        # output_all_selected_experts: (B, TopK, C)
        # 矩阵乘法 (B, 1, TopK) @ (B, TopK, C) -> (B, 1, C)
        final_output = topk_probs.unsqueeze(1) @ output_all_selected_experts
        return final_output

    def forward(self,
                qst: Tensor,  # 问句特征, 形状 (B, D)
                data: Tensor,  # 主模态数据 (例如全局音频或视频特征), 形状 (B, T, D)
                sub_data: Optional[Tensor] = None,  # 可选的子模态数据 (例如音频和视频的patch特征)。
                # 如果提供, 它是 [[B, T, D], [B, T, D]] 形式的列表。
                ) -> Union[Tensor, List[Tensor]]:
        """
        前向传播。
        Args:
            qst (Tensor): 问句特征。
            data (Tensor): 主数据流。
            sub_data (Optional[Tensor]): 可选的辅助数据流。
        Returns:
            Union[Tensor, List[Tensor]]: 如果 sub_data 为 None，返回单个聚合特征 (B, 1, C)。
                                         如果 sub_data 不为 None，返回两个聚合特征的列表 [a_out, v_out]，
                                         分别对应 sub_data 中的两个元素经过MoE处理后的结果。
        """
        B, T, C = data.size()  # 获取 Batch_size, Time_steps, Channels
        # 将主数据 data 的维度从 (B, T, C) 转换为 (T, B, C) 以适应 MultiheadAttention
        # 注意：变量 'data' 被重新赋值为其 permute 后的版本
        data = data.permute(1, 0, 2)

        # 将问句特征 qst 从 (B, D) unsqueeze到 (1, B, D) 作为 MultiheadAttention 的 query
        qst = qst.unsqueeze(0)
        # 问句对主数据 (permuted_data) 进行注意力，获取时间上的重要性特征/权重 (temp_w)
        # temp_w 形状: (1, B, C) -> squeeze(0) -> (B, C)
        temp_w = self.qst_attn(qst, data, data)[0].squeeze(0)

        # 路由网络：根据 temp_w 决定选择哪些专家
        router_logits = self.router(temp_w)  # (B, N_experts)，每个专家的原始打分
        router_probs = F.softmax(router_logits, dim=-1)  # (B, N_experts)，转换为概率分布
        # 选择概率最高的 top-K 个专家及其概率和索引
        topk_probs, topk_inds = torch.topk(router_probs, self.topK, dim=-1)  # 形状都是 (B, TopK)
        # 归一化 top-K 概率，使其和为1 (作为最终的门控权重)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # (B, TopK)

        # 高斯参数预测：根据 temp_w 预测高斯分布的中心偏移和宽度缩放因子
        gauss_cw = self.gauss_pred(temp_w)  # (B, 2*N_experts)
        gauss_cw = gauss_cw.view(B, self.n_experts, 2)  # (B, N_experts, 2)
        # 中心偏移量使用 tanh 激活并乘以 margin，使其值域在 [-margin, margin]
        gauss_cw[:, :, 0] = torch.tanh(gauss_cw[:, :, 0]) * self.margin
        # 宽度缩放因子使用 sigmoid 激活，使其值域在 (0, 1)
        gauss_cw[:, :, 1] = torch.sigmoid(gauss_cw[:, :, 1])
        # 根据预测的参数和选中的 topK 专家，生成高斯时间权重
        gauss_weight = self.generate_gaussian(gauss_cw, topk_inds=topk_inds, T=T)  # (B, TopK, T)

        if sub_data is not None:
            # 如果有子数据 (例如，音频和视频的 patch 特征)
            # 处理第一个子数据 (假设为音频 patch)
            a_sub_data_permuted = sub_data[0].permute(1, 0, 2)  # (T, B, C)
            # 将子数据与主数据 (已permute) 相加 (元素级，类似残差连接)
            a_combined_data = data + a_sub_data_permuted  # (T, B, C)
            # 所有专家处理组合后的音频数据
            # a_experts_outputs_stack: (T, B, N_experts, C)
            a_experts_outputs_stack = torch.stack([exprt(a_combined_data) for exprt in self.experts], dim=2)
            # 聚合专家输出，得到音频分支的最终输出
            a_outs = self.get_output(a_experts_outputs_stack, gauss_weight, topk_inds, topk_probs,
                                     (B, T, C))  # (B, 1, C)

            # 处理第二个子数据 (假设为视频 patch)
            v_sub_data_permuted = sub_data[1].permute(1, 0, 2)  # (T, B, C)
            v_combined_data = data + v_sub_data_permuted  # (T, B, C)
            v_experts_outputs_stack = torch.stack([exprt(v_combined_data) for exprt in self.experts], dim=2)
            v_outs = self.get_output(v_experts_outputs_stack, gauss_weight, topk_inds, topk_probs,
                                     (B, T, C))  # (B, 1, C)

            # 对音频和视频分支的输出分别进行层归一化 (如果 vis_branch=True，则使用 self.anorm 和 self.vnorm)
            return self.anorm(a_outs), self.vnorm(v_outs)
        else:
            # 如果没有子数据，只处理主数据 (已permute的data)
            # main_experts_outputs_stack: (T, B, N_experts, C)
            main_experts_outputs_stack = torch.stack([exprt(data) for exprt in self.experts], dim=2)
            # 聚合专家输出
            main_outs = self.get_output(main_experts_outputs_stack, gauss_weight, topk_inds, topk_probs,
                                        (B, T, C))  # (B, 1, C)
            # 层归一化 (如果 vis_branch=False，则使用 self.norm)
            return self.norm(main_outs)


class PatchSelecter(nn.Module):
    """
    Patch 选择器模块。
    该模块用于根据全局的音频和视频特征，来选择或调整局部的 patch 特征。
    它首先对 patch 特征进行自注意力，然后让全局音视频特征与 (自注意力后的) patch 特征进行交叉注意力，
    最后通过一个 MLP 得到调整后的、分别与音频和视频相关的 patch 级特征贡献。
    """

    def __init__(self,
                 d_model: int = 512,  # 模型维度
                 nhead: int = 8,  # 多头注意力头数
                 dropout: float = 0.1):  # Dropout 比率
        super(PatchSelecter, self).__init__()

        self.d_model = d_model  # 模型维度
        self.nhead = nhead  # 多头注意力头数
        self.dropout = dropout  # Dropout 比率 (浮点数值), 会被下面的 nn.Dropout 模块覆盖并使用

        self.vnorm = nn.LayerNorm(d_model)  # 视频相关特征的层归一化
        self.anorm = nn.LayerNorm(d_model)  # 音频相关特征的层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout 层 (nn.Module)，这将是实际使用的 self.dropout
        # Patch 内部的自注意力层
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 全局音视频特征与 Patch 特征之间的交叉注意力层
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        # 初始化 MLP 权重
        self.mlp.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """自定义权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,
                patch: Tensor,  # Patch 特征, 形状 (B, T, P, D)，P 是每个时间步的 patch 数量
                audio: Tensor,  # 全局音频特征, 形状 (B, T, D)
                video: Tensor,  # 全局视频特征, 形状 (B, T, D)
                ) -> List[Tensor]:
        """
        前向传播。
        Args:
            patch (Tensor): 局部 patch 特征。
            audio (Tensor): 全局音频特征。
            video (Tensor): 全局视频特征。
        Returns:
            List[Tensor]: 包含两个张量，分别是调整后的音频相关 patch 特征贡献和视频相关 patch 特征贡献，
                          形状都为 (B, T, D)。可以看作是从 patch 中提取的、分别对音频和视频重要的信息。
        """
        B, T, P, D = patch.size()  # Batch, Time, Num_Patches, Dim
        # Reshape 特征以进行批处理注意力计算 (将 Batch 和 Time 合并)
        # audio: (B, T, D) -> (B*T, 1, D) # 每个时间步的全局音频特征看作长度为1的序列
        audio_reshaped = audio.reshape(B * T, 1, D)
        # video: (B, T, D) -> (B*T, 1, D) # 同上
        video_reshaped = video.reshape(B * T, 1, D)
        # patch: (B, T, P, D) -> (B*T, P, D) # 每个时间步的 P 个 patch 特征
        patch_reshaped = patch.reshape(B * T, P, D)

        # Permute for MultiheadAttention: (Seq_len, Batch', Dim) where Batch' = B*T
        video_permuted = video_reshaped.permute(1, 0, 2)  # (1, B*T, D)
        audio_permuted = audio_reshaped.permute(1, 0, 2)  # (1, B*T, D)
        patch_permuted = patch_reshaped.permute(1, 0, 2)  # (P, B*T, D)

        # 1. Patch 自注意力: patch 特征内部交互，增强表示
        # patch_permuted (query, key, value) -> output (P, B*T, D)
        # 残差连接
        patch_self_attended = patch_permuted + self.slf_attn(patch_permuted, patch_permuted, patch_permuted)[0]

        # 2. 准备交叉注意力的 query: 拼接全局视频和音频特征
        # query_for_crs_attn 形状: (2, B*T, D)，其中 2 代表视频和音频两个模态
        query_for_crs_attn = torch.cat([video_permuted, audio_permuted], dim=0)

        # 3. 交叉注意力: 全局音视频特征 (query_for_crs_attn) 作为 query，
        #    经过自注意力后的 patch 特征 (patch_self_attended) 作为 key 和 value。
        #    目的是找出 patch 中哪些部分与全局音频/视频相关。
        # attn_output_crs 形状: (2, B*T, D)
        attn_output_crs = self.crs_attn(query_for_crs_attn, patch_self_attended, patch_self_attended)[0]
        # Permute back: (B*T, 2, D)
        attn_output_crs = attn_output_crs.permute(1, 0, 2)
        # 通过 MLP 和 Dropout进行非线性变换和正则化
        mlp_output = self.mlp(self.dropout(attn_output_crs))  # (B*T, 2, D)

        # 将 MLP 输出分割为视频相关和音频相关的部分
        # v_related_patch_info, a_related_patch_info: 每个形状为 (B*T, 1, D)
        v_related_patch_info, a_related_patch_info = torch.chunk(mlp_output, 2, dim=1)

        # Reshape 回原始的 (B, T, D) 格式
        v_final = v_related_patch_info.reshape(B, T, D)
        a_final = a_related_patch_info.reshape(B, T, D)

        # 分别进行层归一化并返回
        return [
            self.anorm(a_final),  # 音频相关的 patch 贡献 (B, T, D)
            self.vnorm(v_final),  # 视频相关的 patch 贡献 (B, T, D)
        ]


# [参考] TSPM 模型中的片段选择部分: https://github.com/GeWu-Lab/TSPM/blob/0106ce4127b8aa6728ee09439a59fbe7f19fe2f4/nets/net.py#L80-L146
class TSPM_topKSelection(nn.Module):
    """
    TSPM (Temporal Segment Proposal Module) 风格的 Top-K 时间片段选择模块。
    这个模块的目的是根据问句特征，从视频（或音频）序列中选择出最重要的 Top-K 个时间片段。
    """

    def __init__(self, topK: int = 10):  # 要选择的 top-K 片段数量，默认为 10
        super(TSPM_topKSelection, self).__init__()

        self.topK = topK
        # 问句-视频片段注意力层 (Qst-Query Attention)
        # d_model=512, nhead=4, dropout=0.1
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        # 后续处理注意力的 FFN (Feed-Forward Network)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        # 层归一化
        self.qst_query_visual_norm = nn.LayerNorm(512)

    def QstQueryClipAttn(self, query_feat: Tensor, kv_feat: Tensor):
        """
        问句-视频片段注意力计算。
        使用问句特征作为 query，视频片段特征序列作为 key 和 value。
        Args:
            query_feat (Tensor): 问句特征, 形状 (Batch, Dim_q)。一般是全局问句表示。
            kv_feat (Tensor): 视频片段特征序列, 形状 (Batch, Seq_kv, Dim_kv)。
        Returns:
            Tuple[Tensor, Tensor]:
                - attn_fused_feat (Tensor): 注意力融合后的特征, 形状 (Batch, Dim_q)。
                - temp_weights (Tensor): 时间注意力权重, 形状 (Batch, 1, Seq_kv)。
                                         表示问句对视频每个时间片段的关注程度。
        """
        # kv_feat: (Batch, Seq_kv, Dim) -> (Seq_kv, Batch, Dim) for MultiheadAttention
        kv_feat = kv_feat.permute(1, 0, 2)
        # query_feat: (Batch, Dim) -> (1, Batch, Dim) for MultiheadAttention query
        query_feat = query_feat.unsqueeze(0)

        # 多头注意力: query_feat 作为 query, kv_feat 作为 key 和 value
        # attn_output: (1, Batch, Dim), temp_weights: (Batch, 1, Seq_kv)
        attn_output, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat,
                                                        attn_mask=None, key_padding_mask=None)
        attn_fused_feat = attn_output.squeeze(0)  # (Batch, Dim)

        # FFN + 残差连接 + 层归一化 (Transformer 解码器/编码器块的典型结构)
        ffn_inter = self.qst_query_linear1(attn_fused_feat)
        ffn_inter = self.qst_query_relu(ffn_inter)
        ffn_inter = self.qst_query_dropout1(ffn_inter)
        ffn_output = self.qst_query_linear2(ffn_inter)
        ffn_output = self.qst_query_dropout2(ffn_output)

        attn_fused_feat = attn_fused_feat + ffn_output  # 残差连接
        attn_fused_feat = self.qst_query_visual_norm(attn_fused_feat)  # 层归一化

        return attn_fused_feat, temp_weights  # 返回融合特征和时间注意力权重

    def SelectTopK(self, temp_weights: Tensor, audio_input: Tensor, visual_input: Tensor,
                   patch_inputs: List[Tensor], B: int, C: int):
        '''
        根据时间注意力权重选择 Top-K 时间片段。
        Args:
            temp_weights (Tensor): 时间注意力权重, 形状 (Batch, 1, T_visual)。
            audio_input (Tensor): 原始音频特征序列, 形状 (Batch, T_audio, C)。
            visual_input (Tensor): 原始视频特征序列, 形状 (Batch, T_visual, C)。
                                   (主要用于确定时间维度 T_visual, 与 temp_weights 对应)。
            patch_inputs (List[Tensor]): patch 特征列表 [audio_patch, video_patch]。
                                         audio_patch: (Batch, T_audio, C)
                                         video_patch: (Batch, T_visual, C)
            B (int): Batch size。
            C (int): 特征维度。
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - output_audio (Tensor): 选择出的 top-K 音频片段, 形状 (Batch, TopK, C)。
                - output_patches (Tuple[Tensor, Tensor]):
                    - (选择出的 top-K 音频 patch, 形状 (Batch, TopK, C))
                    - (选择出的 top-K 视频 patch, 形状 (Batch, TopK, C))
        Assumption: T_audio == T_visual, or indices from visual can be applied to audio.
        '''

        # 根据时间注意力权重排序，得到索引
        # torch.argsort 默认升序，所以权重最高的索引在最后面
        sort_index = torch.argsort(temp_weights, dim=-1)  # (B, 1, T_visual), 升序排序后的原始索引
        # 取最后 K 个索引，即权重最高的 K 个片段的索引
        top_k_indices_unsorted = sort_index[:, :, -self.topK:]  # (B, 1, TopK)
        # 对这 TopK 个索引本身再进行排序，确保选出的片段是按时间顺序的（如果需要）
        top_k_indices_sorted, _ = torch.sort(top_k_indices_unsorted)  # (B, 1, TopK)
        top_k_indices_sorted_np = top_k_indices_sorted.cpu().numpy()  # 转到 CPU 并转为 numpy 数组，方便迭代

        # 初始化输出张量
        output_audio = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_a_patches = torch.zeros(B, self.topK, C).to(audio_input.device)  # 假设 patch_inputs[0] 是音频 patch
        out_v_patches = torch.zeros(B, self.topK, C).to(audio_input.device)  # 假设 patch_inputs[1] 是视频 patch

        # 遍历 batch，根据排好序的 top-K 索引选择对应的音频和 patch 片段
        for batch_idx in range(B):
            current_output_idx = 0
            # 遍历当前 batch 的 top-K 时间索引
            for time_idx_from_visual in top_k_indices_sorted_np.tolist()[batch_idx][0]:
                # 使用从视觉特征中得到的 time_idx_from_visual 来索引音频和patch特征。
                # 这隐含了音频和视频在时间上是对齐的，并且具有相同的时间步长 T。
                output_audio[batch_idx, current_output_idx, :] = audio_input[batch_idx, time_idx_from_visual, :]
                out_a_patches[batch_idx, current_output_idx, :] = patch_inputs[0][batch_idx, time_idx_from_visual, :]
                out_v_patches[batch_idx, current_output_idx, :] = patch_inputs[1][batch_idx, time_idx_from_visual, :]
                current_output_idx += 1
        return output_audio, (out_a_patches, out_v_patches)

    def forward(self, audio_input: Tensor, visual_input: Tensor,
                patch_inputs: List[Tensor], qst_input: Tensor):
        """
        前向传播。
        Args:
            audio_input (Tensor): 音频特征序列, (B, T, C)。
            visual_input (Tensor): 视频特征序列, (B, T, C)。
            patch_inputs (List[Tensor]): Patch 特征列表 [audio_patch (B,T,C), video_patch (B,T,C)]。
            qst_input (Tensor): 问句特征, (B, C_q)。
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - selected_audio (Tensor): 选择的 top-K 音频片段。
                - selected_patches (Tuple[Tensor, Tensor]): 选择的 top-K 音频和视频 patch 片段。
        """
        B, T, C = audio_input.size()  # 获取 batch_size, 时间长度, 特征维度

        # 1. 计算问句对视频片段的时间注意力权重
        # temp_clip_attn_feat 是融合了问句的视频特征 (此处未使用)，主要关注 temp_weights
        _, temp_weights = self.QstQueryClipAttn(qst_input, visual_input)  # temp_weights: (B, 1, T)

        # 2. 根据时间注意力权重选择 Top-K 片段 (音频和对应的patch)
        output_audio, output_patches = self.SelectTopK(temp_weights, audio_input, visual_input, patch_inputs, B, C)

        return output_audio, output_patches
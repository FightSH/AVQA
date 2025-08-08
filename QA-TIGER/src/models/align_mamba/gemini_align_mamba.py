
from mamba_ssm import Mamba
import torch
import torch.nn as nn
import torch.nn.functional as F


class OTLocalAlignment(nn.Module):
    """
    基于最优传输（Optimal Transport）的局部对齐模块。
    该模块实现了论文中描述的松弛版OT，用于将源模态（如视频、音频）
    与目标锚点模态（如语言）进行token级别的对齐。
    """

    def __init__(self):
        super().__init__()

    def _compute_cost_matrix(self, src_tokens, tgt_tokens):
        """
        计算源模态和目标模态token之间的成本矩阵。
        成本定义为 1 - 余弦相似度。

        Args:
            src_tokens (torch.Tensor): 源模态的token序列 (B, T_src, D)
            tgt_tokens (torch.Tensor): 目标模态的token序列 (B, T_tgt, D)

        Returns:
            torch.Tensor: 成本矩阵 (B, T_src, T_tgt)
        """
        # 归一化特征以计算余弦相似度
        src_norm = F.normalize(src_tokens, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_tokens, p=2, dim=-1)

        # 计算余弦相似度矩阵 (B, T_src, T_tgt)
        # (B, T_src, D) @ (B, D, T_tgt) -> (B, T_src, T_tgt)
        cosine_sim = torch.bmm(src_norm, tgt_norm.transpose(1, 2))

        # 成本矩阵 = 1 - 余弦相似度
        cost_matrix = 1 - cosine_sim
        return cost_matrix

    def _get_transport_plan(self, cost_matrix):
        """
        根据成本矩阵计算松弛版OT的传输计划（Transport Plan）。
        对于源模态的每个token，贪婪地找到成本最低的目标token。

        Args:
            cost_matrix (torch.Tensor): 成本矩阵 (B, T_src, T_tgt)

        Returns:
            torch.Tensor: 传输矩阵 (B, T_src, T_tgt)
        """
        B, T_src, T_tgt = cost_matrix.shape

        # 找到每个源token对应的成本最低的目标token的索引
        # min_indices 的形状为 (B, T_src)
        _, min_indices = torch.min(cost_matrix, dim=-1)

        # 构建传输矩阵 M
        # M 是一个稀疏矩阵，每行只有一个非零元素
        # M 的形状为 (B, T_src, T_tgt)
        transport_plan = torch.zeros_like(cost_matrix)

        # 使用 scatter_ 在指定位置填充值
        # 这里的值是 1 / T_src，遵循论文中的归一化
        # unsqueeze(-1) 是为了匹配 scatter_ 的维度要求
        transport_plan.scatter_(dim=-1, index=min_indices.unsqueeze(-1), value=1.0 / T_src)

        return transport_plan

    def forward(self, src_tokens, tgt_tokens):
        """
        执行局部对齐。

        Args:
            src_tokens (torch.Tensor): 源模态特征 (B, T_src, D)
            tgt_tokens (torch.Tensor): 目标（锚点）模态特征 (B, T_tgt, D)

        Returns:
            torch.Tensor: 对齐后的源模态特征 (B, T_tgt, D)
        """
        # 1. 计算成本矩阵
        cost_matrix = self._compute_cost_matrix(src_tokens, tgt_tokens)

        # 2. 计算传输计划
        transport_plan = self._get_transport_plan(cost_matrix)  # (B, T_src, T_tgt)

        # 3. 应用传输计划，生成对齐后的特征
        # (B, T_tgt, T_src) @ (B, T_src, D) -> (B, T_tgt, D)
        # 注意这里需要转置 transport_plan
        aligned_src_tokens = torch.bmm(transport_plan.transpose(1, 2), src_tokens)

        return aligned_src_tokens


class MMDLoss(nn.Module):
    """
    最大均值差异（Maximum Mean Discrepancy）损失。
    用于衡量两个分布之间的差异，这里用它来拉近不同模态的特征分布。
    """

    def __init__(self, kernel_type='gaussian', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算高斯核矩阵。

        Args:
            x (torch.Tensor): 第一个样本集 (B, T, D)
            y (torch.Tensor): 第二个样本集 (B, T, D)

        Returns:
            torch.Tensor: 核矩阵
        """
        B, T, D = x.shape
        # 计算L2距离的平方
        dist = torch.cdist(x, y, p=2).pow(2)  # (B, T, T)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(dist.data) / (B * T * T - B * T)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 计算高斯核值
        kernel_val = [torch.exp(-dist / band) for band in bandwidth_list]
        return sum(kernel_val)

    def forward(self, x, y):
        """
        计算x和y之间的MMD^2损失。

        Args:
            x (torch.Tensor): 第一个分布的样本 (B, T, D)
            y (torch.Tensor): 第二个分布的样本 (B, T, D)

        Returns:
            torch.Tensor: MMD损失值 (一个标量)
        """
        if self.kernel_type == 'gaussian':
            kernel = self._gaussian_kernel
        else:
            raise NotImplementedError("Only Gaussian kernel is implemented.")

        # 计算 K(x, x), K(y, y), K(x, y)
        xx = kernel(x, x, self.kernel_mul, self.kernel_num)
        yy = kernel(y, y, self.kernel_mul, self.kernel_num)
        xy = kernel(x, y, self.kernel_mul, self.kernel_num)

        # 根据MMD^2公式计算损失
        # MMD^2 = E[K(x,x)] + E[K(y,y)] - 2*E[K(x,y)]
        # 这里使用无偏估计
        B, T, _ = x.shape
        loss = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        return loss


class AlignMamba(nn.Module):
    """
    AlignMamba模型，整合了局部对齐、全局对齐和Mamba骨干网络。
    """

    def __init__(self, d_model, n_layers, mmd_lambda=1.0):
        """
        Args:
            d_model (int): 模型特征维度
            n_layers (int): Mamba层数
            mmd_lambda (float): MMD损失的权重
        """
        super().__init__()
        self.d_model = d_model
        self.mmd_lambda = mmd_lambda

        # 1. 局部对齐模块
        self.local_aligner = OTLocalAlignment()

        # 2. 全局对齐损失
        self.global_align_loss_fn = MMDLoss()

        # 3. Mamba骨干网络
        # 假设输入是交错后的序列，所以Mamba层是共享的
        self.mamba_backbone = nn.Sequential(
            *[Mamba(d_model=d_model) for _ in range(n_layers)]
        )

        # 4. 最终的分类头 (示例)
        # 假设对融合后的序列取平均池化后进行分类
        self.classifier = nn.Linear(d_model, 1)  # 二元分类

    def _interleave_features(self, aligned_v, aligned_a, l):
        """
        根据论文描述，将对齐后的模态特征与语言特征进行时间优先的交错。

        Args:
            aligned_v (torch.Tensor): 对齐后的视频特征 (B, T_l, D)
            aligned_a (torch.Tensor): 对齐后的音频特征 (B, T_l, D)
            l (torch.Tensor): 语言特征 (B, T_l, D)

        Returns:
            torch.Tensor: 交错后的统一序列 (B, 3 * T_l, D)
        """
        B, T_l, D = l.shape

        # 将特征堆叠起来 (B, 3, T_l, D)
        stacked_features = torch.stack([aligned_v, aligned_a, l], dim=1)

        # 调整维度顺序并展平
        # (B, 3, T_l, D) -> (B, T_l, 3, D)
        interleaved = stacked_features.transpose(1, 2)
        # (B, T_l, 3, D) -> (B, T_l * 3, D)
        interleaved = interleaved.reshape(B, T_l * 3, D)

        return interleaved

    def forward(self, v, a, l, labels=None):
        """
        AlignMamba的前向传播过程。

        Args:
            v (torch.Tensor): 视频特征 (B, T_v, D)
            a (torch.Tensor): 音频特征 (B, T_a, D)
            l (torch.Tensor): 语言特征 (B, T_l, D)
            labels (torch.Tensor, optional): 真实标签，用于计算损失。

        Returns:
            dict: 包含logits和损失（如果提供了标签）的字典。
        """
        # --- 步骤1: 局部对齐 ---
        # 将视频和音频模态向语言模态对齐
        aligned_v = self.local_aligner(v, l)  # (B, T_l, D)
        aligned_a = self.local_aligner(a, l)  # (B, T_l, D)

        # --- 步骤2: Mamba融合 ---
        # 将对齐后的特征与语言特征交错
        x_mm = self._interleave_features(aligned_v, aligned_a, l)

        # 送入Mamba骨干网络进行融合
        fused_representation = self.mamba_backbone(x_mm)

        # --- 步骤3: 分类与损失计算 ---
        # 使用平均池化得到最终的句子级表示
        pooled_output = fused_representation.mean(dim=1)

        # 通过分类头得到logits
        logits = self.classifier(pooled_output).squeeze(-1)

        # 计算总损失
        total_loss = 0
        if labels is not None:
            # 任务损失 (例如，BCEWithLogitsLoss)
            task_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            # 全局对齐损失 (MMD Loss)
            mmd_loss_v = self.global_align_loss_fn(aligned_v, l)
            mmd_loss_a = self.global_align_loss_fn(aligned_a, l)
            global_align_loss = mmd_loss_v + mmd_loss_a

            # 总损失 = 任务损失 + λ * 对齐损失
            total_loss = task_loss + self.mmd_lambda * global_align_loss

        return {
            'logits': logits,
            'loss': total_loss
        }
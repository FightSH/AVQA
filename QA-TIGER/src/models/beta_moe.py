import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Optional, List

class BetaMoE(nn.Module):
    """
    时间混合专家 (Temporal Mixture of Experts) 模块，使用Beta分布进行时间建模。
    该模块利用问句信息动态地选择和组合多个“专家”网络的输出。
    与高斯版本不同，它使用Beta分布为每个专家生成时间权重，以提供更灵活的时间焦点。
    """

    def __init__(self,
                 d_model: int = 512,         # 模型维度
                 nhead: int = 8,             # 多头注意力头数 (用于问句注意力)
                 topK: int = 5,              # 选择 top-K 个专家进行组合
                 n_experts: int = 10,        # 专家网络总数
                 dropout: float = 0.1,       # Dropout 比率
                 vis_branch: bool = False,   # 指示是否为视觉特定分支
                 ):
        super(BetaMoE, self).__init__()

        self.topK = topK
        self.n_experts = n_experts

        if vis_branch:
            self.anorm = nn.LayerNorm(d_model)
            self.vnorm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        
        # Beta分布参数预测器：一个线性层，为每个专家输出alpha和beta两个参数
        self.beta_pred = nn.Sequential(
            nn.Linear(d_model, 2 * n_experts)
        )
        # 路由网络 (Router)：决定选择哪些专家
        self.router = nn.Sequential(
            nn.Linear(d_model, n_experts)
        )
        # 专家网络列表：每个专家是一个简单的MLP
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(d_model, int(d_model // 2)),
                nn.ReLU(),
                nn.Linear(int(d_model // 2), d_model)
            ])
            for _ in range(n_experts)
        ])
        
        # 初始化权重
        self.experts.apply(self._init_weights)
        self.router.apply(self._init_weights)
        self.beta_pred.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """自定义权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_beta_weights(self,
                              pred: torch.Tensor,  # 预测的Beta分布参数 (Batch, N_experts, 2)
                              topk_inds: torch.Tensor,  # top-K 专家的索引 (Batch, TopK)
                              T: int = 60  # 时间序列的长度
                              ) -> Tensor:
        """
        为选中的 top-K 专家生成Beta分布的时间权重。
        Args:
            pred (torch.Tensor): 所有专家预测的 (alpha, beta) 参数。
            topk_inds (torch.Tensor): 被选中的 top-K 专家的索引。
            T (int): 时间序列的长度。
        Returns:
            Tensor: Beta分布时间权重 (Batch, TopK, T)。
        """
        B, _ = pred.shape[:2]
        # 提取alpha和beta的预测值
        # 使用softplus确保alpha和beta参数为正，并加一个小的epsilon防止为0
        alpha = F.softplus(pred[:, :, 0]) + 1e-6
        beta = F.softplus(pred[:, :, 1]) + 1e-6

        # 根据topk_inds选择对应专家的alpha和beta参数
        selected_alpha = torch.gather(alpha, 1, topk_inds)  # (Batch, TopK)
        selected_beta = torch.gather(beta, 1, topk_inds)    # (Batch, TopK)

        # 创建归一化的时间轴 [0, 1]
        t_axis = torch.linspace(0, 1, T, device=pred.device)
        t_axis = t_axis.view(1, 1, -1).expand(B, self.topK, -1) # (B, TopK, T)

        # 为了数值稳定性，在log空间计算Beta PDF
        # log(t^(a-1)) = (a-1) * log(t)
        # log((1-t)^(b-1)) = (b-1) * log(1-t)
        # log(BetaFunc(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
        # 添加epsilon防止log(0)
        log_t = torch.log(t_axis + 1e-12)
        log_1_minus_t = torch.log(1 - t_axis + 1e-12)

        # 扩展alpha和beta的维度以匹配t_axis
        a = selected_alpha.unsqueeze(-1) # (B, TopK, 1)
        b = selected_beta.unsqueeze(-1)  # (B, TopK, 1)

        # 计算log PDF
        log_pdf = (a - 1) * log_t + (b - 1) * log_1_minus_t
        log_beta_func = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        log_pdf -= log_beta_func

        # 转换回正常空间
        weights = torch.exp(log_pdf) # (B, TopK, T)

        # 对每个专家的权重进行峰值归一化，使其最大值为1
        weights = weights / (weights.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        return weights

    def get_output(self, experts_logits, gauss_weight, topk_inds, topk_probs, shape):
        """与高斯版本相同的输出聚合逻辑"""
        B, T, C = shape
        experts_logits_reshaped = experts_logits.permute(1, 0, 2, 3).reshape(B * T, self.n_experts, C)
        topk_inds_expanded = topk_inds.repeat(T, 1).unsqueeze(-1).repeat(1, 1, C)
        selected_experts_logits = torch.gather(experts_logits_reshaped, 1, topk_inds_expanded)
        selected_experts_logits = selected_experts_logits.reshape(B, T, self.topK, C).contiguous()

        output_per_expert_list = []
        for i in range(self.topK):
            weighted_expert_output = gauss_weight[:, i, :].unsqueeze(1) @ selected_experts_logits[:, :, i, :]
            output_per_expert_list.append(weighted_expert_output)

        output_all_selected_experts = torch.cat(output_per_expert_list, dim=1)
        final_output = topk_probs.unsqueeze(1) @ output_all_selected_experts
        return final_output

    def forward(self, qst, data, sub_data=None):
        B, T, C = data.size()
        data = data.permute(1, 0, 2)
        qst = qst.unsqueeze(0)
        temp_w = self.qst_attn(qst, data, data)[0].squeeze(0)

        router_logits = self.router(temp_w)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_inds = torch.topk(router_probs, self.topK, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 预测Beta分布的参数
        beta_params = self.beta_pred(temp_w)
        beta_params = beta_params.view(B, self.n_experts, 2)
        
        # 生成Beta时间权重
        time_weights = self.generate_beta_weights(beta_params, topk_inds=topk_inds, T=T)

        if sub_data is not None:
            a_sub_data_permuted = sub_data[0].permute(1, 0, 2)
            a_combined_data = data + a_sub_data_permuted
            a_experts_outputs_stack = torch.stack([exprt(a_combined_data) for exprt in self.experts], dim=2)
            a_outs = self.get_output(a_experts_outputs_stack, time_weights, topk_inds, topk_probs, (B, T, C))

            v_sub_data_permuted = sub_data[1].permute(1, 0, 2)
            v_combined_data = data + v_sub_data_permuted
            v_experts_outputs_stack = torch.stack([exprt(v_combined_data) for exprt in self.experts], dim=2)
            v_outs = self.get_output(v_experts_outputs_stack, time_weights, topk_inds, topk_probs, (B, T, C))

            return self.anorm(a_outs), self.vnorm(v_outs)
        else:
            main_experts_outputs_stack = torch.stack([exprt(data) for exprt in self.experts], dim=2)
            main_outs = self.get_output(main_experts_outputs_stack, time_weights, topk_inds, topk_probs, (B, T, C))
            return self.norm(main_outs)

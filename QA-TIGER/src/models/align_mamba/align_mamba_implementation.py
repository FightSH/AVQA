"""
AlignMamba: 基于最优传输和MMD的多模态对齐Mamba框架

本实现严格遵循method.md中的数学公式和算法描述，包含：
1. 基于最优传输的局部跨模态对齐 (公式1-6)
2. 基于MMD的全局跨模态对齐 (公式7-10)  
3. 基于Mamba的融合与优化 (公式11-12)
"""

from typing import Optional, Literal, Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OptimalTransportAlignment(nn.Module):
    """
    基于最优传输的局部跨模态对齐模块
    
    实现论文3.2节中的松弛版OT算法，用于建立不同模态token之间的细粒度对应关系。
    以语言模态为锚点，将音频和视频模态向其对齐。
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def compute_cost_matrix(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算成本矩阵 C_{v2l}(i,j) = 1 - cos(X_v^i, X_l^j)
        
        Args:
            src_tokens: 源模态特征 (B, T_src, D)
            tgt_tokens: 目标模态特征 (B, T_tgt, D)
            
        Returns:
            cost_matrix: 成本矩阵 (B, T_src, T_tgt)
            
        公式(3): C_{v2l}(i,j) = 1 - (X_v(i)·X_l(j)) / (||X_v(i)||_2 ||X_l(j)||_2)
        """
        # L2归一化以计算余弦相似度
        src_norm = F.normalize(src_tokens, p=2, dim=-1, eps=self.eps)  # (B, T_src, D)
        tgt_norm = F.normalize(tgt_tokens, p=2, dim=-1, eps=self.eps)  # (B, T_tgt, D)
        
        # 计算余弦相似度矩阵
        cosine_sim = torch.bmm(src_norm, tgt_norm.transpose(1, 2))  # (B, T_src, T_tgt)
        
        # 成本矩阵 = 1 - 余弦相似度，并确保数值稳定性
        cost_matrix = 1.0 - cosine_sim.clamp(-1.0 + self.eps, 1.0 - self.eps)
        
        return cost_matrix
    
    def solve_relaxed_ot(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        求解松弛版最优传输问题
        
        Args:
            cost_matrix: 成本矩阵 (B, T_src, T_tgt)
            
        Returns:
            transport_matrix: 传输矩阵 (B, T_src, T_tgt)
            
        公式(5): M_{v2l}(i,j) = 1/T_v if j = argmin_j' C_{v2l}(i,j'), else 0
        """
        B, T_src, T_tgt = cost_matrix.shape
        
        # 找到每个源token对应的最小成本目标token
        min_indices = torch.argmin(cost_matrix, dim=-1)  # (B, T_src)
        
        # 构建传输矩阵
        transport_matrix = torch.zeros_like(cost_matrix)
        batch_indices = torch.arange(B, device=cost_matrix.device).unsqueeze(1)
        src_indices = torch.arange(T_src, device=cost_matrix.device).unsqueeze(0)
        
        # 在最小成本位置设置权重 1/T_src
        transport_matrix[batch_indices, src_indices, min_indices] = 1.0 / T_src
        
        return transport_matrix
    
    def apply_transport(self, src_features: torch.Tensor, transport_matrix: torch.Tensor) -> torch.Tensor:
        """
        应用传输矩阵进行特征对齐
        
        Args:
            src_features: 源模态特征 (B, T_src, D)
            transport_matrix: 传输矩阵 (B, T_src, T_tgt)
            
        Returns:
            aligned_features: 对齐后的特征 (B, T_tgt, D)
            
        公式(6): X̃_v = M_{v2l}^T @ X_v
        """
        # M^T @ X: (B, T_tgt, T_src) @ (B, T_src, D) -> (B, T_tgt, D)
        aligned_features = torch.bmm(transport_matrix.transpose(1, 2), src_features)
        return aligned_features
    
    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行完整的OT对齐过程
        
        Args:
            src_tokens: 源模态特征 (B, T_src, D)
            tgt_tokens: 目标锚点模态特征 (B, T_tgt, D)
            
        Returns:
            aligned_features: 对齐后的源模态特征 (B, T_tgt, D)
            transport_matrix: 传输矩阵 (B, T_src, T_tgt)
        """
        # 步骤1: 计算成本矩阵
        cost_matrix = self.compute_cost_matrix(src_tokens, tgt_tokens)
        
        # 步骤2: 求解松弛版OT
        transport_matrix = self.solve_relaxed_ot(cost_matrix)
        
        # 步骤3: 应用传输矩阵
        aligned_features = self.apply_transport(src_tokens, transport_matrix)
        
        return aligned_features, transport_matrix


class MMDGlobalAlignment(nn.Module):
    """
    基于最大均值差异(MMD)的全局跨模态对齐模块
    
    实现论文3.3节中的MMD距离计算，用于确保不同模态在分布级别的一致性。
    """
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算高斯核矩阵
        
        Args:
            x: 第一个特征序列 (B, T, D)
            y: 第二个特征序列 (B, T, D)
            
        Returns:
            kernel_matrix: 核矩阵 (B, T, T)
            
        公式(9): k(x,y) = exp(-||x-y||_2^2 / (2σ^2))
        """
        # 计算平方欧几里得距离
        x_sqnorms = (x ** 2).sum(dim=-1, keepdim=True)  # (B, T, 1)
        y_sqnorms = (y ** 2).sum(dim=-1).unsqueeze(1)   # (B, 1, T)
        xy_dot = torch.bmm(x, y.transpose(1, 2))         # (B, T, T)
        
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        squared_distances = (x_sqnorms + y_sqnorms - 2 * xy_dot).clamp_min(0.0)
        
        # 高斯核
        kernel_matrix = torch.exp(-squared_distances / (2 * self.sigma ** 2))
        
        return kernel_matrix
    
    def compute_mmd_squared(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算MMD^2距离
        
        Args:
            x: 第一个分布的样本 (B, T, D)
            y: 第二个分布的样本 (B, T, D)
            
        Returns:
            mmd_squared: MMD^2距离 (B,)
            
        公式(8): MMD^2(X,Y) = (1/T^2)[∑∑k(x_i,x_i') + ∑∑k(y_j,y_j') - 2∑∑k(x_i,y_j)]
        """
        B, T, D = x.shape
        
        # 计算各项核矩阵
        k_xx = self.gaussian_kernel(x, x)  # (B, T, T)
        k_yy = self.gaussian_kernel(y, y)  # (B, T, T)
        k_xy = self.gaussian_kernel(x, y)  # (B, T, T)
        
        # MMD^2计算
        scale = 1.0 / (T * T)
        mmd_squared = scale * (
            k_xx.sum(dim=(1, 2)) + 
            k_yy.sum(dim=(1, 2)) - 
            2 * k_xy.sum(dim=(1, 2))
        )
        
        return mmd_squared
    
    def forward(self, aligned_v: torch.Tensor, aligned_a: torch.Tensor, 
                anchor_l: torch.Tensor) -> torch.Tensor:
        """
        计算全局对齐损失
        
        Args:
            aligned_v: 对齐后的视频特征 (B, T_l, D)
            aligned_a: 对齐后的音频特征 (B, T_l, D)
            anchor_l: 语言锚点特征 (B, T_l, D)
            
        Returns:
            alignment_loss: 全局对齐损失标量
            
        公式(10): L_align = MMD^2(X̃_v, X_l) + MMD^2(X̃_a, X_l)
        """
        mmd_vl = self.compute_mmd_squared(aligned_v, anchor_l)  # (B,)
        mmd_al = self.compute_mmd_squared(aligned_a, anchor_l)  # (B,)
        
        # 对batch维度求平均
        alignment_loss = (mmd_vl + mmd_al).mean()
        
        return alignment_loss


class MambaBackbone(nn.Module):
    """
    Mamba骨干网络，支持自动回退到Transformer
    
    优先使用mamba-ssm实现，如果不可用则回退到TransformerEncoder
    """
    
    def __init__(self, d_model: int, n_layers: int = 4, dropout: float = 0.1,
                 n_heads: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        
        try:
            # 尝试使用Mamba实现
            from mamba_ssm import Mamba
            
            layers = []
            for _ in range(n_layers):
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                        nn.Dropout(dropout)
                    )
                )
            self.backbone = nn.Sequential(*layers)
            self.is_mamba = True
            
        except ImportError:
            # 回退到Transformer实现
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.is_mamba = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列 (B, T, D)
        Returns:
            output: 输出序列 (B, T, D)
        """
        return self.backbone(x)


class AlignMamba(nn.Module):
    """
    AlignMamba: 整合局部和全局跨模态对齐机制的多模态融合框架
    
    该框架通过以下步骤处理多模态数据：
    1. 单模态编码：将原始信号转换为统一维度的嵌入
    2. 局部对齐：使用OT建立token级别的跨模态对应关系
    3. 全局对齐：使用MMD确保分布级别的一致性
    4. 多模态融合：通过时间优先交错策略和Mamba进行融合
    5. 任务预测：根据任务类型输出最终结果
    """
    
    def __init__(
        self,
        # 模态维度
        dim_audio: int,
        dim_video: int, 
        dim_language: int,
        # 模型配置
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        # 对齐参数
        sigma: float = 1.0,
        lambda_align: float = 0.1,
        ot_eps: float = 1e-8,
        # 任务配置
        task_type: Literal["classification", "regression", "feature_extraction"] = "feature_extraction",
        num_classes: Optional[int] = None,
        pooling: Literal["mean", "max", "last"] = "mean"
    ):
        super().__init__()
        
        # 保存配置
        self.d_model = d_model
        self.task_type = task_type
        self.pooling = pooling
        self.lambda_align = lambda_align
        
        # 单模态投影层
        self.audio_proj = nn.Linear(dim_audio, d_model) if dim_audio != d_model else nn.Identity()
        self.video_proj = nn.Linear(dim_video, d_model) if dim_video != d_model else nn.Identity()
        self.language_proj = nn.Linear(dim_language, d_model) if dim_language != d_model else nn.Identity()
        
        # 对齐模块
        self.ot_aligner = OptimalTransportAlignment(eps=ot_eps)
        self.mmd_aligner = MMDGlobalAlignment(sigma=sigma)
        
        # Mamba骨干网络
        self.backbone = MambaBackbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward
        )
        
        # 任务头
        self._setup_task_head(num_classes)
    
    def _setup_task_head(self, num_classes: Optional[int]):
        """设置任务相关的输出头和损失函数"""
        if self.task_type == "classification":
            assert num_classes is not None and num_classes > 1, "分类任务需要指定类别数"
            self.task_head = nn.Linear(self.d_model, num_classes)
            self.task_loss_fn = nn.CrossEntropyLoss()
        elif self.task_type == "regression":
            out_dim = 1 if num_classes is None else num_classes
            self.task_head = nn.Linear(self.d_model, out_dim)
            self.task_loss_fn = nn.MSELoss()
        else:  # feature_extraction
            self.task_head = nn.Identity()
            self.task_loss_fn = None
    
    def _pool_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        对序列进行池化操作
        
        Args:
            sequence: 输入序列 (B, T, D)
        Returns:
            pooled: 池化后的特征 (B, D)
        """
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        elif self.pooling == "max":
            return sequence.max(dim=1).values
        elif self.pooling == "last":
            return sequence[:, -1, :]
        else:
            raise ValueError(f"不支持的池化方式: {self.pooling}")
    
    def build_interleaved_sequence(self, aligned_video: torch.Tensor, 
                                 aligned_audio: torch.Tensor, 
                                 language: torch.Tensor) -> torch.Tensor:
        """
        构建时间优先的交错多模态序列
        
        Args:
            aligned_video: 对齐后的视频特征 (B, T_l, D)
            aligned_audio: 对齐后的音频特征 (B, T_l, D)
            language: 语言特征 (B, T_l, D)
            
        Returns:
            interleaved_seq: 交错序列 (B, 3*T_l, D)
            
        公式(11): X_mm = [X̃_v^1, X̃_a^1, X_l^1, X̃_v^2, X̃_a^2, X_l^2, ..., X̃_v^T_l, X̃_a^T_l, X_l^T_l]
        """
        B, T_l, D = language.shape
        
        # 将三个模态在时间维度上堆叠: (B, T_l, 3, D)
        stacked = torch.stack([aligned_video, aligned_audio, language], dim=2)
        
        # 重塑为交错序列: (B, T_l*3, D)
        interleaved_seq = stacked.reshape(B, T_l * 3, D)
        
        return interleaved_seq
    
    def forward(
        self,
        audio: torch.Tensor,      # (B, T_a, D_a)
        video: torch.Tensor,      # (B, T_v, D_v)  
        language: torch.Tensor,   # (B, T_l, D_l)
        labels: Optional[torch.Tensor] = None,
        return_alignments: bool = False,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        AlignMamba前向传播
        
        Args:
            audio: 音频特征序列
            video: 视频特征序列
            language: 语言特征序列
            labels: 任务标签（可选）
            return_alignments: 是否返回对齐信息
            return_features: 是否返回中间特征
            
        Returns:
            包含预测结果和损失的字典
        """
        # === 步骤1: 单模态编码 ===
        Xa = self.audio_proj(audio)      # (B, T_a, d_model)
        Xv = self.video_proj(video)      # (B, T_v, d_model)
        Xl = self.language_proj(language) # (B, T_l, d_model)
        
        # === 步骤2: 局部对齐（OT-based） ===
        # 以语言为锚点，对齐音频和视频
        Xa_tilde, Ma2l = self.ot_aligner(Xa, Xl)  # (B, T_l, d_model)
        Xv_tilde, Mv2l = self.ot_aligner(Xv, Xl)  # (B, T_l, d_model)
        
        # === 步骤3: 全局对齐损失（MMD-based） ===
        alignment_loss = self.mmd_aligner(Xv_tilde, Xa_tilde, Xl)
        
        # === 步骤4: 多模态融合 ===
        # 构建交错序列
        X_mm = self.build_interleaved_sequence(Xv_tilde, Xa_tilde, Xl)  # (B, 3*T_l, d_model)
        
        # Mamba骨干网络处理
        fused_features = self.backbone(X_mm)  # (B, 3*T_l, d_model)
        
        # === 步骤5: 任务预测 ===
        outputs = {
            "alignment_loss": alignment_loss
        }
        
        # 根据任务类型处理输出
        if self.task_type != "feature_extraction":
            # 池化得到句子级表示
            pooled_features = self._pool_sequence(fused_features)  # (B, d_model)
            
            # 通过任务头得到预测
            task_output = self.task_head(pooled_features)
            outputs["logits"] = task_output
            
            # 计算任务损失
            if labels is not None and self.task_loss_fn is not None:
                if self.task_type == "regression" and labels.dim() == 1:
                    labels = labels.unsqueeze(-1)
                task_loss = self.task_loss_fn(task_output, labels)
                outputs["task_loss"] = task_loss
                
                # 总损失：公式(12) L = L_task + λ * L_align
                total_loss = task_loss + self.lambda_align * alignment_loss
                outputs["loss"] = total_loss
            else:
                outputs["loss"] = self.lambda_align * alignment_loss
        else:
            # 特征提取模式
            outputs["loss"] = self.lambda_align * alignment_loss
        
        # 可选返回项
        if return_features:
            outputs["features"] = fused_features
            outputs["pooled_features"] = self._pool_sequence(fused_features) if self.task_type != "feature_extraction" else None
        
        if return_alignments:
            outputs["transport_matrices"] = {"audio_to_lang": Ma2l, "video_to_lang": Mv2l}
            outputs["aligned_features"] = {"audio": Xa_tilde, "video": Xv_tilde}
        
        return outputs


# 工厂函数，便于快速创建不同配置的模型
def create_align_mamba_classifier(
    dim_audio: int, dim_video: int, dim_language: int,
    num_classes: int, d_model: int = 256, **kwargs
) -> AlignMamba:
    """创建分类任务的AlignMamba模型"""
    return AlignMamba(
        dim_audio=dim_audio,
        dim_video=dim_video, 
        dim_language=dim_language,
        d_model=d_model,
        task_type="classification",
        num_classes=num_classes,
        **kwargs
    )


def create_align_mamba_regressor(
    dim_audio: int, dim_video: int, dim_language: int,
    output_dim: int = 1, d_model: int = 256, **kwargs
) -> AlignMamba:
    """创建回归任务的AlignMamba模型"""
    return AlignMamba(
        dim_audio=dim_audio,
        dim_video=dim_video,
        dim_language=dim_language, 
        d_model=d_model,
        task_type="regression",
        num_classes=output_dim,
        **kwargs
    )


def create_align_mamba_feature_extractor(
    dim_audio: int, dim_video: int, dim_language: int,
    d_model: int = 256, **kwargs
) -> AlignMamba:
    """创建特征提取的AlignMamba模型"""
    return AlignMamba(
        dim_audio=dim_audio,
        dim_video=dim_video,
        dim_language=dim_language,
        d_model=d_model,
        task_type="feature_extraction",
        **kwargs
    )


if __name__ == "__main__":
    # 示例用法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建分类模型
    model = create_align_mamba_classifier(
        dim_audio=128,
        dim_video=256, 
        dim_language=768,
        num_classes=10,
        d_model=256,
        n_layers=4,
        lambda_align=0.1
    ).to(device)
    
    # 模拟数据
    batch_size = 4
    audio = torch.randn(batch_size, 50, 128).to(device)    # 音频：50帧，128维
    video = torch.randn(batch_size, 30, 256).to(device)    # 视频：30帧，256维
    language = torch.randn(batch_size, 20, 768).to(device) # 语言：20个token，768维
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    # 前向传播
    outputs = model(audio, video, language, labels=labels, return_alignments=True)
    
    print(f"模型使用: {'Mamba' if model.backbone.is_mamba else 'Transformer'}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Total loss: {outputs['loss'].item():.4f}")
    print(f"Task loss: {outputs['task_loss'].item():.4f}")
    print(f"Alignment loss: {outputs['alignment_loss'].item():.4f}")
    print(f"Transport matrix shapes: {[v.shape for v in outputs['transport_matrices'].values()]}")
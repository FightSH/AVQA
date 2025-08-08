import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignMamba(nn.Module):
    """
    AlignMamba框架实现
    """
    
    def __init__(self, audio_dim, video_dim, text_dim, hidden_dim, kernel_bandwidth=1.0):
        super(AlignMamba, self).__init__()
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.kernel_bandwidth = kernel_bandwidth
        
        # 单模态编码器 (这里简化为线性变换)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        self.video_encoder = nn.Linear(video_dim, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        
        # Mamba骨干网络 (这里简化实现)
        self.mamba_backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8), 
            num_layers=2
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def local_alignment(self, src_features, tgt_features):
        """
        基于最优传输的局部跨模态对齐
        实现公式(3)-(6)
        """
        # 计算成本矩阵 (余弦距离)
        # 公式(3)
        src_norm = F.normalize(src_features, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_features, p=2, dim=-1)
        cost_matrix = 1 - torch.matmul(src_norm, tgt_norm.transpose(-2, -1))
        
        # 松弛版OT求解
        # 公式(5)
        transport_matrix = torch.zeros_like(cost_matrix)
        min_indices = torch.argmin(cost_matrix, dim=-1)
        batch_indices = torch.arange(cost_matrix.size(0)).unsqueeze(1)
        time_indices = torch.arange(cost_matrix.size(1)).unsqueeze(0)
        transport_matrix[batch_indices, time_indices, min_indices] = 1.0 / cost_matrix.size(1)
        
        # 生成对齐特征
        # 公式(6)
        aligned_features = torch.matmul(transport_matrix.transpose(-2, -1), src_features)
        
        return aligned_features, transport_matrix
    
    def compute_mmd(self, x, y):
        """
        基于MMD的全局跨模态对齐
        实现公式(7)-(10)
        """
        # 高斯核函数
        # 公式(9)
        def gaussian_kernel(x, y, sigma=self.kernel_bandwidth):
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
            
            x = x.unsqueeze(1)  # (x_size, 1, dim)
            y = y.unsqueeze(0)  # (1, y_size, dim)
            tiled_x = x.expand(x_size, y_size, dim)
            tiled_y = y.expand(x_size, y_size, dim)
            
            kernel = torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / (2 * sigma**2))
            return kernel
        
        # 计算MMD距离
        # 公式(8)
        x_kernel = gaussian_kernel(x, x)
        y_kernel = gaussian_kernel(y, y)
        xy_kernel = gaussian_kernel(x, y)
        
        mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
        return mmd
    
    def global_alignment_loss(self, video_features, audio_features, text_features):
        """
        计算全局对齐损失
        实现公式(10)
        """
        # 计算各模态间的MMD距离
        mmd_vl = self.compute_mmd(video_features, text_features)
        mmd_al = self.compute_mmd(audio_features, text_features)
        
        # 公式(10)
        alignment_loss = mmd_vl + mmd_al
        return alignment_loss
    
    def multimodal_fusion(self, video_features, audio_features, text_features):
        """
        基于Mamba的多模态融合
        实现公式(11)
        """
        # 时间优先的交错策略
        # 公式(11)
        batch_size, seq_len, hidden_dim = text_features.shape
        multimodal_seq = []
        
        for t in range(seq_len):
            multimodal_seq.append(video_features[:, t, :])   # video
            multimodal_seq.append(audio_features[:, t, :])   # audio
            multimodal_seq.append(text_features[:, t, :])    # text
            
        # 组合成统一序列
        multimodal_features = torch.stack(multimodal_seq, dim=1)  # (batch, 3*seq_len, hidden_dim)
        
        # Mamba骨干网络处理
        fused_features = self.mamba_backbone(multimodal_features)
        
        return fused_features
    
    def forward(self, audio_input, video_input, text_input, task_labels=None):
        """
        前向传播过程
        实现整个AlignMamba框架
        """
        # 1. 单模态编码
        # 公式描述中的单模态编码步骤
        audio_embeddings = self.audio_encoder(audio_input)  # Xa
        video_embeddings = self.video_encoder(video_input)  # Xv
        text_embeddings = self.text_encoder(text_input)     # Xl
        
        # 2. 局部对齐 (以text为锚点)
        # 3.2节 OT-based Local Cross-modal Alignment
        aligned_video, _ = self.local_alignment(video_embeddings, text_embeddings)  # X̃v
        aligned_audio, _ = self.local_alignment(audio_embeddings, text_embeddings)  # X̃a
        
        # 3. 全局对齐损失计算 (如果在训练模式)
        # 3.3节 MMD-based Global Cross-modal Alignment
        alignment_loss = None
        if self.training and task_labels is not None:
            alignment_loss = self.global_alignment_loss(
                aligned_video, aligned_audio, text_embeddings
            )  # L_align
        
        # 4. 多模态融合
        # 3.4节 Mamba-based Fusion
        fused_features = self.multimodal_fusion(aligned_video, aligned_audio, text_embeddings)
        
        # 5. 输出处理
        output = self.output_layer(fused_features)
        
        return output, alignment_loss
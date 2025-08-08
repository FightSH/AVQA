import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignMamba(nn.Module):
    def __init__(self, d_model=256, lambda_mmd=0.5):
        super().__init__()
        self.d_model = d_model
        self.lambda_mmd = lambda_mmd
        
        # Mamba backbone (示例结构)
        self.mamba = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, Xa, Xv, Xl):
        """
        Args:
            Xa: Audio features (B, Ta, d)
            Xv: Visual features (B, Tv, d)
            Xl: Language features (B, Tl, d)
        Returns:
            fused_features: 多模态融合特征
            loss: 总损失
        """
        # 3.2 局部对齐
        Xv_align = self.ot_alignment(Xv, Xl)  # 视频到语言对齐
        Xa_align = self.ot_alignment(Xa, Xl)  # 音频到语言对齐
        
        # 3.3 全局对齐损失
        loss_mmd = self.mmd_loss(Xv_align, Xl) + self.mmd_loss(Xa_align, Xl)
        
        # 3.4 多模态特征交错
        X_mm = self.time_priority_interleave(Xv_align, Xa_align, Xl)
        
        # Mamba融合
        fused_features = self.mamba(X_mm)
        
        # 总损失
        loss = self.task_loss(fused_features) + self.lambda_mmd * loss_mmd
        
        return fused_features, loss
    
    def ot_alignment(self, X_source, X_target):
        """基于最优传输的局部对齐"""
        # 计算成本矩阵 (B, Ts, Tt)
        C = 1 - F.cosine_similarity(
            X_source.unsqueeze(2),  # (B, Ts, 1, d)
            X_target.unsqueeze(1),  # (B, 1, Tt, d)
            dim=-1
        )
        
        # 获取传输矩阵 (B, Ts, Tt)
        with torch.no_grad():
            indices = C.argmin(dim=-1)  # (B, Ts)
            M = torch.zeros_like(C)
            M.scatter_(-1, indices.unsqueeze(-1), 1.0/Tv)  # Tv为源模态序列长度
            
        # 对齐特征变换 (B, Tt, d)
        X_align = torch.bmm(M.transpose(1,2), X_source)  # M^T * X_source
        
        return X_align
    
    def mmd_loss(self, X, Y, sigma=1.0):
        """基于高斯核的MMD损失"""
        def gaussian_kernel(x, y):
            dist = torch.cdist(x, y)
            return torch.exp(-dist**2 / (2 * sigma**2))
            
        Kxx = gaussian_kernel(X, X)
        Kyy = gaussian_kernel(Y, Y)
        Kxy = gaussian_kernel(X, Y)
        
        return (Kxx.mean() + Kyy.mean() - 2*Kxy.mean())
    
    def time_priority_interleave(self, Xv, Xa, Xl):
        """时间优先的交错策略"""
        B, Tl, d = Xl.shape
        
        # 创建交错序列 (B, 3*Tl, d)
        X_mm = torch.zeros(B, 3*Tl, d, device=Xl.device)
        X_mm[:, 0::3] = Xv  # 视频特征在位置 0,3,6...
        X_mm[:, 1::3] = Xa  # 音频特征在位置 1,4,7...
        X_mm[:, 2::3] = Xl  # 语言特征在位置 2,5,8...
        
        return X_mm
    
    def task_loss(self, features):
        """任务特定损失（示例）"""
        # 这里假设是分类任务
        logits = torch.randn(features.shape[0], 10)  # 假设输出10类
        return F.cross_entropy(logits, torch.randint(0, 10, (features.shape[0],)))
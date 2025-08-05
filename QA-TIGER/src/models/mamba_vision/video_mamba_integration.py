#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor

from .models.video_mamba_vision import VideoMambaVision


class VideoMambaAdapter(nn.Module):
    """
    适配器，将VideoMambaVision集成到QA-TIGER架构中
    """
    
    def __init__(
        self,
        d_model: int = 512,
        video_dim: int = 512,
        audio_dim: int = 512,
        mamba_hidden_dim: int = 256,
        depths: list = None,
        num_heads: list = None,
        use_pretrained: bool = False,
        use_feature_adapter: bool = False,  # 是否使用特征适配层
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.mamba_hidden_dim = mamba_hidden_dim
        self.use_feature_adapter = use_feature_adapter
        
        # 设置默认配置
        if depths is None:
            depths = [2, 2, 8, 2]  # base配置
        if num_heads is None:
            num_heads = [4, 8, 16, 32]  # base配置
        
        # 过滤掉可能冲突的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['hidden_dim', 'depths', 'num_heads']}
        
        # 创建VideoMambaVision模型
        # 注意：如果传入的是投影后的特征，img_dim和audio_dim都应该是d_model
        self.video_mamba = VideoMambaVision(
            img_dim=d_model,      # 投影后的特征维度
            audio_dim=d_model,    # 投影后的特征维度
            hidden_dim=mamba_hidden_dim,
            depths=depths,
            num_heads=num_heads,
            num_classes=0,  # 不需要分类头，只要特征
            **filtered_kwargs
        )
        
        # 特征维度适配层（可选）
        mamba_output_dim = mamba_hidden_dim * 2  # VideoMambaVision输出维度
        if use_feature_adapter and mamba_output_dim != d_model:
            # 只有在维度不匹配时才需要适配
            self.feature_adapter = nn.Sequential(
                nn.Linear(mamba_output_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif use_feature_adapter:
            # 维度匹配但仍想要特征变换
            self.feature_adapter = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            # 不使用适配层
            self.feature_adapter = nn.Identity()
        
        # 时序特征增强层
        self.temporal_enhance = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, video_seq: Tensor, audio_seq: Tensor) -> Tensor:
        """
        Args:
            video_seq: (B, T, video_dim) 视频特征序列
            audio_seq: (B, T, audio_dim) 音频特征序列
        Returns:
            enhanced_features: (B, T, d_model) 增强后的多模态特征
        """
        # 直接使用VideoMambaVision的时序特征，保留完整的时序信息
        temporal_features = self.video_mamba.forward_temporal_features(video_seq, audio_seq)  # (B, T, mamba_output_dim)
        
        # 特征适配（如果需要）
        if isinstance(self.feature_adapter, nn.Identity):
            # 不需要适配，直接返回
            enhanced_features = temporal_features
        else:
            # 需要适配或特征变换
            enhanced_features = self.feature_adapter(temporal_features)  # (B, T, d_model)
        
        return enhanced_features


class MambaEnhancedQATiger(nn.Module):
    """
    集成VideoMambaVision的增强版QA-TIGER模型
    """
    
    def __init__(
        self,
        d_model: int = 512,
        video_dim: int = 512,
        patch_dim: int = 512,
        audio_dim: int = 128,
        use_video_mamba: bool = True,
        mamba_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        
        self.use_video_mamba = use_video_mamba
        self.d_model = d_model
        
        # VideoMamba适配器
        if use_video_mamba:
            mamba_config = mamba_config or {}
            self.video_mamba_adapter = VideoMambaAdapter(
                d_model=d_model,
                video_dim=video_dim,
                audio_dim=audio_dim,
                **mamba_config
            )
            
            # 特征融合层
            self.mamba_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # 原始特征投影层
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.patch_proj = nn.Linear(patch_dim, d_model)
        
    def forward(self, audio: Tensor, video: Tensor, patch: Tensor) -> tuple:
        """
        Args:
            audio: (B, T, audio_dim)
            video: (B, T, video_dim)  
            patch: (B, T, P, patch_dim)
        Returns:
            enhanced_audio, enhanced_video, enhanced_patch
        """
        # 原始特征投影
        audio_proj = self.audio_proj(audio)  # (B, T, d_model)
        video_proj = self.video_proj(video)  # (B, T, d_model)
        
        # Patch特征处理
        B, T, P, _ = patch.shape
        patch_flat = patch.view(B * T, P, -1)
        patch_proj = self.patch_proj(patch_flat)  # (B*T, P, d_model)
        patch_proj = patch_proj.view(B, T, P, self.d_model)
        
        if self.use_video_mamba:
            # 通过VideoMamba获取增强特征
            mamba_features = self.video_mamba_adapter(video, audio)  # (B, T, d_model)
            
            # 融合原始特征和Mamba特征
            # 音频增强
            audio_combined = torch.cat([audio_proj, mamba_features], dim=-1)  # (B, T, 2*d_model)
            enhanced_audio = self.mamba_fusion(audio_combined)  # (B, T, d_model)
            
            # 视频增强
            video_combined = torch.cat([video_proj, mamba_features], dim=-1)
            enhanced_video = self.mamba_fusion(video_combined)
            
            # Patch增强 - 广播Mamba特征到所有patch
            mamba_expanded = mamba_features.unsqueeze(2).expand(B, T, P, self.d_model)  # (B, T, P, d_model)
            patch_combined = torch.cat([patch_proj, mamba_expanded], dim=-1)  # (B, T, P, 2*d_model)
            patch_combined_flat = patch_combined.view(B * T * P, -1)
            enhanced_patch_flat = self.mamba_fusion(patch_combined_flat)
            enhanced_patch = enhanced_patch_flat.view(B, T, P, self.d_model)
            
        else:
            enhanced_audio = audio_proj
            enhanced_video = video_proj
            enhanced_patch = patch_proj
            
        return enhanced_audio, enhanced_video, enhanced_patch


def create_mamba_enhanced_qa_tiger(
    d_model: int = 512,
    video_dim: int = 512,
    audio_dim: int = 128,
    patch_dim: int = 768,
    use_video_mamba: bool = True,
    mamba_depths: list = [2, 2, 6, 2],
    mamba_num_heads: list = [4, 8, 16, 32],
    **kwargs
) -> MambaEnhancedQATiger:
    """
    创建集成VideoMamba的QA-TIGER模型
    """
    mamba_config = {
        'mamba_hidden_dim': d_model // 2,
        'depths': mamba_depths,
        'num_heads': mamba_num_heads,
        'drop_path_rate': 0.1,
        'layer_scale': 1e-6,
        **kwargs
    }
    
    model = MambaEnhancedQATiger(
        d_model=d_model,
        video_dim=video_dim,
        audio_dim=audio_dim,
        patch_dim=patch_dim,
        use_video_mamba=use_video_mamba,
        mamba_config=mamba_config
    )
    
    return model


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = create_mamba_enhanced_qa_tiger(
        d_model=512,
        video_dim=512,
        audio_dim=128,
        patch_dim=768,
        use_video_mamba=True
    ).to(device)
    
    # 测试数据
    batch_size = 2
    seq_len = 30
    num_patches = 10
    
    audio = torch.randn(batch_size, seq_len, 128).to(device)
    video = torch.randn(batch_size, seq_len, 512).to(device)
    patch = torch.randn(batch_size, seq_len, num_patches, 768).to(device)
    
    # 前向传播
    with torch.no_grad():
        enhanced_audio, enhanced_video, enhanced_patch = model(audio, video, patch)
        
        print(f"输入音频: {audio.shape}")
        print(f"输入视频: {video.shape}")
        print(f"输入patch: {patch.shape}")
        print(f"增强音频: {enhanced_audio.shape}")
        print(f"增强视频: {enhanced_video.shape}")
        print(f"增强patch: {enhanced_patch.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
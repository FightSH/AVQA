#!/usr/bin/env python3

"""
VideoMamba聚合器，用于替代QA-TIGER中的TempMoE聚合器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor

from .models.video_mamba_vision import VideoMambaVision


class VideoMambaAggregator(nn.Module):
    """
    基于VideoMambaVision的时序特征聚合器
    用于替代TempMoE，提供更强的时序建模能力
    """
    
    def __init__(
        self,
        d_model: int = 512,
        mamba_hidden_dim: int = 256,
        depths: list = None,
        num_heads: list = None,
        question_fusion: str = "cross_attn",  # "concat", "cross_attn", "add"
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.question_fusion = question_fusion
        
        # 默认配置
        if depths is None:
            depths = [1, 1, 2, 1]  # 相对较小的配置，适合聚合任务
        if num_heads is None:
            num_heads = [4, 8, 16, 32]
        
        # 问题特征处理
        if question_fusion == "concat":
            # 拼接方式：将问题特征扩展到时序维度并拼接
            self.quest_expand = nn.Linear(d_model, d_model)
            video_input_dim = d_model * 2  # 拼接后的维度
            audio_input_dim = d_model * 2
        elif question_fusion == "cross_attn":
            # 交叉注意力方式：先用注意力融合问题信息
            self.quest_cross_attn = nn.MultiheadAttention(d_model, 8, dropout=dropout)
            video_input_dim = d_model
            audio_input_dim = d_model
        else:  # "add"
            # 相加方式：直接将问题特征加到每个时间步
            video_input_dim = d_model
            audio_input_dim = d_model
        
        # VideoMambaVision核心
        self.video_mamba = VideoMambaVision(
            img_dim=video_input_dim,
            audio_dim=audio_input_dim,
            hidden_dim=mamba_hidden_dim,
            depths=depths,
            num_heads=num_heads,
            num_classes=0,  # 不需要分类头
            drop_rate=dropout,
            **kwargs
        )
        
        # 输出适配层
        mamba_output_dim = mamba_hidden_dim * 2
        self.output_adapter = nn.Sequential(
            nn.Linear(mamba_output_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def _fuse_question(
        self, 
        features: Tensor, 
        question: Tensor
    ) -> Tensor:
        """
        将问题特征融入到时序特征中
        
        Args:
            features: (B, T, D) 时序特征
            question: (B, D) 问题特征
        Returns:
            fused_features: 融合后的特征
        """
        B, T, D = features.shape
        
        if self.question_fusion == "concat":
            # 扩展问题特征到时序维度并拼接
            quest_expanded = self.quest_expand(question).unsqueeze(1).expand(B, T, -1)
            fused = torch.cat([features, quest_expanded], dim=-1)  # (B, T, 2*D)
            return fused
            
        elif self.question_fusion == "cross_attn":
            # 使用交叉注意力融合
            # features作为query，question作为key和value
            quest_expanded = question.unsqueeze(1)  # (B, 1, D)
            features_t = features.transpose(0, 1)  # (T, B, D)
            quest_t = quest_expanded.transpose(0, 1)  # (1, B, D)
            
            attended_features, _ = self.quest_cross_attn(
                features_t, quest_t, quest_t
            )  # (T, B, D)
            return attended_features.transpose(0, 1)  # (B, T, D)
            
        else:  # "add"
            # 直接相加
            quest_expanded = question.unsqueeze(1).expand(B, T, -1)
            return features + quest_expanded
    
    def forward(
        self, 
        question: Tensor, 
        audio: Optional[Tensor] = None, 
        video: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播
        兼容TempMoE的调用方式
        
        Args:
            question: (B, D) 问题特征
            audio: (B, T, D) 音频时序特征
            video: (B, T, D) 视频时序特征
        Returns:
            aggregated: (B, 1, D) 聚合后的全局特征 - 保持与TempMoE一致的输出形状
        """
        # 确保至少有一个模态的输入
        if audio is None and video is None:
            raise ValueError("至少需要提供audio或video中的一个")
        
        # 处理单模态情况
        if audio is None:
            audio = torch.zeros_like(video)
        if video is None:
            video = torch.zeros_like(audio)
        
        # 融合问题信息
        audio_fused = self._fuse_question(audio, question)
        video_fused = self._fuse_question(video, question)
        
        # 通过VideoMambaVision聚合
        global_features = self.video_mamba.forward_features(video_fused, audio_fused)
        
        # 输出适配
        output = self.output_adapter(global_features)  # [B, D]
        
        # 为了与TempMoE的输出格式保持一致，添加时间维度
        # TempMoE输出 (B, 1, D)，所以我们也输出相同格式
        output = output.unsqueeze(1)  # [B, 1, D]
        
        return output


class DualVideoMambaAggregator(nn.Module):
    """
    双输出VideoMamba聚合器
    用于替代vt_aggregator，输出两个全局特征
    兼容TempMoE的调用接口
    """
    
    def __init__(
        self,
        d_model: int = 512,
        mamba_hidden_dim: int = 256,
        depths: list = None,
        num_heads: list = None,
        question_fusion: str = "cross_attn",
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 核心聚合器
        self.aggregator = VideoMambaAggregator(
            d_model=d_model,
            mamba_hidden_dim=mamba_hidden_dim,
            depths=depths,
            num_heads=num_heads,
            question_fusion=question_fusion,
            dropout=dropout,
            **kwargs
        )
        
        # 双头输出
        self.head1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        question: Tensor, 
        video: Tensor, 
        patch
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播，输出两个全局特征
        兼容TempMoE的调用方式: vt_aggregator(quest, video, patch)
        
        Args:
            question: (B, D) 问题特征
            video: (B, T, D) 视频时序特征  
            patch: List[Tensor] 或 Tensor
                   - 如果是List: [audio_patch, video_patch]，每个形状为 [B, T, D]
                   - 如果是Tensor: [B, T, P, D] patch特征
        Returns:
            output1: (B, 1, D) 第一个全局特征 (对应原来的ap_global)
            output2: (B, 1, D) 第二个全局特征 (对应原来的vp_global)
        """
        # 处理不同类型的patch输入
        if isinstance(patch, list):
            # PatchSelecter的输出: [audio_patch, video_patch]，每个形状为 [B, T, D]
            if len(patch) != 2:
                raise ValueError(f"Expected patch list to have 2 elements, got {len(patch)}")
            
            audio_patch, video_patch = patch  # 每个都是 [B, T, D]
            
            # 分别处理video和patch
            video_global = self.aggregator(question, video=video)        # [B, 1, D]
            audio_patch_global = self.aggregator(question, video=audio_patch)  # [B, 1, D]
            video_patch_global = self.aggregator(question, video=video_patch)  # [B, 1, D]
            
            # 先squeeze再通过头处理，最后再unsqueeze
            video_global_squeezed = video_global.squeeze(1)  # [B, D]
            audio_patch_squeezed = audio_patch_global.squeeze(1)  # [B, D]
            video_patch_squeezed = video_patch_global.squeeze(1)  # [B, D]
            
            # 通过不同的头输出
            # 根据TempMoE的逻辑，第一个输出对应audio_patch，第二个对应video_patch
            output1 = self.head1(audio_patch_squeezed)  # [B, D] - ap_global equivalent
            output2 = self.head2(video_patch_squeezed)  # [B, D] - vp_global equivalent
            
        elif isinstance(patch, torch.Tensor):
            # 原始的Tensor输入: [B, T, P, D]
            B, T, P, D = patch.shape  # 例如: [2, 60, 14, 512]
            
            # 将patch特征展平为时序特征
            patch_flattened = patch.view(B, T * P, D)  # [B, 60*14, 512] = [B, 840, 512]
            
            # 分别处理video和patch
            video_global = self.aggregator(question, video=video)      # [B, 1, D]
            patch_global = self.aggregator(question, video=patch_flattened)  # [B, 1, D]
            
            # 先squeeze再通过头处理，最后再unsqueeze
            video_global_squeezed = video_global.squeeze(1)  # [B, D]
            patch_global_squeezed = patch_global.squeeze(1)  # [B, D]
            
            # 通过不同的头输出
            output1 = self.head1(patch_global_squeezed)  # [B, D] - ap_global equivalent
            output2 = self.head2(video_global_squeezed)  # [B, D] - vp_global equivalent
            
        else:
            raise TypeError(f"Expected patch to be torch.Tensor or List[Tensor], got {type(patch)}")
        
        # 添加时间维度以匹配TempMoE的输出格式
        output1 = output1.unsqueeze(1)  # [B, 1, D]
        output2 = output2.unsqueeze(1)  # [B, 1, D]
        
        return output1, output2


class UnifiedVideoMambaAggregator(nn.Module):
    """
    统一的VideoMamba聚合器
    将所有模态统一处理
    """
    
    def __init__(
        self,
        d_model: int = 512,
        mamba_hidden_dim: int = 256,
        depths: list = None,
        num_heads: list = None,
        question_fusion: str = "concat",
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 模态特定的预处理
        self.audio_preprocess = nn.Linear(d_model, d_model)
        self.video_preprocess = nn.Linear(d_model, d_model)
        self.patch_preprocess = nn.Linear(d_model, d_model)
        
        # 核心聚合器
        self.aggregator = VideoMambaAggregator(
            d_model=d_model,
            mamba_hidden_dim=mamba_hidden_dim,
            depths=depths,
            num_heads=num_heads,
            question_fusion=question_fusion,
            dropout=dropout,
            **kwargs
        )
        
        # 多头输出，对应原来的不同全局特征
        self.audio_head = nn.Linear(d_model, d_model)
        self.video_head = nn.Linear(d_model, d_model)
        self.patch_head = nn.Linear(d_model, d_model)
    
    def forward(
        self, 
        question: Tensor, 
        audio: Tensor, 
        video: Tensor, 
        patch: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        统一处理所有模态
        
        Args:
            question: (B, D) 问题特征
            audio: (B, T, D) 音频特征
            video: (B, T, D) 视频特征
            patch: (B, T, P, D) patch特征
        Returns:
            a_global: (B, D) 音频全局特征
            v_global: (B, D) 视频全局特征
            p_global: (B, D) patch全局特征
        """
        B, T, P, D = patch.shape
        
        # 预处理
        audio_proc = self.audio_preprocess(audio)
        video_proc = self.video_preprocess(video)
        patch_proc = self.patch_preprocess(patch.view(B, T * P, D))
        
        # 拼接所有模态 (可选的统一处理方式)
        # 这里我们分别处理，保持灵活性
        
        # 分别聚合
        a_global = self.aggregator(question, audio=audio_proc)
        v_global = self.aggregator(question, video=video_proc)
        p_global = self.aggregator(question, video=patch_proc)  # patch当作video处理
        
        # 通过不同头输出
        a_global = self.audio_head(a_global)
        v_global = self.video_head(v_global)
        p_global = self.patch_head(p_global)
        
        return a_global, v_global, p_global


def create_mamba_aggregator(
    aggregator_type: str = "single",
    d_model: int = 512,
    mamba_config: dict = None,
    **kwargs
) -> nn.Module:
    """
    工厂函数，创建不同类型的VideoMamba聚合器
    
    Args:
        aggregator_type: "single", "dual", "unified"
        d_model: 模型维度
        mamba_config: VideoMamba配置
    Returns:
        聚合器实例
    """
    if mamba_config is None:
        mamba_config = {}
    
    config = {
        'd_model': d_model,
        'mamba_hidden_dim': mamba_config.get('mamba_hidden_dim', d_model // 2),
        'depths': mamba_config.get('depths', [1, 1, 2, 1]),
        'num_heads': mamba_config.get('num_heads', [4, 8, 16, 32]),
        'question_fusion': mamba_config.get('question_fusion', 'concat'),
        'dropout': mamba_config.get('dropout', 0.1),
        **kwargs
    }
    
    if aggregator_type == "single":
        return VideoMambaAggregator(**config)
    elif aggregator_type == "dual":
        return DualVideoMambaAggregator(**config)
    elif aggregator_type == "unified":
        return UnifiedVideoMambaAggregator(**config)
    else:
        raise ValueError(f"Unknown aggregator_type: {aggregator_type}")


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试单输出聚合器
    print("测试VideoMambaAggregator...")
    aggregator = VideoMambaAggregator(d_model=512).to(device)
    
    question = torch.randn(2, 512).to(device)
    audio = torch.randn(2, 60, 512).to(device)
    video = torch.randn(2, 60, 512).to(device)
    
    with torch.no_grad():
        output = aggregator(question, audio, video)
        print(f"单输出聚合器输出形状: {output.shape}")  # 期望: [2, 1, 512]
    
    # 测试双输出聚合器 - 使用实际的patch维度
    print("\n测试DualVideoMambaAggregator...")
    dual_aggregator = DualVideoMambaAggregator(d_model=512).to(device)
    
    # 测试Tensor输入
    patch_tensor = torch.randn(2, 60, 14, 512).to(device)  # 实际的patch维度
    with torch.no_grad():
        out1, out2 = dual_aggregator(question, video, patch_tensor)
        print(f"Tensor输入 - 输出1形状: {out1.shape}, 输出2形状: {out2.shape}")  # 期望: [2, 1, 512], [2, 1, 512]
    
    # 测试List输入 (PatchSelecter的输出格式)
    patch_list = [torch.randn(2, 60, 512).to(device), torch.randn(2, 60, 512).to(device)]  # [audio_patch, video_patch]
    with torch.no_grad():
        out1, out2 = dual_aggregator(question, video, patch_list)
        print(f"List输入 - 输出1形状: {out1.shape}, 输出2形状: {out2.shape}")  # 期望: [2, 1, 512], [2, 1, 512]
    
    print("✅ 所有测试通过！")
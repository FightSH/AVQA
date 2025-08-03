#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class QuestionAwareMultiModalPreprocessor(nn.Module):
    """问题感知的多模态特征预处理器"""
    
    def __init__(self, img_dim, audio_dim, question_dim, hidden_dim, 
                 fusion_strategy='cross_attention', dropout=0.1):
        super().__init__()
        self.img_dim = img_dim
        self.audio_dim = audio_dim
        self.question_dim = question_dim
        self.hidden_dim = hidden_dim
        self.fusion_strategy = fusion_strategy
        
        # 基础投影层
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.question_proj = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 根据融合策略选择不同的融合模块
        if fusion_strategy == 'early_concat':
            self.fusion_module = self._build_early_concat_fusion(hidden_dim)
        elif fusion_strategy == 'cross_attention':
            self.fusion_module = self._build_cross_attention_fusion(hidden_dim)
        elif fusion_strategy == 'question_guided':
            self.fusion_module = self._build_question_guided_fusion(hidden_dim)
        elif fusion_strategy == 'adaptive_fusion':
            self.fusion_module = self._build_adaptive_fusion(hidden_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # 模态嵌入
        self.img_modal_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.audio_modal_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.question_modal_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def _build_early_concat_fusion(self, hidden_dim):
        """策略1: 早期拼接融合"""
        return nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 3个模态 -> 2个模态输出
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def _build_cross_attention_fusion(self, hidden_dim):
        """策略2: 交叉注意力融合"""
        class CrossAttentionFusion(nn.Module):
            def __init__(self, dim, num_heads=8):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.scale = self.head_dim ** -0.5
                
                # 为视频特征生成 Q，为问题生成 K,V
                self.q_proj = nn.Linear(dim, dim)
                self.kv_proj = nn.Linear(dim, dim * 2)
                self.out_proj = nn.Linear(dim, dim)
                
            def forward(self, video_feat, question_feat):
                B, T, D = video_feat.shape
                
                # Q 来自视频特征，K,V 来自问题特征
                q = self.q_proj(video_feat)  # (B, T, D)
                kv = self.kv_proj(question_feat.unsqueeze(1))  # (B, 1, 2D)
                k, v = kv.chunk(2, dim=-1)  # 各自 (B, 1, D)
                
                # 重塑为多头格式
                q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
                k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, 1, d)
                v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, 1, d)
                
                # 计算注意力
                attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, 1)
                attn = attn.softmax(dim=-1)
                
                # 应用注意力
                out = (attn @ v).transpose(1, 2).reshape(B, T, D)  # (B, T, D)
                return self.out_proj(out)
        
        return CrossAttentionFusion(hidden_dim)
    
    def _build_question_guided_fusion(self, hidden_dim):
        """策略3: 问题引导的融合"""
        class QuestionGuidedFusion(nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 问题生成门控信号
                self.img_gate = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.Sigmoid()
                )
                self.audio_gate = nn.Sequential(
                    nn.Linear(dim, dim), 
                    nn.Sigmoid()
                )
                # 特征融合
                self.fusion_proj = nn.Linear(dim * 2, dim * 2)
                
            def forward(self, img_feat, audio_feat, question_feat):
                B, T, D = img_feat.shape
                
                # 问题特征扩展到序列长度
                question_expanded = question_feat.unsqueeze(1).expand(B, T, D)
                
                # 生成门控信号
                img_gate = self.img_gate(question_expanded)
                audio_gate = self.audio_gate(question_expanded)
                
                # 应用门控
                gated_img = img_feat * img_gate
                gated_audio = audio_feat * audio_gate
                
                # 融合
                fused = torch.cat([gated_img, gated_audio], dim=-1)
                return self.fusion_proj(fused)
        
        return QuestionGuidedFusion(hidden_dim)
    
    def _build_adaptive_fusion(self, hidden_dim):
        """策略4: 自适应融合"""
        class AdaptiveFusion(nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 学习融合权重
                self.weight_net = nn.Sequential(
                    nn.Linear(dim * 3, dim),
                    nn.ReLU(),
                    nn.Linear(dim, 3),  # 3个模态的权重
                    nn.Softmax(dim=-1)
                )
                self.fusion_proj = nn.Linear(dim, dim * 2)
                
            def forward(self, img_feat, audio_feat, question_feat):
                B, T, D = img_feat.shape
                
                # 计算全局特征用于权重计算
                img_global = img_feat.mean(dim=1)      # (B, D)
                audio_global = audio_feat.mean(dim=1)  # (B, D)
                
                # 计算融合权重
                combined_global = torch.cat([img_global, audio_global, question_feat], dim=-1)
                weights = self.weight_net(combined_global)  # (B, 3)
                
                # 扩展权重到序列维度
                w_img = weights[:, 0:1].unsqueeze(1).expand(B, T, D)      # (B, T, D)
                w_audio = weights[:, 1:2].unsqueeze(1).expand(B, T, D)    # (B, T, D)
                w_question = weights[:, 2:3].unsqueeze(1).expand(B, T, D) # (B, T, D)
                
                # 加权融合
                question_expanded = question_feat.unsqueeze(1).expand(B, T, D)
                weighted_sum = (w_img * img_feat + 
                              w_audio * audio_feat + 
                              w_question * question_expanded)
                
                return self.fusion_proj(weighted_sum)
        
        return AdaptiveFusion(hidden_dim)
    
    def forward(self, img_feat, audio_feat, question_feat):
        """
        Args:
            img_feat: (B, T, img_dim) 图像特征序列
            audio_feat: (B, T, audio_dim) 音频特征序列
            question_feat: (B, question_dim) 问题特征
        Returns:
            fused_feat: (B, T, hidden_dim * 2) 融合后的特征
        """
        B, T = img_feat.shape[:2]
        
        # 投影到统一空间
        img_proj = self.img_proj(img_feat)          # (B, T, hidden_dim)
        audio_proj = self.audio_proj(audio_feat)    # (B, T, hidden_dim)
        question_proj = self.question_proj(question_feat)  # (B, hidden_dim)
        
        # 添加模态嵌入
        img_proj = img_proj + self.img_modal_embed.expand(B, T, -1)
        audio_proj = audio_proj + self.audio_modal_embed.expand(B, T, -1)
        question_proj = question_proj + self.question_modal_embed.squeeze(1).expand(B, -1)
        
        # 根据策略进行融合
        if self.fusion_strategy == 'early_concat':
            # 问题特征扩展到序列长度
            question_expanded = question_proj.unsqueeze(1).expand(B, T, -1)
            # 拼接三个模态
            concat_feat = torch.cat([img_proj, audio_proj, question_expanded], dim=-1)
            fused_feat = self.fusion_module(concat_feat)
            
        elif self.fusion_strategy == 'cross_attention':
            # 先融合图像和音频
            video_feat = torch.cat([img_proj, audio_proj], dim=-1)  # (B, T, 2*hidden_dim)
            # 使用交叉注意力融合问题
            fused_feat = self.fusion_module(video_feat, question_proj)
            
        elif self.fusion_strategy == 'question_guided':
            fused_feat = self.fusion_module(img_proj, audio_proj, question_proj)
            
        elif self.fusion_strategy == 'adaptive_fusion':
            fused_feat = self.fusion_module(img_proj, audio_proj, question_proj)
        
        return fused_feat


class QuestionAwareVideoMambaVision(nn.Module):
    """问题感知的视频 MambaVision 模型"""
    
    def __init__(
        self,
        img_dim=2048,
        audio_dim=128,
        question_dim=512,
        hidden_dim=256,
        depths=[2, 2, 8, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.,
        num_classes=1000,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        layer_scale=1e-6,
        causal=False,
        fusion_strategy='cross_attention',  # 融合策略
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fusion_strategy = fusion_strategy
        
        # 问题感知的多模态预处理
        self.multimodal_prep = QuestionAwareMultiModalPreprocessor(
            img_dim=img_dim,
            audio_dim=audio_dim,
            question_dim=question_dim,
            hidden_dim=hidden_dim,
            fusion_strategy=fusion_strategy,
            dropout=drop_rate
        )
        
        # 时序位置编码
        self.pos_encoding = TemporalPositionalEncoding(hidden_dim * 2)
        
        # 构建 Mamba 层（复用之前的实现）
        feature_dim = hidden_dim * 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        
        for stage_idx, depth in enumerate(depths):
            stage_blocks = nn.ModuleList()
            
            for block_idx in range(depth):
                global_block_idx = sum(depths[:stage_idx]) + block_idx
                
                if stage_idx < 2:
                    # 时序卷积块
                    from .video_mamba_vision import TemporalConvBlock
                    block = TemporalConvBlock(
                        dim=feature_dim,
                        drop_path=dpr[global_block_idx],
                        layer_scale=layer_scale
                    )
                else:
                    # Mamba 或 Attention 块
                    mixer_type = "attention" if block_idx >= depth // 2 else "mamba"
                    
                    from .video_mamba_vision import VideoMambaBlock
                    block = VideoMambaBlock(
                        dim=feature_dim,
                        mixer_type=mixer_type,
                        num_heads=num_heads[stage_idx],
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[global_block_idx],
                        layer_scale=layer_scale,
                        causal=causal,
                    )
                
                stage_blocks.append(block)
            
            self.stages.append(stage_blocks)
        
        # 最终归一化和分类头
        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Linear(feature_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, img_seq, audio_seq, question_vec):
        """
        Args:
            img_seq: (B, T, img_dim) 图像特征序列
            audio_seq: (B, T, audio_dim) 音频特征序列
            question_vec: (B, question_dim) 问题向量
        Returns:
            x: (B, feature_dim) 全局特征
        """
        # 问题感知的多模态融合
        x = self.multimodal_prep(img_seq, audio_seq, question_vec)  # (B, T, hidden_dim * 2)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 通过各阶段
        for stage in self.stages:
            for block in stage:
                x = block(x)
        
        # 最终归一化
        x = self.norm(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (B, hidden_dim * 2)
        
        return x
    
    def forward(self, img_seq, audio_seq, question_vec):
        """
        Args:
            img_seq: (B, T, img_dim) 图像特征序列
            audio_seq: (B, T, audio_dim) 音频特征序列
            question_vec: (B, question_dim) 问题向量
        Returns:
            logits: (B, num_classes) 分类结果
        """
        x = self.forward_features(img_seq, audio_seq, question_vec)
        x = self.head(x)
        return x


# 导入必要的组件
class TemporalPositionalEncoding(nn.Module):
    """时序位置编码"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


# 便捷的模型构建函数
def question_aware_video_mamba_base(fusion_strategy='cross_attention', **kwargs):
    """构建问题感知的视频 MambaVision Base 模型"""
    model = QuestionAwareVideoMambaVision(
        hidden_dim=256,
        depths=[2, 2, 8, 2],
        num_heads=[4, 8, 16, 32],
        fusion_strategy=fusion_strategy,
        **kwargs
    )
    return model


def question_aware_video_mamba_small(fusion_strategy='cross_attention', **kwargs):
    """构建问题感知的视频 MambaVision Small 模型"""
    model = QuestionAwareVideoMambaVision(
        hidden_dim=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        fusion_strategy=fusion_strategy,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试不同融合策略
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    strategies = ['early_concat', 'cross_attention', 'question_guided', 'adaptive_fusion']
    
    for strategy in strategies:
        print(f"\n测试融合策略: {strategy}")
        
        model = question_aware_video_mamba_small(
            img_dim=2048,
            audio_dim=128,
            question_dim=512,
            num_classes=100,
            fusion_strategy=strategy
        ).to(device)
        
        # 测试数据
        batch_size, seq_len = 2, 20
        img_features = torch.randn(batch_size, seq_len, 2048).to(device)
        audio_features = torch.randn(batch_size, seq_len, 128).to(device)
        question_vector = torch.randn(batch_size, 512).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(img_features, audio_features, question_vector)
            
        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
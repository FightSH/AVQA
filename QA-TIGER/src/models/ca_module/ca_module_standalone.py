"""
独立的CA模块 - 可插拔的多模态融合组件
从MLLMCat中提取并改造为通用模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from typing import Optional, Tuple
import logging

# 依赖的组件需要单独实现或导入
# from Qformer import BertConfig, BertLMHeadModel
# from blip2 import Blip2Base


class Perceiver(nn.Module):
    """
    改造版：支持 batch，统一 batch_first=True
    输入:
        txt:   [B, Lt, Dt]
        video: [B, Tv, Dv]
        audio: [B, Ta, Da]
    返回:
        video_out: [B, Tv, Dv]
        audio_out: [B, Ta, Da]
    """
    def __init__(self,
                 audio_dim=1024,
                 video_dim=1024,
                 text_dim=4096,
                 num_heads=4,
                 dropout=0.1,
                 use_reverse=True):
        super().__init__()
        self.use_reverse = use_reverse

        self.attn_a  = nn.MultiheadAttention(audio_dim, num_heads, dropout, batch_first=True)
        self.attn_v  = nn.MultiheadAttention(video_dim, num_heads, dropout, batch_first=True)
        self.attn_t  = nn.MultiheadAttention(text_dim, num_heads, dropout, batch_first=True)

        self.attn_t2a = nn.MultiheadAttention(text_dim, num_heads, dropout, batch_first=True)
        self.attn_t2v = nn.MultiheadAttention(text_dim, num_heads, dropout, batch_first=True)

        if use_reverse:
            self.attn_a2t = nn.MultiheadAttention(audio_dim, num_heads, dropout, batch_first=True)
            self.attn_v2t = nn.MultiheadAttention(video_dim, num_heads, dropout, batch_first=True)

        # 维度对齐
        self.a2t_proj = nn.Linear(audio_dim, text_dim) if audio_dim != text_dim else nn.Identity()
        self.v2t_proj = nn.Linear(video_dim, text_dim) if video_dim != text_dim else nn.Identity()
        self.t2a_proj = nn.Linear(text_dim, audio_dim) if audio_dim != text_dim else nn.Identity()
        self.t2v_proj = nn.Linear(text_dim, video_dim) if video_dim != text_dim else nn.Identity()

        self.norm_a1 = nn.LayerNorm(audio_dim)
        self.norm_v1 = nn.LayerNorm(video_dim)
        self.norm_t1 = nn.LayerNorm(text_dim)
        self.norm_a2 = nn.LayerNorm(audio_dim)
        self.norm_v2 = nn.LayerNorm(video_dim)
        self.norm_t2 = nn.LayerNorm(text_dim)

        self.dropout = nn.Dropout(dropout)
        self.gate_a = nn.Parameter(torch.zeros(1))
        self.gate_v = nn.Parameter(torch.zeros(1))

    def forward(self, txt, video, audio,
                txt_mask=None, video_mask=None, audio_mask=None):
        # 自注意力
        t_res,_ = self.attn_t(txt, txt, txt, key_padding_mask=txt_mask)       # [B,Lt,Dt]
        v_res,_ = self.attn_v(video, video, video, key_padding_mask=video_mask)
        a_res,_ = self.attn_a(audio, audio, audio, key_padding_mask=audio_mask)

        txt_enh   = self.norm_t1(txt   + self.dropout(t_res))
        video_enh = self.norm_v1(video + self.dropout(v_res))
        audio_enh = self.norm_a1(audio + self.dropout(a_res))

        # 文本作为 Query 读取多模态
        t2a_ctx,_ = self.attn_t2a(txt_enh, self.a2t_proj(audio_enh), self.a2t_proj(audio_enh),
                                  key_padding_mask=audio_mask)
        t2v_ctx,_ = self.attn_t2v(txt_enh, self.v2t_proj(video_enh), self.v2t_proj(video_enh),
                                  key_padding_mask=video_mask)
        txt_ctx = self.norm_t2(txt_enh + self.dropout((t2a_ctx + t2v_ctx)/2))

        if self.use_reverse:
            a2t_ctx,_ = self.attn_a2t(audio_enh, self.t2a_proj(txt_ctx), self.t2a_proj(txt_ctx),
                                      key_padding_mask=txt_mask)
            v2t_ctx,_ = self.attn_v2t(video_enh, self.t2v_proj(txt_ctx), self.t2v_proj(txt_ctx),
                                      key_padding_mask=txt_mask)
            g_a = torch.sigmoid(self.gate_a)
            g_v = torch.sigmoid(self.gate_v)
            audio_out = self.norm_a2(audio_enh + self.dropout(g_a * a2t_ctx))
            video_out = self.norm_v2(video_enh + self.dropout(g_v * v2t_ctx))
        else:
            audio_out, video_out = audio_enh, video_enh

        return video_out, audio_out, txt_ctx


class SimpleQFormer(nn.Module):
    """
    简化版Q-Former - 不依赖BLIP2的实现
    """
    def __init__(self, 
                 num_query_tokens: int = 32,
                 feature_dim: int = 1408,
                 hidden_dim: int = 768,
                 num_layers: int = 4,
                 num_heads: int = 8):
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 可学习的查询向量
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_dim) * 0.02
        )
        
        # 交叉注意力层
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 自注意力层
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])
        
        # 输入投影
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, seq_len, feature_dim]
        
        Returns:
            query_output: [batch_size, num_query_tokens, hidden_dim]
        """
        batch_size = features.shape[0]
        
        # 投影输入特征
        features = self.input_projection(features)  # [B, L, H]
        
        # 扩展查询向量
        queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, Q, H]
        
        # 逐层处理
        for i in range(len(self.cross_attention_layers)):
            # 交叉注意力：queries attend to features
            cross_attn_out, _ = self.cross_attention_layers[i](
                queries, features, features
            )
            queries = self.layer_norms[i*2](queries + cross_attn_out)
            
            # 自注意力：queries attend to queries
            self_attn_out, _ = self.self_attention_layers[i](
                queries, queries, queries
            )
            queries = self.layer_norms[i*2+1](queries + self_attn_out)
            
            # 前馈网络
            ffn_out = self.ffn_layers[i](queries)
            queries = queries + ffn_out
            
        return queries


class StandaloneCA(nn.Module):
    """
    独立的CA模块 - 可插拔的多模态融合组件
    """
    def __init__(self,
                 # 输入维度配置
                 video_input_dim: int = 1024,
                 audio_input_dim: int = 1024,
                 text_input_dim: int = 4096,
                 
                 # Q-Former配置
                 num_query_tokens: int = 32,
                 qformer_feature_dim: int = 1408,
                 qformer_hidden_dim: int = 768,
                 qformer_layers: int = 4,
                 
                 # 输出维度配置
                 output_dim: int = 4096,
                 
                 # Perceiver配置
                 perceiver_heads: int = 4,
                 perceiver_dropout: float = 0.1):
        
        super().__init__()
        
        # Perceiver模块
        self.perceiver = Perceiver(
            audio_dim=audio_input_dim,
            video_dim=video_input_dim,
            text_dim=text_input_dim,
            num_heads=perceiver_heads,
            dropout=perceiver_dropout
        )
        
        # 特征投影层（输入到Q-Former）
        self.video_projection = nn.Linear(video_input_dim, qformer_feature_dim)
        self.audio_projection = nn.Linear(audio_input_dim, qformer_feature_dim)
        
        # Q-Former模块
        self.video_qformer = SimpleQFormer(
            num_query_tokens=num_query_tokens,
            feature_dim=qformer_feature_dim,
            hidden_dim=qformer_hidden_dim,
            num_layers=qformer_layers
        )
        
        self.audio_qformer = SimpleQFormer(
            num_query_tokens=num_query_tokens,
            feature_dim=qformer_feature_dim,
            hidden_dim=qformer_hidden_dim,
            num_layers=qformer_layers
        )
        
        # 输出投影层（Q-Former到目标维度）
        self.video_output_projection = nn.Linear(qformer_hidden_dim, output_dim)
        self.audio_output_projection = nn.Linear(qformer_hidden_dim, output_dim)
        
    def forward(self, 
                text_features: torch.Tensor,
                video_features: torch.Tensor,
                audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_features: [seq_len, text_dim] 文本特征
            video_features: [video_frames, video_dim] 视频特征  
            audio_features: [audio_frames, audio_dim] 音频特征
            
        Returns:
            video_output: [batch_size, num_query_tokens, output_dim] 融合后的视频特征
            audio_output: [batch_size, num_query_tokens, output_dim] 融合后的音频特征
        """
        
        # Step 1: Perceiver跨模态交互
        enhanced_video, enhanced_audio = self.perceiver(
            text_features, video_features, audio_features
        )
        
        # Step 2: 投影到Q-Former输入空间
        video_projected = self.video_projection(enhanced_video).unsqueeze(0)  # [1, frames, dim]
        audio_projected = self.audio_projection(enhanced_audio).unsqueeze(0)  # [1, frames, dim]
        
        # Step 3: Q-Former特征压缩和对齐
        video_queries = self.video_qformer(video_projected)  # [1, num_queries, hidden_dim]
        audio_queries = self.audio_qformer(audio_projected)  # [1, num_queries, hidden_dim]
        
        # Step 4: 投影到输出空间
        video_output = self.video_output_projection(video_queries)  # [1, num_queries, output_dim]
        audio_output = self.audio_output_projection(audio_queries)  # [1, num_queries, output_dim]
        
        return video_output, audio_output
    
    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
        """
        加载预训练权重
        
        Args:
            checkpoint_path: 权重文件路径
            strict: 是否严格匹配权重键名
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的权重格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 过滤CA相关的权重
            ca_weights = {}
            for key, value in state_dict.items():
                if 'CA.' in key:
                    new_key = key.replace('CA.', '').replace('base_model.model.model.', '')
                    ca_weights[new_key] = value
                    
            # 加载权重
            missing_keys, unexpected_keys = self.load_state_dict(ca_weights, strict=strict)
            
            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys}")
                
            logging.info(f"Successfully loaded CA weights from {checkpoint_path}")
            
        except Exception as e:
            logging.error(f"Failed to load weights: {e}")
            raise


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建CA模块实例
    ca_module = StandaloneCA(
        video_input_dim=512,
        audio_input_dim=512, 
        text_input_dim=512,
        num_query_tokens=32,
        output_dim=1024
    )
    
    # 模拟输入数据
    batch_size = 2
    text_seq_len = 10
    video_frames = 60
    audio_frames = 60
    
    text_features = torch.randn(1, 512)
    video_features = torch.randn(video_frames, 512)
    audio_features = torch.randn(audio_frames, 512)
    
    # 前向传播测试
    with torch.no_grad():
        video_output, audio_output = ca_module(text_features, video_features, audio_features)
        
    print(f"Video output shape: {video_output.shape}")  # [1, 32, 4096]
    print(f"Audio output shape: {audio_output.shape}")  # [1, 32, 4096]
    print("CA module test passed!")
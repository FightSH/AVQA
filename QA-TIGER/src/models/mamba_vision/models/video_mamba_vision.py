#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class MultiModalPreprocessor(nn.Module):
    """多模态特征预处理器"""
    
    def __init__(self, img_dim, audio_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.img_dim = img_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # 投影层，将不同模态映射到相同维度
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
        
        # 融合后的归一化
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 模态嵌入，帮助模型区分不同模态
        self.img_modal_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.audio_modal_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, img_feat, audio_feat):
        """
        Args:
            img_feat: (B, T, img_dim) 图像特征序列
            audio_feat: (B, T, audio_dim) 音频特征序列
        Returns:
            combined: (B, T, hidden_dim * 2) 融合后的特征
        """
        B, T = img_feat.shape[:2]
        
        # 投影到相同维度
        img_proj = self.img_proj(img_feat)      # (B, T, hidden_dim)
        audio_proj = self.audio_proj(audio_feat) # (B, T, hidden_dim)
        # print(f"img_proj shape: {img_proj.shape}")
        
        # 添加模态嵌入
        img_proj = img_proj + self.img_modal_embed.expand(B, T, -1)
        # print(f"img_proj shape: {img_proj.shape}")
        audio_proj = audio_proj + self.audio_modal_embed.expand(B, T, -1)
        
        # 拼接并归一化
        combined = torch.cat([img_proj, audio_proj], dim=-1)  # (B, T, hidden_dim * 2)
        # print(f"combined shape: {combined.shape}")
        combined = self.fusion_norm(combined)
        
        return combined


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
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D) with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


class SimplifiedTemporalBlock(nn.Module):
    """简化的时序块，不使用卷积"""
    
    def __init__(self, dim, drop_path=0., layer_scale=None):
        super().__init__()
        
        # 使用MLP代替卷积
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Layer scale
        self.layer_scale = layer_scale
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))
            self.use_layer_scale = True
        else:
            self.use_layer_scale = False
            
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D)
        """
        # 第一个MLP块
        if self.use_layer_scale:
            x = x + self.drop_path(self.gamma1 * self.mlp1(x))
        else:
            x = x + self.drop_path(self.mlp1(x))
        
        # 第二个MLP块
        if self.use_layer_scale:
            x = x + self.drop_path(self.gamma2 * self.mlp2(x))
        else:
            x = x + self.drop_path(self.mlp2(x))
        
        return x


class VideoMambaVisionMixer(nn.Module):
    """视频版本的 Mamba 混合器"""
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # 状态空间参数
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, 
            bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        
        # 初始化 dt_proj
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
            
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * 
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # A 矩阵
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D 参数
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # 移除1D卷积层，简化架构

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, D)
        Returns:
            output: (B, T, D)
        """
        batch_size, seq_len, _ = hidden_states.shape
        # print(f"VideoMambaVisionMixer input: {hidden_states.shape}, batch_size={batch_size}, seq_len={seq_len}")
        
        # 输入投影和分割
        xz = self.in_proj(hidden_states)  # (B, T, d_inner)
        # print(f"After in_proj: {xz.shape}")
        xz = rearrange(xz, "b l d -> b d l")  # (B, d_inner, T)
        # print(f"After rearrange to (b d l): {xz.shape}")
        x, z = xz.chunk(2, dim=1)  # 各自 (B, d_inner//2, T)
        # print(f"After chunk - x: {x.shape}, z: {z.shape}")
        
        # A 矩阵
        A = -torch.exp(self.A_log.float())
        
        # 移除1D卷积，直接使用激活函数
        x = F.silu(x)
        z = F.silu(z)
        
        # 状态空间计算
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (B*T, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seq_len)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
        
        # 选择性扫描
        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(), 
            z=None, delta_bias=self.dt_proj.bias.float(), 
            delta_softplus=True, return_last_state=None
        )
        
        # 合并和输出投影
        y = torch.cat([y, z], dim=1)  # (B, d_inner, T)
        y = rearrange(y, "b d l -> b l d")  # (B, T, d_inner)
        output = self.out_proj(y)  # (B, T, D)
        
        return output


class TemporalAttention(nn.Module):
    """时序注意力机制"""
    
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        causal=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D)
        """
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 各自 (B, num_heads, T, head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        # 计算注意力
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, T, T)
        
        # 因果掩码（如果需要）
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn.masked_fill_(mask, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class VideoMambaBlock(nn.Module):
    """视频 Mamba 块"""
    
    def __init__(
        self, 
        dim, 
        mixer_type="mamba",  # "mamba" or "attention"
        num_heads=8,
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_norm=False,
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        layer_scale=None,
        causal=False,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        
        # 选择混合器类型
        if mixer_type == "mamba":
            self.mixer = VideoMambaVisionMixer(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2
            )
        elif mixer_type == "attention":
            self.mixer = TemporalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=drop,
                causal=causal,
            )
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=nn.GELU, 
            drop=drop
        )
        
        # Layer scale
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            x: (B, T, D)
        """
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VideoMambaVision(nn.Module):
    """视频理解的 MambaVision 模型"""
    
    def __init__(
        self,
        img_dim=2048,           # 图像特征维度
        audio_dim=128,          # 音频特征维度
        hidden_dim=256,         # 隐藏层维度
        depths=[2, 2, 6, 2],    # 各阶段的层数
        num_heads=[4, 8, 16, 32], # 各阶段的注意力头数
        mlp_ratio=4.,
        num_classes=1000,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        layer_scale=1e-6,
        causal=False,           # 是否使用因果注意力
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 多模态预处理
        self.multimodal_prep = MultiModalPreprocessor(
            img_dim=img_dim,
            audio_dim=audio_dim, 
            hidden_dim=hidden_dim,
            dropout=drop_rate
        )
        
        # 位置编码
        self.pos_encoding = TemporalPositionalEncoding(hidden_dim * 2)
        
        # 构建各阶段
        feature_dim = hidden_dim * 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        
        for stage_idx, depth in enumerate(depths):
            stage_blocks = nn.ModuleList()
            
            for block_idx in range(depth):
                global_block_idx = sum(depths[:stage_idx]) + block_idx
                
                # 所有阶段都使用 Mamba + Attention，移除卷积
                # 前半部分层使用Mamba，后半部分层使用注意力
                mixer_type = "attention" if block_idx >= depth // 2 else "mamba"
                
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
    
    def forward_features(self, img_seq, audio_seq):
        """
        Args:
            img_seq: (B, T, img_dim) 图像特征序列
            audio_seq: (B, T, audio_dim) 音频特征序列
        Returns:
            x: (B, feature_dim) 全局特征
        """
        # 多模态融合
        x = self.multimodal_prep(img_seq, audio_seq)  # (B, T, hidden_dim * 2)
        # print(f"x shape: {x.shape}")
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
    
    def forward_temporal_features(self, img_seq, audio_seq):
        """
        Args:
            img_seq: (B, T, img_dim) 图像特征序列
            audio_seq: (B, T, audio_dim) 音频特征序列
        Returns:
            x: (B, T, feature_dim) 时序特征，保留时间维度
        """
        # 多模态融合
        x = self.multimodal_prep(img_seq, audio_seq)  # (B, T, hidden_dim * 2)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 通过各阶段
        for stage in self.stages:
            for block in stage:
                x = block(x)
        
        # 最终归一化，保留时序维度
        x = self.norm(x)  # (B, T, hidden_dim * 2)
        
        return x
    
    def forward(self, img_seq, audio_seq):
        """
        Args:
            img_seq: (B, T, img_dim) 图像特征序列
            audio_seq: (B, T, audio_dim) 音频特征序列
        Returns:
            logits: (B, num_classes) 分类结果
        """
        x = self.forward_features(img_seq, audio_seq)
        x = self.head(x)
        return x


# 便捷的模型构建函数
def video_mamba_vision_tiny(**kwargs):
    """构建 Tiny 版本的视频 MambaVision"""
    model = VideoMambaVision(
        hidden_dim=128,
        depths=[1, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        **kwargs
    )
    return model


def video_mamba_vision_small(**kwargs):
    """构建 Small 版本的视频 MambaVision"""
    model = VideoMambaVision(
        hidden_dim=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        **kwargs
    )
    return model


def video_mamba_vision_base(**kwargs):
    """构建 Base 版本的视频 MambaVision"""
    model = VideoMambaVision(
        hidden_dim=256,
        depths=[2, 2, 8, 2],
        num_heads=[4, 8, 16, 32],
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = video_mamba_vision_base(
        img_dim=2048,      # ResNet 特征
        audio_dim=128,     # 音频特征
        num_classes=400,   # Kinetics-400
        causal=False       # 非因果，可以看到未来帧
    ).to(device)
    
    # 测试数据
    batch_size = 2
    seq_len = 30  # 30帧
    
    img_features = torch.randn(batch_size, seq_len, 2048).to(device)
    audio_features = torch.randn(batch_size, seq_len, 128).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(img_features, audio_features)
        print(f"输入图像特征: {img_features.shape}")
        print(f"输入音频特征: {audio_features.shape}")
        print(f"输出分类结果: {output.shape}")
        
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
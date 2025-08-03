#!/usr/bin/env python3

"""
VideoMamba配置文件
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class VideoMambaConfig:
    """VideoMamba模型配置"""
    
    # 基础配置
    mamba_hidden_dim: int = 256
    depths: List[int] = None
    num_heads: List[int] = None
    mlp_ratio: float = 4.0
    
    # Dropout配置
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    
    # 其他配置
    layer_scale: Optional[float] = 1e-6
    causal: bool = False
    
    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 6, 2]
        if self.num_heads is None:
            self.num_heads = [4, 8, 16, 32]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'mamba_hidden_dim': self.mamba_hidden_dim,
            'depths': self.depths,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate,
            'layer_scale': self.layer_scale,
            'causal': self.causal,
        }


# 预定义配置
MAMBA_CONFIGS = {
    'tiny': VideoMambaConfig(
        mamba_hidden_dim=128,
        depths=[1, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.05,
    ),
    
    'small': VideoMambaConfig(
        mamba_hidden_dim=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.1,
    ),
    
    'base': VideoMambaConfig(
        mamba_hidden_dim=256,
        depths=[2, 2, 8, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.1,
    ),
    
    'large': VideoMambaConfig(
        mamba_hidden_dim=384,
        depths=[2, 4, 12, 2],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.15,
    ),
}


def get_mamba_config(config_name: str = 'base') -> VideoMambaConfig:
    """获取预定义的VideoMamba配置"""
    if config_name not in MAMBA_CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(MAMBA_CONFIGS.keys())}")
    
    return MAMBA_CONFIGS[config_name]


def create_custom_mamba_config(**kwargs) -> VideoMambaConfig:
    """创建自定义VideoMamba配置"""
    return VideoMambaConfig(**kwargs)


# QA-TIGER集成配置
@dataclass
class QATigerMambaConfig:
    """QA-TIGER + VideoMamba集成配置"""
    
    # QA-TIGER基础配置
    d_model: int = 512
    video_dim: int = 512
    audio_dim: int = 128
    patch_dim: int = 768
    
    # TempMoE配置
    topK: int = 3
    num_experts: int = 10
    
    # 编码器配置
    encoder_type: str = 'ViT-L/14@336px'
    
    # VideoMamba配置
    use_video_mamba: bool = True
    mamba_config_name: str = 'base'
    mamba_config: Optional[VideoMambaConfig] = None
    
    # 其他模块配置
    use_ams: bool = False
    use_mamba: bool = False
    mccd: Optional[Dict] = None
    
    def __post_init__(self):
        if self.use_video_mamba and self.mamba_config is None:
            self.mamba_config = get_mamba_config(self.mamba_config_name)
    
    def to_qa_tiger_kwargs(self) -> Dict[str, Any]:
        """转换为QA_TIGER构造函数的参数"""
        kwargs = {
            'd_model': self.d_model,
            'video_dim': self.video_dim,
            'audio_dim': self.audio_dim,
            'patch_dim': self.patch_dim,
            'topK': self.topK,
            'num_experts': self.num_experts,
            'encoder_type': self.encoder_type,
            'use_video_mamba': self.use_video_mamba,
            'use_ams': self.use_ams,
            'use_mamba': self.use_mamba,
            'mccd': self.mccd,
        }
        
        if self.use_video_mamba and self.mamba_config is not None:
            kwargs['mamba_config'] = self.mamba_config.to_dict()
        
        return kwargs


# 预定义的完整配置
COMPLETE_CONFIGS = {
    'qa_tiger_mamba_tiny': QATigerMambaConfig(
        d_model=384,
        mamba_config_name='tiny',
    ),
    
    'qa_tiger_mamba_small': QATigerMambaConfig(
        d_model=512,
        mamba_config_name='small',
    ),
    
    'qa_tiger_mamba_base': QATigerMambaConfig(
        d_model=512,
        mamba_config_name='base',
    ),
    
    'qa_tiger_mamba_large': QATigerMambaConfig(
        d_model=768,
        mamba_config_name='large',
        topK=4,
        num_experts=12,
    ),
    
    # 不使用VideoMamba的基线配置
    'qa_tiger_baseline': QATigerMambaConfig(
        use_video_mamba=False,
    ),
}


def get_complete_config(config_name: str = 'qa_tiger_mamba_base') -> QATigerMambaConfig:
    """获取完整的配置"""
    if config_name not in COMPLETE_CONFIGS:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(COMPLETE_CONFIGS.keys())}")
    
    return COMPLETE_CONFIGS[config_name]


def print_config_summary(config: QATigerMambaConfig):
    """打印配置摘要"""
    print("=" * 60)
    print("QA-TIGER + VideoMamba 配置摘要")
    print("=" * 60)
    
    print(f"模型维度: {config.d_model}")
    print(f"视频维度: {config.video_dim}")
    print(f"音频维度: {config.audio_dim}")
    print(f"Patch维度: {config.patch_dim}")
    print(f"编码器类型: {config.encoder_type}")
    print(f"TempMoE - topK: {config.topK}, 专家数: {config.num_experts}")
    
    print(f"\nVideoMamba: {'启用' if config.use_video_mamba else '禁用'}")
    if config.use_video_mamba and config.mamba_config:
        mamba = config.mamba_config
        print(f"  - 隐藏维度: {mamba.mamba_hidden_dim}")
        print(f"  - 层数配置: {mamba.depths}")
        print(f"  - 注意力头数: {mamba.num_heads}")
        print(f"  - DropPath率: {mamba.drop_path_rate}")
        print(f"  - 因果模式: {mamba.causal}")
    
    print(f"\n其他模块:")
    print(f"  - AMS: {'启用' if config.use_ams else '禁用'}")
    print(f"  - 传统Mamba: {'启用' if config.use_mamba else '禁用'}")
    print(f"  - MCCD: {'启用' if config.mccd else '禁用'}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 展示所有预定义配置
    print("可用的VideoMamba配置:")
    for name in MAMBA_CONFIGS.keys():
        config = get_mamba_config(name)
        print(f"\n{name}:")
        print(f"  隐藏维度: {config.mamba_hidden_dim}")
        print(f"  层数: {config.depths}")
        print(f"  注意力头数: {config.num_heads}")
    
    print("\n" + "=" * 80)
    print("可用的完整配置:")
    for name in COMPLETE_CONFIGS.keys():
        print(f"- {name}")
    
    print("\n" + "=" * 80)
    print("示例配置详情:")
    config = get_complete_config('qa_tiger_mamba_base')
    print_config_summary(config)
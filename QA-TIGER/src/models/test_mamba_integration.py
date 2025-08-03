#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.net import QA_TIGER


def test_mamba_integration():
    """测试VideoMamba集成到QA-TIGER的功能"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建带VideoMamba的模型
    mamba_config = {
        'mamba_hidden_dim': 256,
        'depths': [2, 2, 4, 2],
        'num_heads': [4, 8, 16, 32],
        'drop_path_rate': 0.1,
        'layer_scale': 1e-6,
    }
    
    model = QA_TIGER(
        d_model=512,
        video_dim=512,
        audio_dim=128,
        patch_dim=768,
        use_video_mamba=True,
        mamba_config=mamba_config,
        encoder_type='ViT-L/14@336px',
        num_classes=42
    ).to(device)
    
    print("模型创建成功!")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 30
    num_patches = 10
    
    # 模拟输入数据
    reshaped_data = {
        'quest': torch.randint(0, 1000, (batch_size, 77)).to(device),  # 问题token IDs
        'audio': torch.randn(batch_size, seq_len, 128).to(device),     # 音频特征
        'video': torch.randn(batch_size, seq_len, 512).to(device),     # 视频特征
        'patch': torch.randn(batch_size, seq_len, num_patches, 768).to(device)  # Patch特征
    }
    
    print("测试数据创建成功!")
    print(f"问题形状: {reshaped_data['quest'].shape}")
    print(f"音频形状: {reshaped_data['audio'].shape}")
    print(f"视频形状: {reshaped_data['video'].shape}")
    print(f"Patch形状: {reshaped_data['patch'].shape}")
    
    # 前向传播测试
    try:
        with torch.no_grad():
            output = model(reshaped_data)
            
        print("\n前向传播成功!")
        print(f"输出形状: {output['out'].shape}")
        print(f"融合logits形状: {output['fusion_logits'].shape}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 测试不使用VideoMamba的版本进行对比
        print("\n创建不使用VideoMamba的对比模型...")
        model_no_mamba = QA_TIGER(
            d_model=512,
            video_dim=512,
            audio_dim=128,
            patch_dim=768,
            use_video_mamba=False,
            encoder_type='ViT-L/14@336px',
            num_classes=42
        ).to(device)
        
        with torch.no_grad():
            output_no_mamba = model_no_mamba(reshaped_data)
            
        no_mamba_params = sum(p.numel() for p in model_no_mamba.parameters())
        
        print(f"不使用VideoMamba的参数量: {no_mamba_params:,}")
        print(f"VideoMamba增加的参数量: {total_params - no_mamba_params:,}")
        
        print("\n✅ VideoMamba集成测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_video_mamba_adapter_only():
    """单独测试VideoMambaAdapter"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from src.models.mamba_vision.video_mamba_integration import VideoMambaAdapter
        
        adapter = VideoMambaAdapter(
            d_model=512,
            video_dim=512,
            audio_dim=128,
            mamba_hidden_dim=256
        ).to(device)
        
        # 测试数据
        batch_size = 2
        seq_len = 30
        
        video_seq = torch.randn(batch_size, seq_len, 512).to(device)
        audio_seq = torch.randn(batch_size, seq_len, 128).to(device)
        
        with torch.no_grad():
            enhanced_features = adapter(video_seq, audio_seq)
            
        print(f"VideoMambaAdapter测试成功!")
        print(f"输入视频: {video_seq.shape}")
        print(f"输入音频: {audio_seq.shape}")
        print(f"输出增强特征: {enhanced_features.shape}")
        
        adapter_params = sum(p.numel() for p in adapter.parameters())
        print(f"VideoMambaAdapter参数量: {adapter_params:,}")
        
    except Exception as e:
        print(f"VideoMambaAdapter测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("测试VideoMamba集成到QA-TIGER")
    print("=" * 60)
    
    # 首先测试VideoMambaAdapter
    print("\n1. 测试VideoMambaAdapter...")
    test_video_mamba_adapter_only()
    
    print("\n" + "=" * 60)
    
    # 然后测试完整集成
    print("\n2. 测试完整QA-TIGER集成...")
    test_mamba_integration()
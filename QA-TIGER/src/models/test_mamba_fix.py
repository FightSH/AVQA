#!/usr/bin/env python3

"""
测试VideoMamba集成修复
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_video_mamba_creation():
    """测试VideoMamba模型创建"""
    
    try:
        from src.models.mamba_vision.video_mamba_integration import VideoMambaAdapter
        
        print("测试VideoMambaAdapter创建...")
        
        # 基本参数
        adapter = VideoMambaAdapter(
            d_model=512,
            video_dim=768,  # 匹配配置文件
            audio_dim=128,  # 匹配配置文件
            mamba_hidden_dim=256,
        )
        
        print("✅ VideoMambaAdapter创建成功")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 30
        
        video_seq = torch.randn(batch_size, seq_len, 768)
        audio_seq = torch.randn(batch_size, seq_len, 128)
        
        with torch.no_grad():
            output = adapter(video_seq, audio_seq)
            
        print(f"✅ 前向传播成功，输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ VideoMambaAdapter测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_qa_tiger_with_mamba():
    """测试QA_TIGER与VideoMamba集成"""
    
    try:
        from src.models.net import QA_TIGER
        
        print("\n测试QA_TIGER + VideoMamba创建...")
        
        # 模拟配置文件中的参数
        model_args = {
            'd_model': 512,
            'video_dim': 768,
            'patch_dim': 1024,
            'audio_dim': 128,
            'topK': 7,
            'num_experts': 7,
            'encoder_type': 'openai/clip-vit-large-patch14',
            'use_video_mamba': True,
            'mamba_config': {
                'mamba_hidden_dim': 256,
                'depths': [2, 2, 6, 2],
                'num_heads': [4, 8, 16, 32],
                'drop_path_rate': 0.1,
            }
        }
        
        model = QA_TIGER(**model_args)
        print("✅ QA_TIGER + VideoMamba创建成功")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 30
        num_patches = 14
        
        reshaped_data = {
            'quest': torch.randint(0, 1000, (batch_size, 77)),
            'audio': torch.randn(batch_size, seq_len, 128),
            'video': torch.randn(batch_size, seq_len, 768),
            'patch': torch.randn(batch_size, seq_len, num_patches, 1024)
        }
        
        with torch.no_grad():
            output = model(reshaped_data)
            
        print(f"✅ 前向传播成功，输出形状: {output['out'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ QA_TIGER + VideoMamba测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_without_mamba():
    """测试不使用VideoMamba的情况"""
    
    try:
        from src.models.net import QA_TIGER
        
        print("\n测试QA_TIGER (不使用VideoMamba)...")
        
        model_args = {
            'd_model': 512,
            'video_dim': 768,
            'patch_dim': 1024,
            'audio_dim': 128,
            'topK': 7,
            'num_experts': 7,
            'encoder_type': 'openai/clip-vit-large-patch14',
            'use_video_mamba': False,  # 关键：不使用VideoMamba
        }
        
        model = QA_TIGER(**model_args)
        print("✅ QA_TIGER (无VideoMamba) 创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ QA_TIGER (无VideoMamba) 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("VideoMamba集成修复测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # 测试1: VideoMambaAdapter
    if test_video_mamba_creation():
        success_count += 1
    
    # 测试2: QA_TIGER + VideoMamba
    if test_qa_tiger_with_mamba():
        success_count += 1
    
    # 测试3: QA_TIGER 无VideoMamba
    if test_without_mamba():
        success_count += 1
    
    print(f"\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！VideoMamba集成修复成功！")
    else:
        print("⚠️  部分测试失败，需要进一步调试")
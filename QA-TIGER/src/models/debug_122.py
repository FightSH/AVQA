#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_122_issue():
    """测试122问题的来源"""
    
    from src.models.mamba_vision.video_mamba_integration import VideoMambaAdapter
    
    # 创建适配器
    adapter = VideoMambaAdapter(
        d_model=512,
        video_dim=512,  # 注意：现在是512而不是768
        audio_dim=512,  # 注意：现在是512而不是128
        mamba_hidden_dim=256,
    )
    
    print("=" * 60)
    print("测试不同的batch size和seq_len组合")
    print("=" * 60)
    
    test_cases = [
        (1, 60),   # 单个样本
        (2, 60),   # 正常情况
        (3, 60),   # 奇数batch
        (2, 61),   # 奇数seq_len
        (2, 59),   # 减少seq_len
    ]
    
    for batch_size, seq_len in test_cases:
        print(f"\n测试 batch_size={batch_size}, seq_len={seq_len}")
        print(f"期望的 (b*l) = {batch_size * seq_len}")
        
        try:
            video_seq = torch.randn(batch_size, seq_len, 512)
            audio_seq = torch.randn(batch_size, seq_len, 512)
            
            with torch.no_grad():
                output = adapter(video_seq, audio_seq)
                
            print(f"✅ 成功！输出形状: {output.shape}")
            
        except Exception as e:
            print(f"❌ 失败: {str(e)}")
            if "122" in str(e):
                print("🔍 发现122错误！")
                
                # 分析122的可能来源
                print(f"122 / {seq_len} = {122 / seq_len}")
                print(f"122 - {batch_size * seq_len} = {122 - batch_size * seq_len}")


def test_specific_122_case():
    """测试特定的122情况"""
    
    print("\n" + "=" * 60)
    print("测试122的具体情况")
    print("=" * 60)
    
    # 如果122 = 2 * 61，那么可能是seq_len=61
    # 如果122 = 2 * 60 + 2，那么可能是有额外的元素
    
    possible_cases = [
        (2, 61),    # 122 = 2 * 61
        (1, 122),   # 122 = 1 * 122
        (122, 1),   # 122 = 122 * 1
    ]
    
    for batch_size, seq_len in possible_cases:
        print(f"\n尝试 batch_size={batch_size}, seq_len={seq_len} (总计={batch_size*seq_len})")
        
        # 检查这种组合是否会产生122
        if batch_size * seq_len == 122:
            print("🎯 这个组合会产生122！")


def analyze_122_factors():
    """分析122的因数分解"""
    
    print("\n" + "=" * 60)
    print("122的因数分解分析")
    print("=" * 60)
    
    print(f"122 = {122}")
    
    factors = []
    for i in range(1, 123):
        if 122 % i == 0:
            factors.append((i, 122 // i))
    
    print("所有可能的 (batch_size, seq_len) 组合:")
    for b, s in factors:
        print(f"  {b} × {s} = 122")
    
    # 检查哪些组合接近我们期望的值
    print("\n接近期望值的组合:")
    expected_batch = 2
    expected_seq = 60
    expected_total = expected_batch * expected_seq  # 120
    
    print(f"期望: {expected_batch} × {expected_seq} = {expected_total}")
    print(f"实际: ? × ? = 122")
    print(f"差异: 122 - 120 = 2")
    
    print("\n可能的解释:")
    print("1. seq_len 从 60 变成了 61 (padding?)")
    print("2. batch_size 有问题")
    print("3. 数据在某处被错误地reshape")


if __name__ == "__main__":
    # 分析122的数学特性
    analyze_122_factors()
    
    # 测试特定情况
    test_specific_122_case()
    
    # 测试不同组合
    test_122_issue()
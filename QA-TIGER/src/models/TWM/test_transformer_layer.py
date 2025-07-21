#!/usr/bin/env python3

import torch
import sys
import os

# 添加src路径
sys.path.append('src')

from audio_layers import Transformer_Layer

def test_transformer_layer():
    # 测试参数
    device = 'cpu'
    d_model = 512
    d_ff = 2048
    patch_nums = 6
    patch_size = 10
    dynamic = True
    factorized = True
    layer_number = 1
    batch_norm = True
    
    # 创建模型
    transformer_layer = Transformer_Layer(
        device=device,
        d_model=d_model,
        d_ff=d_ff,
        patch_nums=patch_nums,
        patch_size=patch_size,
        dynamic=dynamic,
        factorized=factorized,
        layer_number=layer_number,
        batch_norm=batch_norm
    )
    
    # 测试输入
    batch_size = 5
    temporal_length = patch_nums * patch_size  # 60
    query_length = 10
    
    # 输入数据 (去掉num_nodes维度)
    x = torch.randn(batch_size, temporal_length, d_model)  # [5, 60, 512]
    query = torch.randn(batch_size, query_length, 1536)    # [5, 10, 1536]
    
    print(f"Input x shape: {x.shape}")
    print(f"Input query shape: {query.shape}")
    
    try:
        # 前向传播
        output, attention = transformer_layer(x, query)
        
        print(f"Output shape: {output.shape}")
        print(f"Attention shape: {attention.shape}")
        print("✅ Test passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformer_layer()
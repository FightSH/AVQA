#!/usr/bin/env python3

"""
测试VideoMamba聚合器的集成
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.net import QA_TIGER


def test_mamba_aggregator_integration():
    """测试VideoMamba聚合器集成到QA-TIGER"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试配置
    test_configs = [
        {
            'name': '传统TempMoE聚合器',
            'config': {
                'd_model': 512,
                'video_dim': 768,
                'audio_dim': 128,
                'patch_dim': 1024,
                'topK': 7,
                'num_experts': 7,
                'encoder_type': 'openai/clip-vit-large-patch14',
                'use_mamba_aggregator': False,  # 使用传统聚合器
            }
        },
        {
            'name': 'VideoMamba聚合器',
            'config': {
                'd_model': 512,
                'video_dim': 768,
                'audio_dim': 128,
                'patch_dim': 1024,
                'topK': 7,
                'num_experts': 7,
                'encoder_type': 'openai/clip-vit-large-patch14',
                'use_mamba_aggregator': True,  # 使用VideoMamba聚合器
                'mamba_aggregator_config': {
                    'mamba_hidden_dim': 256,
                    'depths': [1, 1, 2, 1],
                    'num_heads': [4, 8, 16, 32],
                    'question_fusion': 'concat',
                    'dropout': 0.1,
                }
            }
        },
        {
            'name': 'VideoMamba聚合器 (交叉注意力融合)',
            'config': {
                'd_model': 512,
                'video_dim': 768,
                'audio_dim': 128,
                'patch_dim': 1024,
                'topK': 7,
                'num_experts': 7,
                'encoder_type': 'openai/clip-vit-large-patch14',
                'use_mamba_aggregator': True,
                'mamba_aggregator_config': {
                    'mamba_hidden_dim': 256,
                    'depths': [1, 1, 2, 1],
                    'num_heads': [4, 8, 16, 32],
                    'question_fusion': 'cross_attn',  # 使用交叉注意力
                    'dropout': 0.1,
                }
            }
        }
    ]
    
    # 准备测试数据
    batch_size = 2
    seq_len = 30
    num_patches = 14
    
    reshaped_data = {
        'quest': torch.randint(0, 1000, (batch_size, 77)).to(device),
        'audio': torch.randn(batch_size, seq_len, 128).to(device),
        'video': torch.randn(batch_size, seq_len, 768).to(device),
        'patch': torch.randn(batch_size, seq_len, num_patches, 1024).to(device)
    }
    
    print("=" * 80)
    print("VideoMamba聚合器集成测试")
    print("=" * 80)
    
    results = []
    
    for config_info in test_configs:
        config_name = config_info['name']
        model_config = config_info['config']
        
        print(f"\n测试配置: {config_name}")
        print("-" * 60)
        
        try:
            # 创建模型
            model = QA_TIGER(**model_config).to(device)
            model.eval()
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"总参数量: {total_params:,}")
            print(f"可训练参数量: {trainable_params:,}")
            
            # 前向传播测试
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                
                if device.type == 'cuda':
                    start_time.record()
                
                outputs = model(reshaped_data)
                
                if device.type == 'cuda':
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                else:
                    inference_time = 0
            
            print(f"输出形状: {outputs['out'].shape}")
            print(f"推理时间: {inference_time:.2f}ms" if device.type == 'cuda' else "推理时间: N/A (CPU)")
            print("✅ 测试通过")
            
            results.append({
                'name': config_name,
                'params': total_params,
                'time': inference_time if device.type == 'cuda' else 0,
                'success': True
            })
            
            # 清理内存
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            results.append({
                'name': config_name,
                'params': 0,
                'time': 0,
                'success': False,
                'error': str(e)
            })
            import traceback
            traceback.print_exc()
    
    # 结果对比
    print("\n" + "=" * 80)
    print("测试结果对比")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        baseline = successful_results[0]  # 传统TempMoE作为基线
        
        print(f"{'配置名称':<30} {'参数量':<15} {'推理时间':<12} {'参数变化':<12}")
        print("-" * 80)
        
        for result in successful_results:
            param_change = ""
            if result != baseline:
                param_diff = result['params'] - baseline['params']
                param_change = f"{param_diff:+,}"
            
            time_str = f"{result['time']:.2f}ms" if device.type == 'cuda' else "N/A"
            
            print(f"{result['name']:<30} {result['params']:>12,} {time_str:<12} {param_change:<12}")
    
    # 失败的测试
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n失败的测试:")
        for result in failed_results:
            print(f"- {result['name']}: {result['error']}")
    
    print(f"\n总结: {len(successful_results)}/{len(results)} 个配置测试成功")


def test_different_fusion_methods():
    """测试不同的问题融合方法"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fusion_methods = ['concat', 'cross_attn', 'add']
    
    print("\n" + "=" * 80)
    print("测试不同问题融合方法")
    print("=" * 80)
    
    for fusion_method in fusion_methods:
        print(f"\n测试融合方法: {fusion_method}")
        print("-" * 40)
        
        try:
            config = {
                'd_model': 512,
                'video_dim': 768,
                'audio_dim': 128,
                'patch_dim': 1024,
                'encoder_type': 'openai/clip-vit-large-patch14',
                'use_mamba_aggregator': True,
                'mamba_aggregator_config': {
                    'mamba_hidden_dim': 256,
                    'depths': [1, 1, 1, 1],  # 更小的配置用于快速测试
                    'num_heads': [4, 8, 16, 32],
                    'question_fusion': fusion_method,
                    'dropout': 0.1,
                }
            }
            
            model = QA_TIGER(**config).to(device)
            
            # 测试数据
            reshaped_data = {
                'quest': torch.randint(0, 1000, (1, 77)).to(device),
                'audio': torch.randn(1, 30, 128).to(device),
                'video': torch.randn(1, 30, 768).to(device),
                'patch': torch.randn(1, 30, 14, 1024).to(device)
            }
            
            with torch.no_grad():
                outputs = model(reshaped_data)
            
            print(f"✅ {fusion_method} 融合方法测试成功")
            print(f"   输出形状: {outputs['out'].shape}")
            
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ {fusion_method} 融合方法测试失败: {str(e)}")


if __name__ == "__main__":
    print("VideoMamba聚合器集成测试")
    
    # 主要集成测试
    test_mamba_aggregator_integration()
    
    # 融合方法测试
    test_different_fusion_methods()
    
    print("\n🎉 所有测试完成！")
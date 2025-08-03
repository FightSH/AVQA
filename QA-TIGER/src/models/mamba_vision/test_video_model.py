#!/usr/bin/env python3

import torch
import torch.nn as nn
import time
import traceback
from models.video_mamba_vision import (
    video_mamba_vision_tiny, 
    video_mamba_vision_small, 
    video_mamba_vision_base,
    VideoMambaVision
)
from .video_config import get_config


def test_model_creation():
    """测试模型创建"""
    print("🔧 测试模型创建...")
    
    try:
        # 测试预定义模型
        models = {
            'Tiny': video_mamba_vision_tiny(img_dim=512, audio_dim=64, num_classes=10),
            'Small': video_mamba_vision_small(img_dim=1024, audio_dim=128, num_classes=100),
            'Base': video_mamba_vision_base(img_dim=2048, audio_dim=128, num_classes=400),
        }
        
        for name, model in models.items():
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {name} 模型创建成功，参数量: {param_count:,}")
            
        # 测试自定义模型
        custom_model = VideoMambaVision(
            img_dim=1536,
            audio_dim=256,
            hidden_dim=320,
            depths=[2, 3, 8, 3],
            num_heads=[5, 10, 20, 40],
            num_classes=1000,
            causal=True
        )
        param_count = sum(p.numel() for p in custom_model.parameters())
        print(f"  ✅ 自定义模型创建成功，参数量: {param_count:,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n🚀 测试前向传播...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")
    
    test_cases = [
        {
            'name': 'Small Sequence',
            'batch_size': 2,
            'seq_len': 8,
            'img_dim': 512,
            'audio_dim': 64,
            'num_classes': 10
        },
        {
            'name': 'Medium Sequence', 
            'batch_size': 4,
            'seq_len': 30,
            'img_dim': 2048,
            'audio_dim': 128,
            'num_classes': 400
        },
        {
            'name': 'Long Sequence',
            'batch_size': 1,
            'seq_len': 64,
            'img_dim': 1024,
            'audio_dim': 256,
            'num_classes': 1000
        }
    ]
    
    success_count = 0
    
    for case in test_cases:
        try:
            print(f"\n  测试案例: {case['name']}")
            
            # 创建模型
            model = video_mamba_vision_small(
                img_dim=case['img_dim'],
                audio_dim=case['audio_dim'],
                num_classes=case['num_classes']
            ).to(device)
            
            # 创建测试数据
            img_features = torch.randn(
                case['batch_size'], case['seq_len'], case['img_dim']
            ).to(device)
            audio_features = torch.randn(
                case['batch_size'], case['seq_len'], case['audio_dim']
            ).to(device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output = model(img_features, audio_features)
                end_time = time.time()
            
            # 验证输出
            expected_shape = (case['batch_size'], case['num_classes'])
            assert output.shape == expected_shape, f"输出形状错误: {output.shape} vs {expected_shape}"
            
            # 检查输出是否包含 NaN 或 Inf
            assert not torch.isnan(output).any(), "输出包含 NaN"
            assert not torch.isinf(output).any(), "输出包含 Inf"
            
            inference_time = (end_time - start_time) * 1000  # ms
            print(f"    ✅ 输入: img{img_features.shape}, audio{audio_features.shape}")
            print(f"    ✅ 输出: {output.shape}")
            print(f"    ✅ 推理时间: {inference_time:.2f} ms")
            
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ 测试失败: {e}")
            traceback.print_exc()
    
    print(f"\n  前向传播测试结果: {success_count}/{len(test_cases)} 通过")
    return success_count == len(test_cases)


def test_gradient_flow():
    """测试梯度流"""
    print("\n🔄 测试梯度流...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型和数据
        model = video_mamba_vision_small(
            img_dim=512, audio_dim=128, num_classes=10
        ).to(device)
        
        img_features = torch.randn(2, 16, 512, requires_grad=True).to(device)
        audio_features = torch.randn(2, 16, 128, requires_grad=True).to(device)
        targets = torch.randint(0, 10, (2,)).to(device)
        
        # 前向传播
        model.train()
        output = model(img_features, audio_features)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, targets)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm == 0:
                    print(f"    ⚠️  {name} 的梯度为零")
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        print(f"  ✅ 损失值: {loss.item():.4f}")
        print(f"  ✅ 平均梯度范数: {avg_grad_norm:.6f}")
        print(f"  ✅ 有梯度的参数数量: {len(grad_norms)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 梯度流测试失败: {e}")
        traceback.print_exc()
        return False


def test_different_configs():
    """测试不同配置"""
    print("\n⚙️  测试不同配置...")
    
    config_names = ['base', 'kinetics400', 'ucf101', 'realtime', 'emotion']
    success_count = 0
    
    for config_name in config_names:
        try:
            print(f"\n  测试配置: {config_name}")
            
            config = get_config(config_name)
            
            # 创建模型
            model = VideoMambaVision(
                img_dim=config['data']['img_dim'],
                audio_dim=config['data']['audio_dim'],
                num_classes=config['model']['num_classes'],
                **{k: v for k, v in config['model'].items() if k != 'num_classes'}
            )
            
            # 测试数据
            batch_size = 2
            seq_len = config['data']['seq_len']
            img_dim = config['data']['img_dim']
            audio_dim = config['data']['audio_dim']
            
            img_features = torch.randn(batch_size, seq_len, img_dim)
            audio_features = torch.randn(batch_size, seq_len, audio_dim)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(img_features, audio_features)
            
            expected_shape = (batch_size, config['model']['num_classes'])
            assert output.shape == expected_shape
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"    ✅ 参数量: {param_count:,}")
            print(f"    ✅ 输出形状: {output.shape}")
            
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ 配置 {config_name} 测试失败: {e}")
    
    print(f"\n  配置测试结果: {success_count}/{len(config_names)} 通过")
    return success_count == len(config_names)


def test_causal_vs_noncausal():
    """测试因果 vs 非因果模式"""
    print("\n🔀 测试因果 vs 非因果模式...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 测试数据
        batch_size, seq_len = 2, 20
        img_features = torch.randn(batch_size, seq_len, 512).to(device)
        audio_features = torch.randn(batch_size, seq_len, 128).to(device)
        
        # 非因果模型
        model_noncausal = video_mamba_vision_small(
            img_dim=512, audio_dim=128, num_classes=10, causal=False
        ).to(device)
        
        # 因果模型
        model_causal = video_mamba_vision_small(
            img_dim=512, audio_dim=128, num_classes=10, causal=True
        ).to(device)
        
        # 测试输出
        model_noncausal.eval()
        model_causal.eval()
        
        with torch.no_grad():
            output_noncausal = model_noncausal(img_features, audio_features)
            output_causal = model_causal(img_features, audio_features)
        
        print(f"  ✅ 非因果模型输出: {output_noncausal.shape}")
        print(f"  ✅ 因果模型输出: {output_causal.shape}")
        
        # 输出应该不同（因为注意力机制不同）
        diff = torch.abs(output_noncausal - output_causal).mean().item()
        print(f"  ✅ 输出差异: {diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 因果性测试失败: {e}")
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """测试内存效率"""
    print("\n💾 测试内存效率...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA 不可用，跳过内存测试")
        return True
    
    try:
        device = torch.device("cuda")
        
        # 测试不同序列长度的内存使用
        seq_lengths = [8, 16, 32, 64]
        memory_usage = []
        
        for seq_len in seq_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = video_mamba_vision_small(
                img_dim=1024, audio_dim=128, num_classes=100
            ).to(device)
            
            img_features = torch.randn(1, seq_len, 1024).to(device)
            audio_features = torch.randn(1, seq_len, 128).to(device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(img_features, audio_features)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_usage.append(memory_used)
            
            print(f"  序列长度 {seq_len:2d}: {memory_used:6.1f} MB")
        
        # 检查内存增长是否合理（应该接近线性）
        if len(memory_usage) >= 2:
            growth_rate = memory_usage[-1] / memory_usage[0]
            seq_growth_rate = seq_lengths[-1] / seq_lengths[0]
            efficiency = growth_rate / seq_growth_rate
            
            print(f"  ✅ 内存增长效率: {efficiency:.2f} (越接近1越好)")
            
            if efficiency < 2.0:  # 合理的阈值
                print("  ✅ 内存效率良好")
                return True
            else:
                print("  ⚠️  内存效率可能需要优化")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 内存效率测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 Video MambaVision 模型测试")
    print("=" * 60)
    
    tests = [
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("梯度流", test_gradient_flow),
        ("不同配置", test_different_configs),
        ("因果性", test_causal_vs_noncausal),
        ("内存效率", test_memory_efficiency),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎯 测试总结: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！模型可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
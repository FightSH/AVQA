#!/usr/bin/env python3

"""
VideoMamba增强QA-TIGER模型工厂
"""

import torch
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.net import QA_TIGER
from .config import get_complete_config, QATigerMambaConfig, print_config_summary


def create_qa_tiger_with_mamba(
    config_name: str = 'qa_tiger_mamba_base',
    custom_config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> QA_TIGER:
    """
    创建集成VideoMamba的QA-TIGER模型
    
    Args:
        config_name: 预定义配置名称
        custom_config: 自定义配置覆盖
        device: 设备类型 ('cuda', 'cpu', 或 None 自动选择)
        verbose: 是否打印详细信息
    
    Returns:
        QA_TIGER模型实例
    """
    
    # 获取配置
    config = get_complete_config(config_name)
    
    # 应用自定义配置覆盖
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"警告: 未知的配置参数 '{key}'")
    
    # 打印配置信息
    if verbose:
        print_config_summary(config)
    
    # 创建模型
    model_kwargs = config.to_qa_tiger_kwargs()
    model = QA_TIGER(**model_kwargs)
    
    # 移动到指定设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    if verbose:
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型创建完成!")
        print(f"设备: {device}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 估算显存使用
        param_memory = total_params * 4 / (1024**3)  # 假设float32
        print(f"估算参数显存: {param_memory:.2f} GB")
    
    return model


def create_baseline_qa_tiger(
    d_model: int = 512,
    video_dim: int = 512,
    audio_dim: int = 128,
    patch_dim: int = 768,
    device: Optional[str] = None,
    verbose: bool = True
) -> QA_TIGER:
    """
    创建不使用VideoMamba的基线QA-TIGER模型
    """
    
    custom_config = {
        'd_model': d_model,
        'video_dim': video_dim,
        'audio_dim': audio_dim,
        'patch_dim': patch_dim,
        'use_video_mamba': False,
    }
    
    return create_qa_tiger_with_mamba(
        config_name='qa_tiger_baseline',
        custom_config=custom_config,
        device=device,
        verbose=verbose
    )


def compare_model_sizes():
    """比较不同配置的模型大小"""
    
    configs_to_compare = [
        'qa_tiger_baseline',
        'qa_tiger_mamba_tiny',
        'qa_tiger_mamba_small', 
        'qa_tiger_mamba_base',
        'qa_tiger_mamba_large',
    ]
    
    print("=" * 80)
    print("模型大小对比")
    print("=" * 80)
    
    results = []
    
    for config_name in configs_to_compare:
        try:
            model = create_qa_tiger_with_mamba(
                config_name=config_name,
                device='cpu',  # 使用CPU避免显存问题
                verbose=False
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            results.append((config_name, total_params))
            
            print(f"{config_name:25s}: {total_params:>12,} 参数")
            
            # 清理内存
            del model
            
        except Exception as e:
            print(f"{config_name:25s}: 创建失败 - {str(e)}")
    
    # 计算相对增长
    if len(results) > 1:
        baseline_params = results[0][1]  # 假设第一个是baseline
        print(f"\n相对于基线模型的参数增长:")
        for name, params in results[1:]:
            increase = (params - baseline_params) / baseline_params * 100
            print(f"{name:25s}: +{increase:>6.1f}%")


def benchmark_inference_speed(
    batch_size: int = 2,
    seq_len: int = 30,
    num_patches: int = 10,
    num_runs: int = 10
):
    """基准测试推理速度"""
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过速度测试")
        return
    
    device = 'cuda'
    
    configs_to_test = [
        'qa_tiger_baseline',
        'qa_tiger_mamba_base',
    ]
    
    # 准备测试数据
    reshaped_data = {
        'quest': torch.randint(0, 1000, (batch_size, 77)).to(device),
        'audio': torch.randn(batch_size, seq_len, 128).to(device),
        'video': torch.randn(batch_size, seq_len, 512).to(device),
        'patch': torch.randn(batch_size, seq_len, num_patches, 768).to(device)
    }
    
    print("=" * 80)
    print(f"推理速度测试 (batch_size={batch_size}, seq_len={seq_len}, runs={num_runs})")
    print("=" * 80)
    
    for config_name in configs_to_test:
        try:
            model = create_qa_tiger_with_mamba(
                config_name=config_name,
                device=device,
                verbose=False
            )
            model.eval()
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = model(reshaped_data)
            
            # 测试
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time.record()
                    _ = model(reshaped_data)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
            
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            
            print(f"{config_name:25s}: {avg_time:>7.2f} ± {std_time:>5.2f} ms")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{config_name:25s}: 测试失败 - {str(e)}")


if __name__ == "__main__":
    print("VideoMamba增强QA-TIGER模型工厂测试")
    
    # 1. 创建模型示例
    print("\n1. 创建基础模型")
    print("-" * 40)
    try:
        model = create_qa_tiger_with_mamba('qa_tiger_mamba_base')
        print("✅ 模型创建成功")
        del model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
    
    # 2. 模型大小对比
    print("\n2. 模型大小对比")
    print("-" * 40)
    try:
        compare_model_sizes()
        print("✅ 大小对比完成")
    except Exception as e:
        print(f"❌ 大小对比失败: {e}")
    
    # 3. 推理速度测试
    print("\n3. 推理速度测试")
    print("-" * 40)
    try:
        benchmark_inference_speed()
        print("✅ 速度测试完成")
    except Exception as e:
        print(f"❌ 速度测试失败: {e}")
    
    print("\n" + "=" * 80)
    print("使用示例:")
    print("""
# 创建增强模型
model = create_qa_tiger_with_mamba('qa_tiger_mamba_base')

# 创建基线模型
baseline_model = create_baseline_qa_tiger()

# 自定义配置
custom_model = create_qa_tiger_with_mamba(
    'qa_tiger_mamba_base',
    custom_config={'d_model': 768, 'topK': 4}
)
    """)
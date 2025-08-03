#!/usr/bin/env python3

"""
VideoMamba集成到QA-TIGER的使用示例
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.net import QA_TIGER


def create_enhanced_qa_tiger_model():
    """创建集成VideoMamba的增强版QA-TIGER模型"""
    
    # VideoMamba配置
    mamba_config = {
        'mamba_hidden_dim': 256,        # VideoMamba内部隐藏维度
        'depths': [2, 2, 6, 2],         # 各阶段层数
        'num_heads': [4, 8, 16, 32],    # 各阶段注意力头数
        'drop_path_rate': 0.1,          # DropPath率
        'layer_scale': 1e-6,            # Layer Scale参数
        'causal': False,                # 非因果，可以看到未来帧
    }
    
    # 创建模型
    model = QA_TIGER(
        d_model=512,                    # 主要特征维度
        video_dim=512,                  # 视频特征维度
        audio_dim=128,                  # 音频特征维度
        patch_dim=768,                  # Patch特征维度
        topK=3,                         # TempMoE的topK
        num_experts=10,                 # TempMoE的专家数
        encoder_type='ViT-L/14@336px',  # CLIP编码器类型
        use_video_mamba=True,           # 启用VideoMamba
        mamba_config=mamba_config,      # VideoMamba配置
        use_ams=False,                  # 可选：是否使用AMS
        mccd=None,                      # 可选：MCCD配置
    )
    
    return model


def prepare_sample_data(batch_size=2, seq_len=30, num_patches=10, device='cuda'):
    """准备示例数据"""
    
    # 模拟真实的多模态视频问答数据
    reshaped_data = {
        # 问题：可以是token IDs或预编码的特征
        'quest': torch.randint(0, 1000, (batch_size, 77)).to(device),
        
        # 音频特征：通常来自音频编码器（如Wav2Vec2）
        'audio': torch.randn(batch_size, seq_len, 128).to(device),
        
        # 视频特征：通常来自视频编码器（如ResNet、ViT等）
        'video': torch.randn(batch_size, seq_len, 512).to(device),
        
        # Patch特征：通常来自视觉Transformer的patch embeddings
        'patch': torch.randn(batch_size, seq_len, num_patches, 768).to(device)
    }
    
    return reshaped_data


def run_inference_example():
    """运行推理示例"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    print("创建增强版QA-TIGER模型...")
    model = create_enhanced_qa_tiger_model().to(device)
    model.eval()
    
    # 准备数据
    print("准备示例数据...")
    batch_size = 2
    seq_len = 30
    num_patches = 10
    
    reshaped_data = prepare_sample_data(batch_size, seq_len, num_patches, device)
    
    # 推理
    print("执行推理...")
    with torch.no_grad():
        outputs = model(reshaped_data)
    
    # 解析输出
    logits = outputs['out']                    # 主要分类输出 [B, num_classes]
    fusion_logits = outputs['fusion_logits']   # 融合特征的logits
    
    print(f"\n推理结果:")
    print(f"分类logits形状: {logits.shape}")
    print(f"预测类别: {torch.argmax(logits, dim=-1)}")
    print(f"预测概率: {torch.softmax(logits, dim=-1).max(dim=-1)[0]}")
    
    # 计算模型统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    return outputs


def compare_with_without_mamba():
    """对比使用和不使用VideoMamba的性能"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备相同的测试数据
    reshaped_data = prepare_sample_data(batch_size=2, seq_len=30, num_patches=10, device=device)
    
    # 创建两个模型进行对比
    print("创建带VideoMamba的模型...")
    model_with_mamba = create_enhanced_qa_tiger_model().to(device)
    model_with_mamba.eval()
    
    print("创建不带VideoMamba的模型...")
    model_without_mamba = QA_TIGER(
        d_model=512,
        video_dim=512,
        audio_dim=128,
        patch_dim=768,
        encoder_type='ViT-L/14@336px',
        use_video_mamba=False,  # 关键差异
    ).to(device)
    model_without_mamba.eval()
    
    # 推理对比
    with torch.no_grad():
        # 带VideoMamba的推理
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        outputs_with_mamba = model_with_mamba(reshaped_data)
        end_time.record()
        torch.cuda.synchronize()
        time_with_mamba = start_time.elapsed_time(end_time)
        
        # 不带VideoMamba的推理
        start_time.record()
        outputs_without_mamba = model_without_mamba(reshaped_data)
        end_time.record()
        torch.cuda.synchronize()
        time_without_mamba = start_time.elapsed_time(end_time)
    
    # 统计对比
    params_with_mamba = sum(p.numel() for p in model_with_mamba.parameters())
    params_without_mamba = sum(p.numel() for p in model_without_mamba.parameters())
    
    print(f"\n性能对比:")
    print(f"带VideoMamba - 参数量: {params_with_mamba:,}, 推理时间: {time_with_mamba:.2f}ms")
    print(f"不带VideoMamba - 参数量: {params_without_mamba:,}, 推理时间: {time_without_mamba:.2f}ms")
    print(f"参数增加: {params_with_mamba - params_without_mamba:,} ({(params_with_mamba/params_without_mamba-1)*100:.1f}%)")
    print(f"时间增加: {time_with_mamba - time_without_mamba:.2f}ms ({(time_with_mamba/time_without_mamba-1)*100:.1f}%)")
    
    # 输出差异分析
    logits_diff = torch.abs(outputs_with_mamba['out'] - outputs_without_mamba['out']).mean()
    print(f"输出logits平均差异: {logits_diff:.6f}")


def training_example():
    """训练示例（伪代码）"""
    
    print("\n训练配置示例:")
    print("""
    # 创建模型
    model = create_enhanced_qa_tiger_model()
    
    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100
    )
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = criterion(outputs['out'], batch['labels'])
            
            # 如果使用MCCD，可以添加偏置损失
            if outputs['q_bias_logits'] is not None:
                bias_loss = criterion(outputs['q_bias_logits'], batch['labels'])
                loss += 0.1 * bias_loss
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    """)


if __name__ == "__main__":
    print("=" * 80)
    print("VideoMamba集成QA-TIGER使用示例")
    print("=" * 80)
    
    try:
        # 基本推理示例
        print("\n1. 基本推理示例")
        print("-" * 40)
        run_inference_example()
        
        # 性能对比
        print("\n2. 性能对比")
        print("-" * 40)
        compare_with_without_mamba()
        
        # 训练示例
        print("\n3. 训练配置")
        print("-" * 40)
        training_example()
        
        print("\n✅ 所有示例运行成功!")
        
    except Exception as e:
        print(f"\n❌ 运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
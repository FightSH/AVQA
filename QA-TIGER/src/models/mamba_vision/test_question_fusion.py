#!/usr/bin/env python3

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from models.question_aware_video_mamba import (
    question_aware_video_mamba_base,
    question_aware_video_mamba_small
)


def test_fusion_strategies():
    """测试不同融合策略的性能"""
    print("🔬 测试不同问题融合策略")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试配置
    batch_size = 4
    seq_len = 30
    img_dim = 2048
    audio_dim = 128
    question_dim = 512
    num_classes = 100
    
    # 创建测试数据
    img_features = torch.randn(batch_size, seq_len, img_dim).to(device)
    audio_features = torch.randn(batch_size, seq_len, audio_dim).to(device)
    question_vector = torch.randn(batch_size, question_dim).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    strategies = ['early_concat', 'cross_attention', 'question_guided', 'adaptive_fusion']
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 测试策略: {strategy}")
        print("-" * 40)
        
        try:
            # 创建模型
            model = question_aware_video_mamba_small(
                img_dim=img_dim,
                audio_dim=audio_dim,
                question_dim=question_dim,
                num_classes=num_classes,
                fusion_strategy=strategy
            ).to(device)
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 测试前向传播
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                for _ in range(10):  # 多次测试取平均
                    output = model(img_features, audio_features, question_vector)
                end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / 10 * 1000  # ms
            
            # 测试梯度流
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            # 前向传播
            output = model(img_features, audio_features, question_vector)
            loss = criterion(output, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 计算梯度范数
            total_grad_norm = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            # 记录结果
            results[strategy] = {
                'params': total_params,
                'trainable_params': trainable_params,
                'inference_time': avg_inference_time,
                'loss': loss.item(),
                'grad_norm': total_grad_norm,
                'output_shape': output.shape,
                'success': True
            }
            
            print(f"  ✅ 参数量: {total_params:,}")
            print(f"  ✅ 推理时间: {avg_inference_time:.2f} ms")
            print(f"  ✅ 损失值: {loss.item():.4f}")
            print(f"  ✅ 梯度范数: {total_grad_norm:.6f}")
            print(f"  ✅ 输出形状: {output.shape}")
            
        except Exception as e:
            print(f"  ❌ 策略 {strategy} 测试失败: {e}")
            results[strategy] = {'success': False, 'error': str(e)}
    
    return results


def compare_fusion_strategies(results):
    """对比不同融合策略的结果"""
    print(f"\n📈 融合策略对比分析")
    print("=" * 60)
    
    # 过滤成功的结果
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ 没有成功的测试结果")
        return
    
    # 创建对比表格
    print(f"{'策略':<15} {'参数量':<12} {'推理时间(ms)':<12} {'梯度范数':<12} {'相对性能':<10}")
    print("-" * 70)
    
    # 找到基准值（early_concat）
    baseline_time = successful_results.get('early_concat', {}).get('inference_time', 1)
    baseline_params = successful_results.get('early_concat', {}).get('params', 1)
    
    for strategy, result in successful_results.items():
        params = result['params']
        inference_time = result['inference_time']
        grad_norm = result['grad_norm']
        
        # 计算相对性能
        time_ratio = inference_time / baseline_time
        param_ratio = params / baseline_params
        
        print(f"{strategy:<15} {params:<12,} {inference_time:<12.2f} {grad_norm:<12.6f} "
              f"{time_ratio:.2f}x")
    
    # 生成可视化图表
    if len(successful_results) > 1:
        create_comparison_plots(successful_results)


def create_comparison_plots(results):
    """创建对比图表"""
    strategies = list(results.keys())
    
    # 提取数据
    params = [results[s]['params'] / 1e6 for s in strategies]  # 转换为百万参数
    inference_times = [results[s]['inference_time'] for s in strategies]
    grad_norms = [results[s]['grad_norm'] for s in strategies]
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 参数量对比
    axes[0].bar(strategies, params, color='skyblue')
    axes[0].set_title('参数量对比 (M)')
    axes[0].set_ylabel('参数量 (百万)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 推理时间对比
    axes[1].bar(strategies, inference_times, color='lightgreen')
    axes[1].set_title('推理时间对比 (ms)')
    axes[1].set_ylabel('推理时间 (毫秒)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 梯度范数对比
    axes[2].bar(strategies, grad_norms, color='salmon')
    axes[2].set_title('梯度范数对比')
    axes[2].set_ylabel('梯度范数')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('mambavision/fusion_strategies_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图表已保存到: fusion_strategies_comparison.png")


def test_ablation_study():
    """消融实验：测试有无问题向量的差异"""
    print(f"\n🔬 消融实验：问题向量的影响")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试数据
    batch_size = 4
    seq_len = 20
    img_dim = 1024
    audio_dim = 128
    question_dim = 256
    num_classes = 50
    
    img_features = torch.randn(batch_size, seq_len, img_dim).to(device)
    audio_features = torch.randn(batch_size, seq_len, audio_dim).to(device)
    question_vector = torch.randn(batch_size, question_dim).to(device)
    zero_question = torch.zeros(batch_size, question_dim).to(device)
    
    # 测试不同策略下问题向量的影响
    strategies = ['cross_attention', 'question_guided', 'adaptive_fusion']
    
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        print("-" * 30)
        
        try:
            model = question_aware_video_mamba_small(
                img_dim=img_dim,
                audio_dim=audio_dim,
                question_dim=question_dim,
                num_classes=num_classes,
                fusion_strategy=strategy
            ).to(device)
            
            model.eval()
            with torch.no_grad():
                # 有问题向量
                output_with_q = model(img_features, audio_features, question_vector)
                
                # 无问题向量（零向量）
                output_without_q = model(img_features, audio_features, zero_question)
                
                # 计算差异
                output_diff = torch.abs(output_with_q - output_without_q).mean().item()
                
                # 计算输出分布的差异
                prob_with_q = torch.softmax(output_with_q, dim=-1)
                prob_without_q = torch.softmax(output_without_q, dim=-1)
                kl_div = torch.nn.functional.kl_div(
                    prob_without_q.log(), prob_with_q, reduction='batchmean'
                ).item()
                
                print(f"  输出差异 (L1): {output_diff:.6f}")
                print(f"  概率分布差异 (KL): {kl_div:.6f}")
                
                # 判断问题向量的影响程度
                if output_diff > 0.1:
                    print("  ✅ 问题向量有显著影响")
                elif output_diff > 0.01:
                    print("  ⚠️  问题向量有中等影响")
                else:
                    print("  ❌ 问题向量影响很小")
                    
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")


def test_question_sensitivity():
    """测试模型对不同问题的敏感性"""
    print(f"\n🎯 问题敏感性测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = question_aware_video_mamba_small(
        img_dim=512,
        audio_dim=64,
        question_dim=128,
        num_classes=10,
        fusion_strategy='cross_attention'
    ).to(device)
    
    # 固定的视频特征
    batch_size = 1
    seq_len = 16
    img_features = torch.randn(batch_size, seq_len, 512).to(device)
    audio_features = torch.randn(batch_size, seq_len, 64).to(device)
    
    # 生成不同的问题向量
    num_questions = 5
    question_vectors = [torch.randn(batch_size, 128).to(device) for _ in range(num_questions)]
    
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for i, question_vec in enumerate(question_vectors):
            output = model(img_features, audio_features, question_vec)
            outputs.append(output)
            print(f"问题 {i+1}: 预测类别 {output.argmax().item()}, "
                  f"最大概率 {torch.softmax(output, dim=-1).max().item():.3f}")
    
    # 计算输出之间的差异
    print(f"\n输出差异分析:")
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff = torch.abs(outputs[i] - outputs[j]).mean().item()
            print(f"  问题{i+1} vs 问题{j+1}: {diff:.6f}")


def main():
    """主测试函数"""
    print("🧪 问题融合策略测试套件")
    print("=" * 80)
    
    # 测试1: 基础功能测试
    results = test_fusion_strategies()
    
    # 测试2: 策略对比分析
    compare_fusion_strategies(results)
    
    # 测试3: 消融实验
    test_ablation_study()
    
    # 测试4: 问题敏感性测试
    test_question_sensitivity()
    
    print(f"\n🎉 所有测试完成！")
    
    # 给出使用建议
    print(f"\n💡 使用建议:")
    print("1. 对于大多数任务，推荐使用 'cross_attention' 策略")
    print("2. 如果计算资源受限，可以使用 'early_concat'")
    print("3. 如果需要模态选择，考虑 'question_guided'")
    print("4. 对于复杂任务且有充足数据，尝试 'adaptive_fusion'")


if __name__ == "__main__":
    main()
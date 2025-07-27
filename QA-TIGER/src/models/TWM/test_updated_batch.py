import torch
import sys
sys.path.append('src/models/TWM')
from alvs_inter import iterative_sampling as original_sampling
from batch_alvs_inter import iterative_sampling as updated_batch_sampling

def test_updated_version():
    """测试更新后的批量版本是否仍有问题"""
    print("=== 测试更新后的批量版本 ===")
    
    torch.manual_seed(42)
    
    batch_size = 3
    t = 50
    d = 128
    k = 5
    m = 2
    a1, a2 = 0.6, 0.4
    
    # 生成测试数据
    f_v_batch = torch.randn(batch_size, t, d)
    f_text_batch = torch.randn(batch_size, d)
    
    print(f"测试参数: batch_size={batch_size}, t={t}, k={k}, m={m}")
    
    # 方法1: 单独处理每个序列
    individual_results = []
    for i in range(batch_size):
        f_v_single = f_v_batch[i]  # [t, d]
        f_text_single = f_text_batch[i]  # [d]
        
        result = original_sampling(f_v_single, f_text_single, k, m, a1, a2)
        individual_results.append(result)
        print(f"  单独处理批次{i}: {len(result)}帧, 索引={result}")
    
    # 方法2: 更新后的批量处理
    batch_results = updated_batch_sampling(f_v_batch, f_text_batch, k, m, a1, a2)
    print(f"\n更新后批量处理结果:")
    for i, result in enumerate(batch_results):
        print(f"  批量处理批次{i}: {len(result)}帧, 索引={result}")
    
    # 比较结果
    print(f"\n结果比较:")
    all_same = True
    for i in range(batch_size):
        is_same = individual_results[i] == batch_results[i]
        print(f"  批次{i}: {'相同' if is_same else '不同'}")
        if not is_same:
            all_same = False
            print(f"    单独: {individual_results[i]}")
            print(f"    批量: {batch_results[i]}")
            
            # 计算相似度
            set1 = set(individual_results[i])
            set2 = set(batch_results[i])
            intersection = set1 & set2
            union = set1 | set2
            jaccard = len(intersection) / len(union) if union else 1.0
            print(f"    Jaccard相似度: {jaccard:.3f}")
    
    print(f"所有结果是否相同: {'是' if all_same else '否'}")
    return all_same

def test_target_frames_feature():
    """测试新增的target_num_frames功能"""
    print(f"\n=== 测试target_num_frames功能 ===")
    
    torch.manual_seed(123)
    
    batch_size = 4
    target_frames = 20
    
    f_v_batch = torch.randn(batch_size, 60, 256)
    f_text_batch = torch.randn(batch_size, 256)
    
    # 不指定target_num_frames
    results_variable = updated_batch_sampling(f_v_batch, f_text_batch, 8, 3, 0.5, 0.5)
    variable_counts = [len(r) for r in results_variable]
    
    # 指定target_num_frames
    results_fixed = updated_batch_sampling(f_v_batch, f_text_batch, 8, 3, 0.5, 0.5, target_num_frames=target_frames)
    fixed_counts = [len(r) for r in results_fixed]
    
    print(f"可变长度结果: {variable_counts}")
    print(f"固定长度结果: {fixed_counts}")
    print(f"是否全部为{target_frames}: {all(c == target_frames for c in fixed_counts)}")
    
    # 测试转换为张量
    try:
        tensor_result = torch.tensor(results_fixed)
        print(f"成功转换为张量，形状: {tensor_result.shape}")
        return True
    except Exception as e:
        print(f"转换为张量失败: {e}")
        return False

def analyze_improvements():
    """分析改进点"""
    print(f"\n=== 分析改进点 ===")
    
    print("✅ 改进点:")
    print("  1. 添加了target_num_frames参数，解决固定长度输出问题")
    print("  2. 改进了_adjust_indices_to_target_length函数，处理更合理")
    print("  3. 代码结构更清晰，类型注解更完善")
    
    print("\n❌ 仍存在的问题:")
    print("  1. 核心的强制同步批量处理逻辑未改变")
    print("  2. 所有批次项仍然被迫同时进行相同次数的迭代")
    print("  3. 结果仍然与单独处理不一致")
    print("  4. 算法的本质行为被批量化改变了")
    
    print("\n🔍 根本问题:")
    print("  这种批量化方式假设所有序列都应该有相同的迭代模式")
    print("  但实际上每个序列的数据特征不同，需要不同的处理策略")
    print("  强制同步会导致某些序列得到次优结果")

if __name__ == "__main__":
    correctness_ok = test_updated_version()
    target_frames_ok = test_target_frames_feature()
    analyze_improvements()
    
    print(f"\n=== 总结 ===")
    print(f"正确性: {'✓' if correctness_ok else '✗'}")
    print(f"固定帧数功能: {'✓' if target_frames_ok else '✗'}")
    
    if not correctness_ok:
        print("\n⚠️  建议: 使用batch_alvs_inter_correct.py中的实现")
        print("   该版本保证与原版结果完全一致，同时支持固定帧数输出")
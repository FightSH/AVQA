import torch
import sys
sys.path.append('src/models/TWM')
from alvs_inter import iterative_sampling as original_sampling
from batch_alvs_inter_fixed_frames import iterative_sampling as batch_sampling

def compare_single_vs_batch():
    """比较单独处理和批量处理的结果"""
    print("=== 比较单独处理 vs 批量处理 ===")
    
    # 设置相同的随机种子
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
    
    # 方法2: 批量处理
    batch_results = batch_sampling(f_v_batch, f_text_batch, k, m, a1, a2)
    print(f"\n批量处理结果:")
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
    
    print(f"所有结果是否相同: {'是' if all_same else '否'}")
    return individual_results, batch_results

def analyze_batch_synchronization_issue():
    """分析批量同步问题"""
    print(f"\n=== 分析批量同步问题 ===")
    
    # 创建差异很大的数据来突出问题
    torch.manual_seed(123)
    
    # 批次1: 特征变化很大，需要更多迭代
    f_v1 = torch.randn(30, 64) * 2  # 大方差
    f_text1 = torch.randn(64)
    
    # 批次2: 特征变化很小，可能很快收敛
    f_v2 = torch.randn(30, 64) * 0.1  # 小方差
    f_text2 = torch.randn(64)
    
    # 批次3: 中等变化
    f_v3 = torch.randn(30, 64)
    f_text3 = torch.randn(64)
    
    f_v_batch = torch.stack([f_v1, f_v2, f_v3])
    f_text_batch = torch.stack([f_text1, f_text2, f_text3])
    
    print("测试数据特征:")
    print(f"  批次1方差: {f_v1.var().item():.4f} (大方差)")
    print(f"  批次2方差: {f_v2.var().item():.4f} (小方差)")
    print(f"  批次3方差: {f_v3.var().item():.4f} (中等方差)")
    
    # 单独处理
    individual_results = []
    for i, (f_v_single, f_text_single) in enumerate(zip([f_v1, f_v2, f_v3], [f_text1, f_text2, f_text3])):
        result = original_sampling(f_v_single, f_text_single, 5, 3, 0.5, 0.5)
        individual_results.append(result)
        print(f"单独处理批次{i}: {result}")
    
    # 批量处理
    batch_results = batch_sampling(f_v_batch, f_text_batch, 5, 3, 0.5, 0.5)
    print(f"\n批量处理结果:")
    for i, result in enumerate(batch_results):
        print(f"批量处理批次{i}: {result}")
    
    # 分析差异
    print(f"\n差异分析:")
    for i in range(3):
        if individual_results[i] != batch_results[i]:
            print(f"  批次{i}存在差异:")
            print(f"    单独: {individual_results[i]}")
            print(f"    批量: {batch_results[i]}")
            
            # 计算差异程度
            set1 = set(individual_results[i])
            set2 = set(batch_results[i])
            intersection = set1 & set2
            union = set1 | set2
            jaccard = len(intersection) / len(union) if union else 1.0
            print(f"    Jaccard相似度: {jaccard:.3f}")

def identify_core_issues():
    """识别核心问题"""
    print(f"\n=== 核心问题识别 ===")
    
    print("1. 强制同步问题:")
    print("   - 所有批次项必须进行相同次数的迭代")
    print("   - 但不同数据可能需要不同的迭代次数")
    print("   - 有些序列可能很快收敛，有些可能需要更多探索")
    
    print("\n2. 窗口大小统一问题:")
    print("   - 当前实现中，窗口大小基于当前长度计算")
    print("   - 但不同批次的数据特征差异很大")
    print("   - 应该根据数据的实际分布来调整窗口")
    
    print("\n3. 中心选择的批量化问题:")
    print("   - 当前使用argmax选择最佳中心")
    print("   - 但这种选择对每个批次项应该是独立的")
    print("   - 批量化可能导致次优选择")
    
    print("\n4. 建议的解决方案:")
    print("   方案A: 真正的独立批处理")
    print("   - 每个批次项完全独立处理")
    print("   - 只是在GPU上并行执行")
    print("   - 保证结果与单独处理完全一致")
    
    print("\n   方案B: 改进的批量算法")
    print("   - 设计真正适合批量处理的算法")
    print("   - 允许不同批次项有不同的迭代策略")
    print("   - 但这可能改变算法的本质")
    
    print("\n   方案C: 混合方案")
    print("   - 保持当前的批量框架")
    print("   - 但增加更多的独立性检查")
    print("   - 允许批次项提前结束或延长迭代")

if __name__ == "__main__":
    individual_results, batch_results = compare_single_vs_batch()
    analyze_batch_synchronization_issue()
    identify_core_issues()
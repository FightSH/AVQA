import torch
import sys
sys.path.append('src/models/TWM')
from batch_alvs_inter import iterative_sampling

def test_frame_count_consistency():
    """测试批量版本是否产生不同数量的帧"""
    print("=== 测试批量版帧数量一致性 ===")
    
    # 设置不同的随机种子来模拟不同的数据
    batch_size = 5
    t = 100
    d = 512
    k = 10
    m = 3
    a1, a2 = 0.5, 0.5
    
    results = []
    
    for seed in range(5):
        torch.manual_seed(seed)
        f_v_batch = torch.randn(batch_size, t, d)
        f_text_batch = torch.randn(batch_size, d)
        
        indices_batch = iterative_sampling(f_v_batch, f_text_batch, k, m, a1, a2)
        
        frame_counts = [len(indices) for indices in indices_batch]
        print(f"种子{seed}: 帧数量 = {frame_counts}")
        results.append(frame_counts)
    
    # 分析结果
    all_counts = [count for result in results for count in result]
    min_count = min(all_counts)
    max_count = max(all_counts)
    avg_count = sum(all_counts) / len(all_counts)
    
    print(f"\n统计结果:")
    print(f"  最小帧数: {min_count}")
    print(f"  最大帧数: {max_count}")
    print(f"  平均帧数: {avg_count:.2f}")
    print(f"  帧数差异: {max_count - min_count}")
    
    # 检查是否所有批次都有相同的帧数
    consistent = all(len(set(result)) == 1 for result in results)
    print(f"  批次内帧数是否一致: {consistent}")
    
    return results

def analyze_theoretical_frame_count():
    """分析理论上的帧数量"""
    print(f"\n=== 理论分析 ===")
    
    k = 10  # 每次迭代采样k个向量
    m = 3   # 迭代m次
    
    print(f"参数: k={k}, m={m}")
    print(f"理论最大帧数: {k * m} = {k * m}")
    print(f"但由于去重处理，实际帧数会更少")
    
    # 模拟一个简单的情况
    print(f"\n模拟分析:")
    print(f"- 第1次迭代: 采样{k}个帧")
    print(f"- 第2次迭代: 在缩小范围内再采样{k}个帧 (可能有重复)")
    print(f"- 第3次迭代: 在进一步缩小范围内采样{k}个帧 (更多重复)")
    print(f"- 最终去重后: 帧数取决于重复程度")

def propose_solutions():
    """提出解决方案"""
    print(f"\n=== 解决方案 ===")
    
    print("方案1: 固定帧数输出")
    print("  - 设定目标帧数 target_frames")
    print("  - 如果帧数不足，重复最后几帧")
    print("  - 如果帧数过多，截断到目标数量")
    print("  - 优点: 输出形状固定，便于批处理")
    print("  - 缺点: 可能丢失信息或引入冗余")
    
    print("\n方案2: 填充到最大长度")
    print("  - 找到批次中的最大帧数")
    print("  - 用特殊值(如-1)填充较短的序列")
    print("  - 后续处理时忽略填充值")
    print("  - 优点: 保留所有信息")
    print("  - 缺点: 需要额外的掩码处理")
    
    print("\n方案3: 分别处理")
    print("  - 不强制批量处理")
    print("  - 每个序列单独处理")
    print("  - 优点: 最灵活，保留所有信息")
    print("  - 缺点: 失去批处理的效率优势")
    
    print("\n方案4: 修改算法确保一致性")
    print("  - 修改采样策略，确保每次产生固定数量的唯一帧")
    print("  - 例如：强制每次迭代产生不重复的帧")
    print("  - 优点: 从根本上解决问题")
    print("  - 缺点: 可能改变算法行为")

if __name__ == "__main__":
    results = test_frame_count_consistency()
    analyze_theoretical_frame_count()
    propose_solutions()
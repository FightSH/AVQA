import torch
import torch.nn.functional as F

def sample_vectors(f_v, k):
    """原版采样函数"""
    t = f_v.size(0)
    indices = torch.linspace(0, t - 1, steps=k).long()
    sampled_vectors = f_v[indices]
    return sampled_vectors, indices

def calculate_attention(f_qst, f_v_sampled):
    """原版注意力计算函数"""
    print(f"  calculate_attention输入:")
    print(f"    f_qst形状: {f_qst.shape}")
    print(f"    f_v_sampled形状: {f_v_sampled.shape}")
    
    q = f_qst.unsqueeze(0)  # Shape: [1, d]
    k = f_v_sampled         # Shape: [k, d] 或 [1, d]
    v = f_v_sampled         # Shape: [k, d] 或 [1, d]

    print(f"    q形状: {q.shape}")
    print(f"    k形状: {k.shape}")
    print(f"    v形状: {v.shape}")

    # Attention calculation: att_weights = softmax(q * k^T / sqrt(d))
    d = f_qst.size(-1)
    attention_scores = torch.matmul(q, k.T) / (d ** 0.5)
    print(f"    attention_scores形状: {attention_scores.shape}")
    print(f"    attention_scores值: {attention_scores}")
    
    att_weights = F.softmax(attention_scores, dim=-1)
    print(f"    att_weights形状: {att_weights.shape}")
    print(f"    att_weights值: {att_weights}")
    
    att_f_v = torch.matmul(att_weights, v)  # Shape: [1, d] 或 [1, k, d]
    print(f"    att_f_v形状: {att_f_v.shape}")
    
    result = att_f_v.squeeze(0)  # Shape: [d] 或 [k, d]
    print(f"    最终结果形状: {result.shape}")
    return result

def verify_original_attention_logic():
    """验证原版注意力逻辑"""
    print("=== 验证原版注意力逻辑 ===")
    
    # 设置测试数据
    torch.manual_seed(42)
    d = 8  # 特征维度，使用小值便于观察
    k = 3  # 采样数量
    t = 10 # 序列长度
    
    f_v = torch.randn(t, d)
    f_text = torch.randn(d)
    
    print(f"测试数据:")
    print(f"  f_v形状: {f_v.shape}")
    print(f"  f_text形状: {f_text.shape}")
    print(f"  k={k}, t={t}, d={d}")
    
    # 步骤1: 采样向量
    print(f"\n步骤1: 采样向量")
    f_v_sampled, sampled_indices = sample_vectors(f_v, k)
    print(f"  f_v_sampled形状: {f_v_sampled.shape}")
    print(f"  sampled_indices: {sampled_indices}")
    print(f"  f_v_sampled:\n{f_v_sampled}")
    
    # 步骤2: 原版的注意力计算方式
    print(f"\n步骤2: 原版注意力计算方式")
    print("原版代码: att_f_v = torch.stack([calculate_attention(f_text, f_v_sampled[i].unsqueeze(0)) for i in range(k)])")
    
    att_f_v_list = []
    for i in range(k):
        print(f"\n  处理第{i}个采样向量:")
        single_vector = f_v_sampled[i].unsqueeze(0)  # [1, d]
        print(f"    single_vector形状: {single_vector.shape}")
        print(f"    single_vector值: {single_vector}")
        
        att_result = calculate_attention(f_text, single_vector)
        att_f_v_list.append(att_result)
        
        print(f"    注意力结果: {att_result}")
        print(f"    是否等于原向量: {torch.allclose(att_result, single_vector.squeeze(0))}")
    
    att_f_v = torch.stack(att_f_v_list)
    print(f"\n  最终att_f_v形状: {att_f_v.shape}")
    print(f"  att_f_v:\n{att_f_v}")
    
    # 验证是否等于原采样向量
    print(f"  att_f_v是否等于f_v_sampled: {torch.allclose(att_f_v, f_v_sampled)}")
    
    return att_f_v, f_v_sampled

def verify_alternative_attention():
    """验证可能的正确注意力逻辑"""
    print(f"\n=== 验证可能的正确注意力逻辑 ===")
    
    torch.manual_seed(42)
    d = 8
    k = 3
    t = 10
    
    f_v = torch.randn(t, d)
    f_text = torch.randn(d)
    f_v_sampled, _ = sample_vectors(f_v, k)
    
    print("假设原版想要的是: calculate_attention(f_text, f_v_sampled)")
    print("即对所有k个向量一起计算注意力")
    
    att_result = calculate_attention(f_text, f_v_sampled)
    print(f"结果形状: {att_result.shape}")
    print(f"结果: {att_result}")
    
    return att_result

def verify_sim2_calculation():
    """验证sim2的计算"""
    print(f"\n=== 验证sim2计算 ===")
    
    torch.manual_seed(42)
    d = 8
    k = 3
    
    # 使用之前的结果
    att_f_v, f_v_sampled = verify_original_attention_logic()
    
    print(f"\n计算sim2:")
    print(f"  att_f_v形状: {att_f_v.shape}")
    
    if k > 1:
        sim2 = F.cosine_similarity(att_f_v[:-1], att_f_v[1:], dim=-1)
        print(f"  连续向量相似度: {sim2}")
        print(f"  sim2形状: {sim2.shape}")
        
        # 原版的处理：复制最后一个元素
        sim2 = torch.cat([sim2, sim2[-1].unsqueeze(0)])
        print(f"  填充后sim2: {sim2}")
        print(f"  填充后sim2形状: {sim2.shape}")
        
        # 对比：如果直接计算f_v_sampled的连续相似度
        sim2_direct = F.cosine_similarity(f_v_sampled[:-1], f_v_sampled[1:], dim=-1)
        sim2_direct = torch.cat([sim2_direct, sim2_direct[-1].unsqueeze(0)])
        print(f"  直接计算f_v_sampled的sim2: {sim2_direct}")
        print(f"  两种方式是否相同: {torch.allclose(sim2, sim2_direct)}")
    
    return sim2

def main():
    print("开始验证原版逻辑...")
    
    # 验证注意力计算
    att_f_v, f_v_sampled = verify_original_attention_logic()
    
    # 验证可能的正确逻辑
    verify_alternative_attention()
    
    # 验证sim2计算
    verify_sim2_calculation()
    
    print(f"\n=== 总结 ===")
    print("1. 原版的注意力计算对单个向量进行，结果就是原向量本身")
    print("2. 这意味着att_f_v实际上等于f_v_sampled")
    print("3. sim2的计算实际上是在计算采样向量之间的连续相似度")
    print("4. 整个注意力机制在当前实现下是冗余的")

if __name__ == "__main__":
    main()
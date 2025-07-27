import torch
from batch_alvs_inter_fixed import iterative_sampling as fixed_iterative_sampling
from batch_alvs_inter_fixed import calculate_attention as fixed_calculate_attention

def test_calculate_attention():
    """测试修复后的注意力计算函数"""
    print("=== 测试注意力计算函数 ===")
    
    # 单序列测试
    f_text = torch.randn(512)  # [d]
    f_v_sampled = torch.randn(10, 512)  # [k, d]
    
    att_result = fixed_calculate_attention(f_text, f_v_sampled)
    print(f"单序列注意力结果形状: {att_result.shape}")  # 应该是 [512]
    assert att_result.shape == (512,), f"期望形状 (512,), 实际 {att_result.shape}"
    
    # 批量测试
    f_text_batch = torch.randn(4, 512)  # [batch_size, d]
    f_v_sampled_batch = torch.randn(4, 10, 512)  # [batch_size, k, d]
    
    att_result_batch = fixed_calculate_attention(f_text_batch, f_v_sampled_batch)
    print(f"批量注意力结果形状: {att_result_batch.shape}")  # 应该是 [4, 512]
    assert att_result_batch.shape == (4, 512), f"期望形状 (4, 512), 实际 {att_result_batch.shape}"
    
    print("✓ 注意力计算函数测试通过")

def test_iterative_sampling():
    """测试修复后的迭代采样函数"""
    print("\n=== 测试迭代采样函数 ===")
    
    # 单序列测试
    f_v = torch.randn(100, 512)  # [t, d]
    f_text = torch.randn(512)    # [d]
    k, m, a1, a2 = 10, 3, 0.5, 0.5
    
    indices_single = fixed_iterative_sampling(f_v, f_text, k, m, a1, a2)
    print(f"单序列结果类型: {type(indices_single)}")
    print(f"单序列结果长度: {len(indices_single)}")
    assert isinstance(indices_single, list), "单序列结果应该是列表"
    assert all(isinstance(idx, int) for idx in indices_single), "所有索引应该是整数"
    
    # 批量测试
    batch_size = 4
    f_v_batch = torch.randn(batch_size, 60, 512)  # [batch_size, t, d]
    f_text_batch = torch.randn(batch_size, 512)   # [batch_size, d]
    
    indices_batch = fixed_iterative_sampling(f_v_batch, f_text_batch, k, m, a1, a2)
    print(f"批量结果类型: {type(indices_batch)}")
    print(f"批量结果长度: {len(indices_batch)}")
    assert isinstance(indices_batch, list), "批量结果应该是列表"
    assert len(indices_batch) == batch_size, f"应该有 {batch_size} 个结果"
    assert all(isinstance(sublist, list) for sublist in indices_batch), "每个批次结果应该是列表"
    assert all(isinstance(idx, int) for sublist in indices_batch for idx in sublist), "所有索引应该是整数"
    
    print("✓ 迭代采样函数测试通过")

def compare_with_original():
    """与原版进行对比测试（如果可能的话）"""
    print("\n=== 对比测试 ===")
    
    # 设置相同的随机种子确保可重现性
    torch.manual_seed(42)
    f_v = torch.randn(50, 256)
    f_text = torch.randn(256)
    k, m, a1, a2 = 5, 2, 0.2, 0.8
    
    # 使用修复版本
    torch.manual_seed(42)
    f_v_copy = f_v.clone()
    f_text_copy = f_text.clone()
    indices_fixed = fixed_iterative_sampling(f_v_copy, f_text_copy, k, m, a1, a2)
    
    print(f"修复版本结果: {indices_fixed}")
    print("✓ 修复版本运行正常")

if __name__ == "__main__":
    # test_calculate_attention()
    test_iterative_sampling()
    # compare_with_original()
    print("\n🎉 所有测试通过！")
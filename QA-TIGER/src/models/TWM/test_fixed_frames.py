import torch
from batch_alvs_inter_fixed_frames import iterative_sampling

def test_fixed_frame_output():
    """测试固定帧数输出"""
    print("=== 测试固定帧数输出 ===")
    
    batch_size = 4
    t = 100
    d = 512
    k = 10
    m = 3
    a1, a2 = 0.5, 0.5
    target_frames = 20  # 固定输出20帧
    
    # 测试多个随机种子
    for seed in range(3):
        torch.manual_seed(seed)
        f_v_batch = torch.randn(batch_size, t, d)
        f_text_batch = torch.randn(batch_size, d)
        
        # 不指定target_frames的情况
        indices_variable = iterative_sampling(f_v_batch, f_text_batch, k, m, a1, a2)
        variable_counts = [len(indices) for indices in indices_variable]
        
        # 指定target_frames的情况
        indices_fixed = iterative_sampling(f_v_batch, f_text_batch, k, m, a1, a2, target_frames=target_frames)
        fixed_counts = [len(indices) for indices in indices_fixed]
        
        print(f"种子{seed}:")
        print(f"  可变长度: {variable_counts}")
        print(f"  固定长度: {fixed_counts}")
        print(f"  是否全部为{target_frames}: {all(count == target_frames for count in fixed_counts)}")

def test_edge_cases():
    """测试边界情况"""
    print(f"\n=== 测试边界情况 ===")
    
    # 情况1: 原始帧数少于目标帧数
    print("情况1: 原始帧数少于目标帧数")
    torch.manual_seed(42)
    f_v = torch.randn(1, 50, 256)  # 只有50帧
    f_text = torch.randn(1, 256)
    
    indices = iterative_sampling(f_v, f_text, 5, 2, 0.5, 0.5, target_frames=30)
    print(f"  输入50帧，目标30帧，实际输出: {len(indices[0])}帧")
    print(f"  输出索引: {indices[0]}")
    
    # 情况2: 原始帧数多于目标帧数
    print(f"\n情况2: 原始帧数多于目标帧数")
    torch.manual_seed(42)
    f_v = torch.randn(1, 200, 256)  # 200帧
    f_text = torch.randn(1, 256)
    
    indices = iterative_sampling(f_v, f_text, 10, 3, 0.5, 0.5, target_frames=15)
    print(f"  输入200帧，目标15帧，实际输出: {len(indices[0])}帧")
    print(f"  输出索引: {indices[0]}")

def test_batch_consistency():
    """测试批量一致性"""
    print(f"\n=== 测试批量一致性 ===")
    
    batch_size = 6
    target_frames = 25
    
    torch.manual_seed(123)
    f_v_batch = torch.randn(batch_size, 150, 512)
    f_text_batch = torch.randn(batch_size, 512)
    
    indices_batch = iterative_sampling(f_v_batch, f_text_batch, 8, 4, 0.6, 0.4, target_frames=target_frames)
    
    print(f"批次大小: {batch_size}")
    print(f"目标帧数: {target_frames}")
    
    all_consistent = True
    for i, indices in enumerate(indices_batch):
        is_consistent = len(indices) == target_frames
        print(f"  批次{i}: {len(indices)}帧 {'✓' if is_consistent else '✗'}")
        if not is_consistent:
            all_consistent = False
    
    print(f"所有批次是否一致: {'✓' if all_consistent else '✗'}")
    
    # 转换为张量测试
    try:
        indices_tensor = torch.tensor(indices_batch)
        print(f"成功转换为张量，形状: {indices_tensor.shape}")
    except Exception as e:
        print(f"转换为张量失败: {e}")

def test_data_extraction():
    """测试从f_v_batch中提取数据"""
    print(f"\n=== 测试数据提取 ===")
    
    batch_size = 3
    target_frames = 10
    
    torch.manual_seed(456)
    f_v_batch = torch.randn(batch_size, 50, 256)  # [batch_size, seq_len, feature_dim]
    f_text_batch = torch.randn(batch_size, 256)
    
    # 获取采样索引
    indices_batch = iterative_sampling(f_v_batch, f_text_batch, 5, 2, 0.5, 0.5, target_frames=target_frames)
    
    print(f"原始数据形状: {f_v_batch.shape}")
    print(f"目标帧数: {target_frames}")
    
    # 方法1: 使用列表推导式逐个提取
    selected_frames_list = []
    for i, indices in enumerate(indices_batch):
        selected = f_v_batch[i, indices]  # 从第i个样本中选择对应索引的帧
        selected_frames_list.append(selected)
        print(f"批次{i}: 索引{indices[:5]}..., 提取形状: {selected.shape}")
    
    # 方法2: 如果所有批次长度一致，可以直接堆叠
    if all(len(indices) == target_frames for indices in indices_batch):
        # 转换为张量并批量索引
        indices_tensor = torch.tensor(indices_batch)  # [batch_size, target_frames]
        
        # 使用高级索引提取数据
        batch_indices = torch.arange(batch_size).unsqueeze(1)  # [batch_size, 1]
        selected_frames_tensor = f_v_batch[batch_indices, indices_tensor]  # [batch_size, target_frames, feature_dim]
        
        print(f"\n批量提取结果形状: {selected_frames_tensor.shape}")
        
        # 验证两种方法结果一致
        stacked_list = torch.stack(selected_frames_list)
        is_equal = torch.allclose(selected_frames_tensor, stacked_list)
        print(f"两种方法结果一致: {'✓' if is_equal else '✗'}")
        
        return selected_frames_tensor
    else:
        print("批次长度不一致，只能使用列表方式")
        return selected_frames_list

def demonstrate_usage():
    """演示实际使用场景"""
    print(f"\n=== 实际使用演示 ===")
    
    # 模拟实际场景
    batch_size = 4
    original_seq_len = 100
    feature_dim = 512
    target_frames = 20
    
    torch.manual_seed(789)
    f_v_batch = torch.randn(batch_size, original_seq_len, feature_dim)
    f_text_batch = torch.randn(batch_size, feature_dim)
    
    print(f"输入视频特征: {f_v_batch.shape}")
    print(f"输入文本特征: {f_text_batch.shape}")
    
    # 执行采样
    indices_batch = iterative_sampling(f_v_batch, f_text_batch, 10, 3, 0.5, 0.5, target_frames=target_frames)
    
    # 提取选中的帧
    indices_tensor = torch.tensor(indices_batch)
    batch_indices = torch.arange(batch_size).unsqueeze(1)
    selected_frames = f_v_batch[batch_indices, indices_tensor]
    
    print(f"采样后视频特征: {selected_frames.shape}")
    print(f"压缩比: {original_seq_len}/{target_frames} = {original_seq_len/target_frames:.1f}x")
    
    # 可以继续用于后续处理
    # 例如：计算与文本特征的相似度
    # similarity = torch.cosine_similarity(selected_frames, f_text_batch.unsqueeze(1), dim=-1)
    # print(f"相似度矩阵形状: {similarity.shape}")
    
    return selected_frames

def test_data_extraction():
    """测试从f_v_batch中提取数据"""
    print(f"\n=== 测试数据提取 ===")
    
    batch_size = 3
    target_frames = 10
    
    torch.manual_seed(456)
    f_v_batch = torch.randn(batch_size, 50, 256)  # [batch_size, seq_len, feature_dim]
    f_text_batch = torch.randn(batch_size, 256)
    
    # 获取采样索引
    indices_batch = iterative_sampling(f_v_batch, f_text_batch, 5, 2, 0.5, 0.5, target_frames=target_frames)
    
    print(f"原始数据形状: {f_v_batch.shape}")
    print(f"目标帧数: {target_frames}")
    
    # 方法1: 使用列表推导式逐个提取
    selected_frames_list = []
    for i, indices in enumerate(indices_batch):
        selected = f_v_batch[i, indices]  # 从第i个样本中选择对应索引的帧
        selected_frames_list.append(selected)
        print(f"批次{i}: 索引{indices[:5]}..., 提取形状: {selected.shape}")
    
    # 方法2: 如果所有批次长度一致，可以直接堆叠
    if all(len(indices) == target_frames for indices in indices_batch):
        # 转换为张量并批量索引
        indices_tensor = torch.tensor(indices_batch)  # [batch_size, target_frames]
        
        # 使用高级索引提取数据
        batch_indices = torch.arange(batch_size).unsqueeze(1)  # [batch_size, 1]
        selected_frames_tensor = f_v_batch[batch_indices, indices_tensor]  # [batch_size, target_frames, feature_dim]
        
        print(f"\n批量提取结果形状: {selected_frames_tensor.shape}")
        
        # 验证两种方法结果一致
        stacked_list = torch.stack(selected_frames_list)
        is_equal = torch.allclose(selected_frames_tensor, stacked_list)
        print(f"两种方法结果一致: {'✓' if is_equal else '✗'}")
        
        return selected_frames_tensor
    else:
        print("批次长度不一致，只能使用列表方式")
        return selected_frames_list

def demonstrate_usage():
    """演示实际使用场景"""
    print(f"\n=== 实际使用演示 ===")
    
    # 模拟实际场景
    batch_size = 4
    original_seq_len = 100
    feature_dim = 512
    target_frames = 20
    
    torch.manual_seed(789)
    f_v_batch = torch.randn(batch_size, original_seq_len, feature_dim)
    f_text_batch = torch.randn(batch_size, feature_dim)
    
    print(f"输入视频特征: {f_v_batch.shape}")
    print(f"输入文本特征: {f_text_batch.shape}")
    
    # 执行采样
    indices_batch = iterative_sampling(f_v_batch, f_text_batch, 10, 3, 0.5, 0.5, target_frames=target_frames)
    
    # 提取选中的帧
    indices_tensor = torch.tensor(indices_batch)
    batch_indices = torch.arange(batch_size).unsqueeze(1)
    selected_frames = f_v_batch[batch_indices, indices_tensor]
    
    print(f"采样后视频特征: {selected_frames.shape}")
    print(f"压缩比: {original_seq_len}/{target_frames} = {original_seq_len/target_frames:.1f}x")
    
    # 可以继续用于后续处理
    # 例如：计算与文本特征的相似度
    # similarity = torch.cosine_similarity(selected_frames, f_text_batch.unsqueeze(1), dim=-1)
    # print(f"相似度矩阵形状: {similarity.shape}")
    
    return selected_frames

if __name__ == "__main__":
    test_fixed_frame_output()
    test_edge_cases()
    test_batch_consistency()
    test_data_extraction()
    demonstrate_usage()
    test_data_extraction()
    demonstrate_usage()
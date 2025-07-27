import torch
import torch.nn.functional as F

def process_numbers(number_list):
    """
    对数字列表进行去重和排序
    """
    unique_numbers = sorted(set(int(num) for num in number_list))
    return unique_numbers

def pad_or_truncate_indices(indices, target_count):
    """
    将索引列表填充或截断到目标长度
    """
    if len(indices) >= target_count:
        # 如果帧数过多，均匀采样到目标数量
        step = len(indices) / target_count
        selected_indices = [indices[int(i * step)] for i in range(target_count)]
        return selected_indices
    else:
        # 如果帧数不足，重复最后几帧
        result = indices.copy()
        while len(result) < target_count:
            if len(indices) > 0:
                result.append(indices[-1])
            else:
                result.append(0)
        return result[:target_count]

def sample_vectors_single(f_v, k):
    """单序列采样函数（与原版一致）"""
    t = f_v.size(0)
    indices = torch.linspace(0, t - 1, steps=k).long()
    sampled_vectors = f_v[indices]
    return sampled_vectors, indices

def cosine_similarity_single(a, b):
    """单序列余弦相似度（与原版一致）"""
    return F.cosine_similarity(a, b, dim=-1)

def calculate_attention_single(f_qst, f_v_sampled):
    """单序列注意力计算（与原版一致）"""
    q = f_qst.unsqueeze(0)  # Shape: [1, d]
    k = f_v_sampled         # Shape: [k, d] or [1, d]
    v = f_v_sampled         # Shape: [k, d] or [1, d]

    d = f_qst.size(-1)
    att_weights = F.softmax(torch.matmul(q, k.T) / (d ** 0.5), dim=-1)
    att_f_v = torch.matmul(att_weights, v)  # Shape: [1, d] or [1, k, d]
    return att_f_v.squeeze(0)  # Shape: [d] or [k, d]

def iterative_sampling_single(f_v, f_text, k, m, a1, a2):
    """
    单序列迭代采样（与原版完全一致的逻辑）
    """
    t, d = f_v.shape
    indices_record = []
    iter_samples = [0]

    for _ in range(m):
        # 采样k个向量
        f_v_sampled, sampled_indices = sample_vectors_single(f_v, k)
        sampled_indices += sum(iter_samples)

        # 计算sim1（时间一致性）
        sim1 = cosine_similarity_single(f_v_sampled[:-1], f_v_sampled[1:])
        sim1 = torch.cat([sim1, sim1[-1].unsqueeze(0)])

        # 计算注意力向量（这里保持原版逻辑，即使它是冗余的）
        att_f_v = torch.stack([
            calculate_attention_single(f_text, f_v_sampled[i].unsqueeze(0)) 
            for i in range(k)
        ])

        # 计算sim2（注意力向量间的相似性）
        sim2 = cosine_similarity_single(att_f_v[:-1], att_f_v[1:])
        sim2 = torch.cat([sim2, sim2[-1].unsqueeze(0)])

        # 组合相似度并找到最佳索引
        sim = a1 * sim1 + a2 * sim2
        max_sim_index = torch.argmax(sim)

        # 更新范围
        center_idx = sampled_indices[max_sim_index]
        start_idx = max(0, center_idx - t // k)
        end_idx = min(t, center_idx + t // k)
        iter_samples.append(start_idx)
        f_v = f_v[start_idx:end_idx]
        indices_record += list(sampled_indices)

    indices_record = process_numbers(indices_record)
    return indices_record

def iterative_sampling(f_v, f_text, k, m, a1, a2, target_frames=None):
    """
    正确的批量迭代采样：每个批次项完全独立处理
    
    Args:
        f_v (torch.Tensor): 视频特征。形状: [batch_size, t, d] 或 [t, d]
        f_text (torch.Tensor): 文本特征。形状: [batch_size, d] 或 [d]
        k (int): 每次迭代中采样的向量数
        m (int): 迭代次数
        a1 (float): 时间一致性得分的权重
        a2 (float): 文本相关性得分的权重
        target_frames (int, optional): 目标帧数
    
    Returns:
        list[list[int]] 或 list[int]: 选择的索引
    """
    # 处理单序列情况
    is_single_sequence = f_v.dim() == 2
    if is_single_sequence:
        result = iterative_sampling_single(f_v, f_text, k, m, a1, a2)
        if target_frames is not None:
            result = pad_or_truncate_indices(result, target_frames)
        return result

    # 批量处理：每个序列完全独立
    batch_size = f_v.size(0)
    batch_results = []
    
    for i in range(batch_size):
        # 提取单个序列
        f_v_single = f_v[i]      # [t, d]
        f_text_single = f_text[i]  # [d]
        
        # 使用原版逻辑处理
        result = iterative_sampling_single(f_v_single, f_text_single, k, m, a1, a2)
        
        # 如果指定了目标帧数，进行调整
        if target_frames is not None:
            result = pad_or_truncate_indices(result, target_frames)
        
        batch_results.append(result)
    
    return batch_results

def iterative_sampling_parallel(f_v, f_text, k, m, a1, a2, target_frames=None):
    """
    并行版本：利用PyTorch的并行能力，但保持逻辑独立性
    注意：这个版本在概念上是并行的，但实际实现仍然是顺序的
    真正的并行需要更复杂的实现
    """
    return iterative_sampling(f_v, f_text, k, m, a1, a2, target_frames)

# 为了向后兼容，提供一个简化的接口
def iterative_sampling_with_fixed_output(f_v, f_text, k, m, a1, a2, target_frames):
    """
    便捷函数：确保输出固定数量的帧
    """
    return iterative_sampling(f_v, f_text, k, m, a1, a2, target_frames=target_frames)
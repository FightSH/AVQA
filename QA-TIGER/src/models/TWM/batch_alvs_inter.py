import torch
import torch.nn.functional as F

def process_numbers(number_list):
    """
    对数字列表进行去重和排序
    """
    unique_numbers = sorted(set(int(num) for num in number_list))
    return unique_numbers

def _adjust_indices_to_target_length(indices: list[int], target_length: int) -> list[int]:
    """
    将单个索引列表调整到目标长度。

    Args:
        indices (list[int]): 输入的、经过排序和去重的索引列表。
        target_length (int): 目标长度。

    Returns:
        list[int]: 长度为 target_length 的索引列表。
    """
    current_length = len(indices)

    if current_length == target_length:
        return indices

    # Case 1: 帧数过多，需要均匀下采样
    if current_length > target_length:
        # 使用 linspace 创建均匀分布的采样点
        sampler = torch.linspace(0, current_length - 1, steps=target_length).long()
        return [indices[i] for i in sampler]

    # Case 2: 帧数不足，需要重复填充
    if current_length < target_length:
        # 如果原始列表为空，则用0填充
        if not indices:
            return [0] * target_length
        
        num_to_pad = target_length - current_length
        last_index = indices[-1]
        padding = [last_index] * num_to_pad
        return indices + padding

def iterative_sampling_vectorized(f_v, f_text, k, m, a1, a2, target_num_frames: int | None = None):
    """
    向量化的迭代采样（注意：结果与原版不同，这是一个算法变体）
    """
    """
    执行真正的批量迭代采样过程。
    此版本是完全向量化的，并假定批次中的所有序列具有相同的初始长度。

    Args:
        f_v (torch.Tensor): 一批视频特征。形状: [batch_size, t, d]
        f_text (torch.Tensor): 一批文本特征。形状: [batch_size, d]
        k (int): 每次迭代中采样的向量数。
        m (int): 迭代次数。
        a1 (float): 时间一致性得分 (sim1) 的权重。
        a2 (float): 文本相关性得分 (sim2) 的权重。
        target_num_frames (int | None, optional): 最终输出的目标帧数。
            如果提供，将对结果进行采样或填充。默认为 None。

    Returns:
        list[list[int]]: 一个列表的列表，其中每个内部列表包含为该批次项选择的索引。
                         如果提供了 target_num_frames，则所有内部列表的长度都将是该值。
                         如果输入是单个序列，则返回一个单一的列表。
    """
    # 通过 unsqueeze 处理单个序列的情况，以统一批处理逻辑
    is_single_sequence = f_v.dim() == 2
    if is_single_sequence:
        f_v = f_v.unsqueeze(0)
        f_text = f_text.unsqueeze(0)

    batch_size, t, d = f_v.shape
    device = f_v.device

    # 为批次中的每个项目维护状态
    start_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    end_indices = torch.full((batch_size,), t, dtype=torch.long, device=device)
    
    # 记录每个批次项所有采样到的索引
    batch_indices_record = [[] for _ in range(batch_size)]

    for _ in range(m):
        # 1. 从每个批次项的当前范围内采样 k 个向量
        current_lengths = end_indices - start_indices
        indices = torch.linspace(0, 1, steps=k, device=device).unsqueeze(0) * (current_lengths - 1).clamp(min=0).unsqueeze(1)
        indices = start_indices.unsqueeze(1) + indices.long()

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d)
        f_v_sampled = torch.gather(f_v, 1, indices_expanded)

        for i in range(batch_size):
            batch_indices_record[i].extend(indices[i].cpu().tolist())

        # 2. 计算 sim1 (时间一致性)
        sim1 = F.cosine_similarity(f_v_sampled[:, :-1], f_v_sampled[:, 1:], dim=2)
        sim1 = F.pad(sim1, (0, 1), 'replicate')

        # 3. 计算 sim2 (文本相关性)
        sim2 = F.cosine_similarity(f_v_sampled, f_text.unsqueeze(1), dim=2)

        # 4. 组合相似度并为每个批次项找到最佳索引
        sim = a1 * sim1 + a2 * sim2
        max_sim_indices = torch.argmax(sim, dim=1)

        # 5. 为下一次迭代更新开始和结束范围
        center_indices = torch.gather(indices, 1, max_sim_indices.unsqueeze(1)).squeeze(1)
        window_half_size = current_lengths // k
        start_indices = (center_indices - window_half_size).clamp(min=0)
        end_indices = (center_indices + window_half_size).clamp(max=t)

    # 6. 处理记录的索引以生成最终输出
    final_indices = [process_numbers(rec) for rec in batch_indices_record]

    # 7. (新增) 如果指定了目标帧数，则对齐长度
    if target_num_frames is not None:
        final_indices = [_adjust_indices_to_target_length(idx, target_num_frames) for idx in final_indices]

    # 如果输入是单个序列，则返回一个扁平化的列表
    if is_single_sequence:
        return final_indices[0]
    
    return final_indices

def _safe_iterative_sampling_single(f_v, f_text, k, m, a1, a2):
    """
    安全的单序列迭代采样，处理边界情况
    """
    t, d = f_v.shape
    
    # 边界检查
    if t == 0:
        return []
    if k >= t:
        # 如果k大于等于序列长度，返回所有索引
        return list(range(t))
    
    indices_record = []
    iter_samples = [0]
    
    for iteration in range(m):
        current_t = f_v.size(0)
        if current_t == 0:
            break
            
        # 调整k以适应当前序列长度
        actual_k = min(k, current_t)
        
        # 采样向量
        indices = torch.linspace(0, current_t - 1, steps=actual_k).long()
        f_v_sampled = f_v[indices]
        sampled_indices = indices + sum(iter_samples)
        
        # 计算sim1
        if actual_k > 1:
            sim1 = F.cosine_similarity(f_v_sampled[:-1], f_v_sampled[1:], dim=-1)
            sim1 = torch.cat([sim1, sim1[-1].unsqueeze(0)])
        else:
            sim1 = torch.tensor([0.0])
        
        # 计算注意力向量
        att_f_v = torch.stack([
            _calculate_attention_single(f_text, f_v_sampled[i].unsqueeze(0))
            for i in range(actual_k)
        ])
        
        # 计算sim2
        if actual_k > 1:
            sim2 = F.cosine_similarity(att_f_v[:-1], att_f_v[1:], dim=-1)
            sim2 = torch.cat([sim2, sim2[-1].unsqueeze(0)])
        else:
            sim2 = torch.tensor([0.0])
        
        # 组合相似度
        sim = a1 * sim1 + a2 * sim2
        max_sim_index = torch.argmax(sim)
        
        # 更新范围
        center_idx = sampled_indices[max_sim_index]
        window_size = max(1, t // k)  # 使用原始长度计算窗口
        start_idx = max(0, center_idx - window_size)
        end_idx = min(current_t, center_idx + window_size)
        
        # 确保新范围有效
        if start_idx >= end_idx:
            break
            
        iter_samples.append(start_idx)
        f_v = f_v[start_idx:end_idx]
        indices_record.extend(sampled_indices.tolist())
    
    return process_numbers(indices_record)

def _calculate_attention_single(f_qst, f_v_sampled):
    """单序列注意力计算"""
    q = f_qst.unsqueeze(0)  # Shape: [1, d]
    k = f_v_sampled         # Shape: [1, d] or [k, d]
    v = f_v_sampled         # Shape: [1, d] or [k, d]

    d = f_qst.size(-1)
    att_weights = F.softmax(torch.matmul(q, k.T) / (d ** 0.5), dim=-1)
    att_f_v = torch.matmul(att_weights, v)  # Shape: [1, d] or [1, k, d]
    return att_f_v.squeeze(0)  # Shape: [d] or [k, d]

def iterative_sampling(f_v, f_text, k, m, a1, a2, target_num_frames: int | None = None):
    """
    正确的批量迭代采样：每个批次项完全独立处理，保证与原版结果一致
    
    如果需要使用向量化版本（结果不同但可能更快），请调用 iterative_sampling_vectorized
    """
    # 处理单序列情况
    is_single_sequence = f_v.dim() == 2
    if is_single_sequence:
        result = _safe_iterative_sampling_single(f_v, f_text, k, m, a1, a2)
        if target_num_frames is not None:
            result = _adjust_indices_to_target_length(result, target_num_frames)
        return result

    # 批量处理：每个序列完全独立
    batch_size = f_v.size(0)
    batch_results = []
    
    for i in range(batch_size):
        # 提取单个序列
        f_v_single = f_v[i]      # [t, d]
        f_text_single = f_text[i]  # [d]
        
        # 使用安全的单序列处理
        result = _safe_iterative_sampling_single(f_v_single, f_text_single, k, m, a1, a2)
        
        # 如果指定了目标帧数，进行调整
        if target_num_frames is not None:
            result = _adjust_indices_to_target_length(result, target_num_frames)
        
        batch_results.append(result)
    
    return batch_results

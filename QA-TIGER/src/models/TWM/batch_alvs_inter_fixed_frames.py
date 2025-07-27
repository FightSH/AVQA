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
    Args:
        indices: 索引列表
        target_count: 目标帧数
    Returns:
        固定长度的索引列表
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
            # 重复最后一帧，或者重复整个序列
            if len(indices) > 0:
                result.append(indices[-1])
            else:
                result.append(0)  # 极端情况的默认值
        return result[:target_count]

def iterative_sampling(f_v, f_text, k, m, a1, a2, target_frames=None):
    """
    执行批量迭代采样过程，支持固定帧数输出。
    
    Args:
        f_v (torch.Tensor): 一批视频特征。形状: [batch_size, t, d]
        f_text (torch.Tensor): 一批文本特征。形状: [batch_size, d]
        k (int): 每次迭代中采样的向量数。
        m (int): 迭代次数。
        a1 (float): 时间一致性得分 (sim1) 的权重。
        a2 (float): 文本相关性得分 (sim2) 的权重。
        target_frames (int, optional): 目标帧数。如果指定，所有批次将输出相同数量的帧。
    
    Returns:
        list[list[int]]: 一个列表的列表，每个内部列表包含选择的索引。
                         如果指定了target_frames，所有列表长度相同。
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

    for iteration in range(m):
        # 1. 从每个批次项的当前范围内采样 k 个向量
        current_lengths = end_indices - start_indices
        
        # 检查是否有序列长度为0
        if torch.any(current_lengths <= 0):
            print(f"警告：第{iteration}次迭代时有序列长度为0，提前结束")
            break
        
        # 使用归一化的 linspace 来生成采样步长，然后缩放并偏移
        indices = torch.linspace(0, 1, steps=k, device=device).unsqueeze(0) * (current_lengths - 1).clamp(min=0).unsqueeze(1)
        indices = start_indices.unsqueeze(1) + indices.long()

        # 使用 gather 根据索引获取采样向量
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d)
        f_v_sampled = torch.gather(f_v, 1, indices_expanded)  # 形状: [B, k, D]

        # 记录刚刚采样的绝对索引
        for i in range(batch_size):
            batch_indices_record[i].extend(indices[i].cpu().tolist())

        # 2. 计算 sim1 (时间一致性)
        if k > 1:
            sim1 = F.cosine_similarity(f_v_sampled[:, :-1], f_v_sampled[:, 1:], dim=2)  # 形状: [B, k-1]
            # 复制最后一个元素以得到形状 [B, k]
            sim1 = F.pad(sim1, (0, 1), 'replicate')
        else:
            sim1 = torch.zeros(batch_size, 1, device=device)

        # 3. 计算 sim2 (文本相关性)
        sim2 = F.cosine_similarity(f_v_sampled, f_text.unsqueeze(1), dim=2)  # 形状: [B, k]

        # 4. 组合相似度并为每个批次项找到最佳索引
        sim = a1 * sim1 + a2 * sim2  # 形状: [B, k]
        max_sim_indices = torch.argmax(sim, dim=1)  # 形状: [B]

        # 5. 为下一次迭代更新开始和结束范围
        center_indices = torch.gather(indices, 1, max_sim_indices.unsqueeze(1)).squeeze(1)  # 形状: [B]
        
        # 窗口大小根据当前范围长度进行缩放
        window_half_size = current_lengths // k
        window_half_size = window_half_size.clamp(min=1)  # 确保窗口大小至少为1
        
        # 计算新的开始和结束索引
        start_indices = (center_indices - window_half_size).clamp(min=0)
        end_indices = (center_indices + window_half_size).clamp(max=t)
        
        # 确保新范围不为空
        invalid_ranges = start_indices >= end_indices
        if torch.any(invalid_ranges):
            print(f"警告：第{iteration}次迭代时有无效范围，调整...")
            end_indices[invalid_ranges] = start_indices[invalid_ranges] + 1

    # 6. 处理记录的索引以生成最终输出
    final_indices = [process_numbers(rec) for rec in batch_indices_record]
    
    # 7. 如果指定了目标帧数，进行填充或截断
    if target_frames is not None:
        final_indices = [pad_or_truncate_indices(indices, target_frames) for indices in final_indices]

    # 如果输入是单个序列，则返回一个扁平化的列表
    if is_single_sequence:
        return final_indices[0]
    
    return final_indices

def iterative_sampling_with_fixed_output(f_v, f_text, k, m, a1, a2, target_frames):
    """
    便捷函数：确保输出固定数量的帧
    """
    return iterative_sampling(f_v, f_text, k, m, a1, a2, target_frames=target_frames)
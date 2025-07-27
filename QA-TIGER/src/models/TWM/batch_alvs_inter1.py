import torch
import torch.nn.functional as F


def process_numbers(number_list):
    """
    对数字列表进行去重和排序
    """
    unique_numbers = sorted(set(int(num) for num in number_list))
    return unique_numbers


def iterative_sampling(f_v, f_text, k, m, a1, a2):
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

    Returns:
        list[list[int]]: 一个列表的列表，其中每个内部列表包含为该批次项选择的唯一排序索引。
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
        # 创建一个 [B, k] 的索引张量
        current_lengths = end_indices - start_indices
        # 使用归一化的 linspace 来生成采样步长，然后缩放并偏移
        indices = torch.linspace(0, 1, steps=k, device=device).unsqueeze(0) * (current_lengths - 1).clamp(
            min=0).unsqueeze(1)
        indices = start_indices.unsqueeze(1) + indices.long()

        # 使用 gather 根据索引获取采样向量
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d)
        f_v_sampled = torch.gather(f_v, 1, indices_expanded)  # 形状: [B, k, D]

        # 记录刚刚采样的绝对索引
        for i in range(batch_size):
            batch_indices_record[i].extend(indices[i].cpu().tolist())

        # 2. 计算 sim1 (时间一致性)
        sim1 = F.cosine_similarity(f_v_sampled[:, :-1], f_v_sampled[:, 1:], dim=2)  # 形状: [B, k-1]
        # 复制最后一个元素以得到形状 [B, k]
        sim1 = F.pad(sim1, (0, 1), 'replicate')

        # 3. 计算 sim2 (文本相关性) - 已修复并优化的逻辑
        # 这是对原始失效的注意力逻辑的修正，直接计算采样帧和文本的相似度。
        sim2 = F.cosine_similarity(f_v_sampled, f_text.unsqueeze(1), dim=2)  # 形状: [B, k]

        # 4. 组合相似度并为每个批次项找到最佳索引
        sim = a1 * sim1 + a2 * sim2  # 形状: [B, k]
        max_sim_indices = torch.argmax(sim, dim=1)  # 形状: [B]

        # 5. 为下一次迭代更新开始和结束范围
        # 获取与最大相似度得分对应的在原始 f_v 中的实际索引
        center_indices = torch.gather(indices, 1, max_sim_indices.unsqueeze(1)).squeeze(1)  # 形状: [B]

        # 窗口大小根据当前范围长度进行缩放
        window_half_size = current_lengths // k

        # 计算新的开始和结束索引，并确保它们在原始边界 [0, t] 内
        start_indices = (center_indices - window_half_size).clamp(min=0)
        end_indices = (center_indices + window_half_size).clamp(max=t)

    # 6. 处理记录的索引以生成最终输出
    final_indices = [process_numbers(rec) for rec in batch_indices_record]

    # 如果输入是单个序列，则返回一个扁平化的列表
    if is_single_sequence:
        return final_indices[0]

    return final_indices
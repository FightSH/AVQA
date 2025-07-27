import torch
import torch.nn.functional as F

def sample_vectors(f_v, k):
    """
    Uniformly sample k vectors from f_v.
    Args:
        f_v: [batch_size, t, d] or [t, d]
        k: int
    Returns:
        sampled_vectors: [batch_size, k, d] or [k, d]
        indices: [k] (same for all batches)
    """
    if f_v.dim() == 2:
        # Single sequence case
        t = f_v.size(0)
        indices = torch.linspace(0, t - 1, steps=k).long()
        sampled_vectors = f_v[indices]
        return sampled_vectors, indices
    else:
        # Batch case
        batch_size, t, d = f_v.shape
        indices = torch.linspace(0, t - 1, steps=k).long()
        sampled_vectors = f_v[:, indices, :]  # [batch_size, k, d]
        return sampled_vectors, indices

def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two tensors a and b.
    Args:
        a, b: [batch_size, ...] or [...]
    """
    return F.cosine_similarity(a, b, dim=-1)

def calculate_attention(f_qst, f_v_sampled):
    """
    Calculate attention values - 修复版本，保持与原版一致的语义
    Args:
        f_qst: [batch_size, d] or [d]
        f_v_sampled: [batch_size, k, d] or [k, d]
    Returns:
        att_f_v: [batch_size, d] or [d] - 返回加权平均后的单个向量
    """
    if f_qst.dim() == 1:
        # Single sequence case - 与原版完全一致
        q = f_qst.unsqueeze(0)  # Shape: [1, d]
        k = f_v_sampled         # Shape: [k, d]
        v = f_v_sampled         # Shape: [k, d]
        d = f_qst.size(-1)
        att_weights = F.softmax(torch.matmul(q, k.T) / (d ** 0.5), dim=-1)
        att_f_v = torch.matmul(att_weights, v)  # Shape: [1, d]
        return att_f_v.squeeze(0)  # Shape: [d]
    else:
        # Batch case - 批量计算加权平均
        batch_size, k, d = f_v_sampled.shape
        q = f_qst.unsqueeze(1)  # Shape: [batch_size, 1, d]
        k = f_v_sampled         # Shape: [batch_size, k, d]
        v = f_v_sampled         # Shape: [batch_size, k, d]
        att_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5), dim=-1)
        att_f_v = torch.matmul(att_weights, v)  # Shape: [batch_size, 1, d]
        return att_f_v.squeeze(1)  # Shape: [batch_size, d]

def process_numbers(number_list):
    """
    Process numbers - unchanged from original
    """
    unique_numbers = sorted(set(int(num) for num in number_list))
    return unique_numbers

def iterative_sampling(f_v, f_text, k, m, a1, a2):
    """
    Perform the iterative sampling process with batch support.
    Args:
        f_v: [batch_size, t, d] or [t, d]
        f_text: [batch_size, d] or [d]
        k, m: int
        a1, a2: float
    Returns:
        indices_record: list of indices (single sequence) or list of lists (batch)
    """
    if f_v.dim() == 2:
        # Single sequence case - use original logic
        return _iterative_sampling_single(f_v, f_text, k, m, a1, a2)
    else:
        # Batch case - each batch gets independent processing
        return _iterative_sampling_batch(f_v, f_text, k, m, a1, a2)

def _iterative_sampling_single(f_v, f_text, k, m, a1, a2):
    """
    Single sequence iterative sampling (修复版本，保持与原版逻辑一致)
    """
    t, d = f_v.shape
    indices_record = []
    iter_samples = [0]

    for _ in range(m):
        f_v_sampled, sampled_indices = sample_vectors(f_v, k)
        sampled_indices += sum(iter_samples)

        # 计算连续向量间的相似度
        sim1 = cosine_similarity(f_v_sampled[:-1], f_v_sampled[1:])
        sim1 = torch.cat([sim1, sim1[-1].unsqueeze(0)])

        # 修复：对每个采样向量计算注意力，得到k个注意力向量
        att_f_v = torch.stack([
            calculate_attention(f_text, f_v_sampled[i].unsqueeze(0)) 
            for i in range(k)
        ])  # Shape: [k, d]

        # 计算连续注意力向量间的相似度
        sim2 = cosine_similarity(att_f_v[:-1], att_f_v[1:])
        sim2 = torch.cat([sim2, sim2[-1].unsqueeze(0)])

        sim = a1 * sim1 + a2 * sim2
        max_sim_index = torch.argmax(sim)

        center_idx = sampled_indices[max_sim_index]
        start_idx = max(0, center_idx - t // k)
        end_idx = min(t, center_idx + t // k)
        iter_samples.append(start_idx)
        f_v = f_v[start_idx:end_idx]
        indices_record += list(sampled_indices)

    indices_record = process_numbers(indices_record)
    return indices_record

def _iterative_sampling_batch(f_v, f_text, k, m, a1, a2):
    """
    Batch iterative sampling - each batch has independent sampling process
    """
    batch_size, t, d = f_v.shape
    batch_indices_records = []

    # Process each batch independently
    for batch_idx in range(batch_size):
        f_v_single = f_v[batch_idx]  # [t, d]
        f_text_single = f_text[batch_idx]  # [d]

        # Use single sequence processing for each batch
        indices_record = _iterative_sampling_single(f_v_single, f_text_single, k, m, a1, a2)
        batch_indices_records.append(indices_record)

    return batch_indices_records  # List of lists, one for each batch
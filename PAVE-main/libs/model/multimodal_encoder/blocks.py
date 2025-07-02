# This script hold all the implementation of transformer or ResNet blocks.

import copy
import ipdb
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import math

from libs.mm_utils import split_list_lengths

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input
except ImportError:
    flash_attn_varlen_func = None


def FeedForward(dim, dim_feedforward):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim_feedforward, bias = False),
        nn.GELU(),
        nn.Linear(dim_feedforward, dim, bias = False)
    )


def create_idx_cube(T, H, W):
    # create the T index
    t_values = torch.arange(T).unsqueeze(1).unsqueeze(2)
    # Broadcast the depth values across the height and width dimensions
    t_index = t_values.expand(T, H, W)
    
    # create the H index
    h_values = torch.arange(H).unsqueeze(0).unsqueeze(2)
    h_index = h_values.expand(T, H, W)
    
    # create the W index
    w_values = torch.arange(W).unsqueeze(0).unsqueeze(0)
    w_index = w_values.expand(T, H, W)
    
    # create a T, H, W, 3 cube
    final_idx = torch.stack([t_index, h_index, w_index], dim=-1)
    return final_idx

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(input, cos, sin, position_ids, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).
        This function is base modified base on the function from the Qwen2-VL
    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.                          torch.Size([1, 4, 985, 128])
        k (`torch.Tensor`): The key tensor.                            torch.Size([1, 4, 985, 128])
        cos (`torch.Tensor`): The cosine part of the rotary embedding. torch.Size([985, 128])
        sin (`torch.Tensor`): The sine part of the rotary embedding.   torch.Size([985, 128])
        position_ids (`torch.Tensor`):                                 torch.Size([3, 1, 985])
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):                                    [16, 24, 24]
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # select poe for each dimension, since we have 3 position_ids
    
    cos = cos[position_ids] # torch.Size([3, 1, 985, 128])
    sin = sin[position_ids] # torch.Size([3, 1, 985, 128])
    mrope_section = mrope_section * 2 # [16, 24, 24] -> [16, 24, 24, 16, 24, 24]
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    ) # torch.Size([1, 1, 985, 128]) target shape in qwen vl
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    ) # torch.Size([1, 1, 985, 128]) target shape in qwen vl

    input_embed = (input * cos) + (rotate_half(input) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    return input_embed


class VideoCrossAttentionWith3DRope(nn.Module):
    '''
        This version aims to handle the cross-attn between the query and the video feature
    '''
    
    def __init__(self, d_query, d_input, d_model, num_heads, attn_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_query == d_model

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_query, d_model)
        self.k_linear = nn.Linear(d_input, d_model)
        self.v_linear = nn.Linear(d_input, d_model)
        self.attn_drop = attn_drop
        self.rope_scaling = {}
        self.rope_scaling["mrope_section"] = [16, 24, 24]

    def forward(self, q, k, v, attn_mask=None, rope=None, rope_axis='time'):
        # q:    (B, T_q, S_q, C) should be the query tokens
        # k:    (B, T_k, S_k, C) should be the video tokens
        # v:    (B, T_k, S_k, C) should be the video tokens
        # attn_mask: is list of the number which indicate the length of the  or value
        # rope: rotary positional embedding
        
        print("\n=== VideoCrossAttentionWith3DRope Forward 开始 ===")
        
        # 解析输入维度
        Bq, T_q, S_q, C = q.shape
        Bk, T_k, S_k, C_cond = v.shape
        Bv, T_k, S_k, C_cond = k.shape
        
        print(f"输入维度统计:")
        print(f"  - 查询q: {q.shape} (B={Bq}, T_q={T_q}, S_q={S_q}, C={C})")
        print(f"  - 键k: {k.shape} (B={Bk}, T_k={T_k}, S_k={S_k}, C_cond={C_cond})")
        print(f"  - 值v: {v.shape} (B={Bv}, T_k={T_k}, S_k={S_k}, C_cond={C_cond})")
        print(f"  - rope_axis: {rope_axis}")
        print(f"  - attn_mask: {attn_mask}")
        
        assert Bq == Bk == Bv == 1, "当前版本只支持batch_size=1" # for current version, we only support bs = 1
        print(f"  - 验证通过：批次大小均为1")
        
        print(f"\n第一步：线性变换和维度重排")
        # 线性变换和维度重排
        print(f"  对查询q进行线性变换:")
        print(f"    变换前: {q.shape}")
        q = self.q_linear(q).view(Bq, T_q*S_q, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, T_q*S_q, H_dim)
        print(f"    变换后: {q.shape} (B, num_heads={self.num_heads}, T_q*S_q={T_q*S_q}, head_dim={self.head_dim})")
        
        print(f"  对键k进行线性变换:")
        print(f"    变换前: {k.shape}")
        k = self.k_linear(k).view(Bk, T_k*S_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, T_k*S_k, H_dim)
        print(f"    变换后: {k.shape} (B, num_heads={self.num_heads}, T_k*S_k={T_k*S_k}, head_dim={self.head_dim})")
        
        print(f"  对值v进行线性变换:")
        print(f"    变换前: {v.shape}")
        v = self.v_linear(v).view(Bv, T_k*S_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, T_k*S_k, H_dim)
        print(f"    变换后: {v.shape} (B, num_heads={self.num_heads}, T_k*S_k={T_k*S_k}, head_dim={self.head_dim})")
        
        # add the rotary pos embedding
        print(f"\n第二步：旋转位置编码(RoPE)处理")
        if rope is not None: # expect in the format of # (batch, heads, seq len, dimension of head)
            print(f"  检测到RoPE，开始应用3D旋转位置编码")
            # compute the max size 
            T_max = max(T_q, T_k)
            print(f"  最大时间维度: T_max = max({T_q}, {T_k}) = {T_max}")
            
            assert int(math.sqrt(S_q)) ** 2 == S_q, "假设查询的空间维度为正方形" # In here we assume we alway use the square
            assert int(math.sqrt(S_k)) ** 2 == S_k, "假设键值的空间维度为正方形" # In here we assume we alway use the square
            
            H_max = int(math.sqrt(max(S_q, S_k)))
            W_max = H_max
            print(f"  最大空间维度: H_max = W_max = sqrt(max({S_q}, {S_k})) = {H_max}")
            
            # create the poe
            print(f"  生成RoPE的cos和sin值，序列长度: {max(T_max, H_max, W_max)}")
            cos, sin = rope(q, seq_len=max(T_max, H_max, W_max))
            print(f"  RoPE cos形状: {cos.shape}, sin形状: {sin.shape}")
            
            # create a large cube for each dimension
            print(f"  创建3D索引立方体: ({T_max}, {H_max}, {W_max}, 3)")
            idx_cube = create_idx_cube(T_max, H_max, W_max).to(q.device) #  (T_max, H_max, W_max, 3) cube
            print(f"  索引立方体形状: {idx_cube.shape}")
            
            # select the index for slow (the query)
            print(f"  为查询(慢速特征)选择索引:")
            # determine the temporal idx of the slow frames
            temporal_idx = np.linspace(0, T_max-1, T_q, dtype=int).tolist()
            print(f"    时间索引: {temporal_idx} (从{T_max-1}中选择{T_q}个)")
            # directly select from the cube along the temporal axis
            slow_frames_idx = idx_cube[temporal_idx]  # T_q, h_q, w_q, 3
            print(f"    慢速帧索引形状: {slow_frames_idx.shape}")
            
            if S_q < S_k:
                print(f"    查询空间维度({S_q}) < 键值空间维度({S_k})，需要下采样索引")
                H_q = int(math.sqrt(S_q))
                target_split = 2*H_q+1
                H_fast_index = np.linspace(0, H_max, target_split, dtype=int).tolist()
                H_fast_index = H_fast_index[1:-1]
                H_fast_index = H_fast_index[::2]     
                W_fast_index = H_fast_index   # TODO: we are assume the input shape of is squre
                print(f"    下采样后的H/W索引: {H_fast_index}")
                slow_frames_idx = slow_frames_idx[:, H_fast_index][:,:, W_fast_index] # T_k, h_k, w_k, 3 
                print(f"    下采样后慢速帧索引形状: {slow_frames_idx.shape}")
            
            # determine the spatial idx of the fast frames (the key)
            print(f"  为键值(快速特征)选择索引:")
            H_k = int(math.sqrt(S_k))
            target_split = 2*H_k+1
            H_fast_index = np.linspace(0, H_max, target_split, dtype=int).tolist()
            H_fast_index = H_fast_index[1:-1]
            H_fast_index = H_fast_index[::2]     
            W_fast_index = H_fast_index   # TODO: we are assume the input shape of is squre
            print(f"    快速帧H/W索引: {H_fast_index}")
            # directly select from the cube along the temporal axis
            fast_frames_idx = idx_cube[:, H_fast_index][:,:, W_fast_index] # T_k, h_k, w_k, 3
            print(f"    快速帧索引形状: {fast_frames_idx.shape}")
            
            # flatten the dimension to 1d (3, B, len)
            print(f"  展平索引维度:")
            print(f"    快速帧索引展平前: {fast_frames_idx.shape}")
            fast_frames_idx = fast_frames_idx.view(-1, 3).permute([1, 0]).unsqueeze(dim=1) # which is the key
            print(f"    快速帧索引展平后: {fast_frames_idx.shape}")
            
            print(f"    慢速帧索引展平前: {slow_frames_idx.shape}")
            slow_frames_idx = slow_frames_idx.view(-1, 3).permute([1, 0]).unsqueeze(dim=1) # which is the query
            print(f"    慢速帧索引展平后: {slow_frames_idx.shape}")
            
            # apply it to q
            print(f"  对查询q应用RoPE:")
            print(f"    应用前q形状: {q.shape}")
            print(f"    mrope_section: {self.rope_scaling['mrope_section']}")
            q = apply_multimodal_rotary_pos_emb(
                q,  # torch.Size([1, 4, 6272, 128]) (B, Head_num, sequence length, head_dim)
                cos, sin, 
                slow_frames_idx,  # torch.Size([3, 1, 6272]) #
                self.rope_scaling["mrope_section"] # [16, 24, 24]
            )
            print(f"    应用后q形状: {q.shape}")
            
            # apply it to k
            print(f"  对键k应用RoPE:")
            print(f"    应用前k形状: {k.shape}")
            k = apply_multimodal_rotary_pos_emb(
                k,  # torch.Size([1, 28, 32144, 128]) (B, Head_num, sequence length, head_dim)
                cos, sin, 
                fast_frames_idx,  # torch.Size([3, 1, 32144]) #
                self.rope_scaling["mrope_section"] # [16, 24, 24]
            )
            print(f"    应用后k形状: {k.shape}")
        else:
            print(f"  无RoPE，跳过旋转位置编码")
        
        print(f"\n第三步：维度重排和填充处理")
        # if do not need to do the padding
        if T_k % T_q == 0:
            print(f"  时间维度对齐 (T_k={T_k} % T_q={T_q} == 0)，无需填充")
            # reshape 
            print(f"  重排查询q维度:")
            print(f"    重排前: {q.shape}")
            q = rearrange(q, 'B H (T S) D -> (B T) H S D', T=T_q)
            print(f"    重排后: {q.shape}")
            
            print(f"  重排键k维度:")
            print(f"    第一次重排前: {k.shape}")
            k = rearrange(k, 'B H (T S) D -> B H T S D', T=T_k)
            print(f"    第一次重排后: {k.shape}")
            k = rearrange(k, 'B H (n T) S D -> (B n) H (T S) D', n=T_q)
            print(f"    第二次重排后: {k.shape}")
            
            print(f"  重排值v维度:")
            print(f"    第一次重排前: {v.shape}")
            v = rearrange(v, 'B H (T S) D -> B H T S D', T=T_k)
            print(f"    第一次重排后: {v.shape}")
            v = rearrange(v, 'B H (n T) S D -> (B n) H (T S) D', n=T_q)            
            print(f"    第二次重排后: {v.shape}")
            
            # additional param for mask
            attn_mask = None
            print(f"  设置attention_mask为None（无需掩码）")
            
        else: # do the padding for the key if need, and convert it back to batch
            print(f"  时间维度不对齐 (T_k={T_k} % T_q={T_q} != 0)，需要填充处理")
            
            print(f"  重排查询q维度:")
            print(f"    重排前: {q.shape}")
            q = rearrange(q, 'B H (T S) D -> (B T) H S D', T=T_q)
            print(f"    重排后: {q.shape}")
            
            print(f"  重排键值k,v维度:")
            print(f"    k重排前: {k.shape}")
            k = rearrange(k, 'B H (T S) D -> B H T S D', T=T_k)  
            print(f"    k重排后: {k.shape}")
            print(f"    v重排前: {v.shape}")
            v = rearrange(v, 'B H (T S) D -> B H T S D', T=T_k)  
            print(f"    v重排后: {v.shape}")
            
            # find the cutting point 
            print(f"  计算视频特征分割点:")
            video_feat_split_sizes = split_list_lengths(T_k, T_q)
            print(f"    分割大小: {video_feat_split_sizes} (总长度{T_k}分成{T_q}段)")
            
            # split the video into multiple chunks
            print(f"  分割键值张量:")
            splited_k = torch.split(k, video_feat_split_sizes, dim=2) # (B, Head_num, T, S, head_dim)
            splited_v = torch.split(v, video_feat_split_sizes, dim=2) # (B, Head_num, T, S, head_dim)
            print(f"    分割后k段数: {len(splited_k)}, 各段形状: {[chunk.shape for chunk in splited_k]}")
            print(f"    分割后v段数: {len(splited_v)}, 各段形状: {[chunk.shape for chunk in splited_v]}")
            
            # 计算填充参数
            _, H_num, _, S, C = splited_k[0].shape
            num_of_chunks = len(video_feat_split_sizes)
            max_len_of_all_chunks = max(video_feat_split_sizes)
            print(f"  填充参数:")
            print(f"    H_num: {H_num}, S: {S}, C: {C}")
            print(f"    块数: {num_of_chunks}, 最大块长度: {max_len_of_all_chunks}")
            
            print(f"  创建填充张量:")
            k_padded_chunks = torch.zeros(num_of_chunks, H_num, max_len_of_all_chunks, S, C).to(q.device, dtype=q.dtype) # B, H, Max_len, S, C
            v_padded_chunks = torch.zeros(num_of_chunks, H_num, max_len_of_all_chunks, S, C).to(q.device, dtype=q.dtype) # B, H, Max_len, S, C
            print(f"    k填充张量形状: {k_padded_chunks.shape}")
            print(f"    v填充张量形状: {v_padded_chunks.shape}")
            
            print(f"  填充各个块:")
            for i, (len_of_chunk, curr_k, curr_v) in enumerate(zip(video_feat_split_sizes, splited_k, splited_v)):
                print(f"    块{i}: 长度{len_of_chunk}, 填充到位置[{i}, :, :{len_of_chunk}]")
                k_padded_chunks[i, :, :len_of_chunk] = curr_k 
                v_padded_chunks[i, :, :len_of_chunk] = curr_v
            
            print(f"  重排填充后的张量:")
            print(f"    k重排前: {k_padded_chunks.shape}")
            k_padded_chunks = rearrange(k_padded_chunks, "B H L S C -> B H (L S) C")  # flatten the spatial with the chunk len
            print(f"    k重排后: {k_padded_chunks.shape}")
            print(f"    v重排前: {v_padded_chunks.shape}")
            v_padded_chunks = rearrange(v_padded_chunks, "B H L S C -> B H (L S) C")  # flatten the spatial with the chunk len
            print(f"    v重排后: {v_padded_chunks.shape}")
            
            k = k_padded_chunks
            v = v_padded_chunks
            
            # additional param for mask
            attn_mask = torch.tensor(video_feat_split_sizes).to(k_padded_chunks.device)
            T = max_len_of_all_chunks
            print(f"  设置attention_mask: {attn_mask}")
            print(f"  设置T = {T}")

        print(f"\n第四步：注意力计算")
        print(f"注意力计算输入维度:")
        print(f"  - q: {q.shape}")
        print(f"  - k: {k.shape}")
        print(f"  - v: {v.shape}")
        print(f"  - attn_mask: {attn_mask}")
        print(f"  - training模式: {self.training}")
        print(f"  - flash_attn可用: {flash_attn_varlen_func is not None}")

        if not self.training or flash_attn_varlen_func is None: # if no flash attention
            print(f"  使用PyTorch原生注意力计算")
            if attn_mask is not None:
                print(f"  处理注意力掩码:")
                assert len(attn_mask.shape) == 1, "输入应为长度列表" # the input is the len
                assert T == max(attn_mask), f"最大时间长度应与T一致: {T} vs {max(attn_mask)}"
                print(f"    原始mask: {attn_mask}")
                
                # update the mask with the spatial (deep copy)
                spatial_attn_mask = attn_mask.clone().to(q.device)
                spatial_attn_mask *= S # multiply with the spatial dimension
                print(f"    乘以空间维度后: {spatial_attn_mask}")
                
                max_len = max(spatial_attn_mask)
                print(f"    最大长度: {max_len}")
                
                spatial_attn_mask = torch.arange(max_len).expand(len(spatial_attn_mask), max_len).to(q.device) < spatial_attn_mask.unsqueeze(1) # (B, k_l)
                print(f"    生成二进制mask形状: {spatial_attn_mask.shape}")
                
                spatial_attn_mask = spatial_attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
                spatial_attn_mask = spatial_attn_mask.repeat(1, 1, S_q, 1) # (B, 1, q_l, k_l)
                print(f"    扩展后mask形状: {spatial_attn_mask.shape}")
            else:
                spatial_attn_mask = None
                print(f"    无注意力掩码")
            
            print(f"  执行scaled_dot_product_attention")
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=spatial_attn_mask, dropout_p=self.attn_drop)
            print(f"  注意力计算结果形状: {x.shape}")
            
            print(f"  重排输出维度:")
            x = rearrange(x, 'b h s d -> b s (h d)')
            print(f"  最终输出形状: {x.shape}")
            
        else: # use flash attention           
            print(f"  使用Flash Attention计算")
            if attn_mask is not None: 
                print(f"  处理Flash Attention的掩码:")
                assert len(attn_mask.shape) == 1, "输入应为长度列表" # the input is the len
                assert T == max(attn_mask), f"最大时间长度应与T一致: {T} vs {max(attn_mask)}"                
                print(f"    原始mask: {attn_mask}")
                
                # update the mask with the spatial (deep copy)
                spatial_attn_mask = attn_mask.clone()
                spatial_attn_mask *= S # multiply with the spatial dimension
                print(f"    乘以空间维度后: {spatial_attn_mask}")

                # caclulate len of the mask
                k_max_len = max(spatial_attn_mask)
                print(f"    键的最大长度: {k_max_len}")
                
                # create the mask
                kv_padding_mask = torch.tensor([[True]*curr_seq_len + [False]*(k_max_len-curr_seq_len) for curr_seq_len in spatial_attn_mask]).to(k.device)
                print(f"    生成填充掩码形状: {kv_padding_mask.shape}")
                
                # handle the k and v
                print(f"  处理键值张量:")
                print(f"    k转置前: {k.shape}")
                k_transposed = k.transpose(1, 2)
                print(f"    k转置后: {k_transposed.shape}")
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k_transposed, kv_padding_mask) # k: (batch_size, seqlen_k, nheads, d)
                print(f"    k unpad后: {k_unpad.shape}")
                
                print(f"    v转置前: {v.shape}")
                v_transposed = v.transpose(1, 2)
                print(f"    v转置后: {v_transposed.shape}")
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v_transposed, kv_padding_mask) # v: (batch_size, seqlen_v, nheads, d)
                print(f"    v unpad后: {v_unpad.shape}")
                
                # handle the q (B, H_num, q_len, H_dim)
                print(f"  处理查询张量:")
                cu_seqlens_q = torch.tensor([S_q * i for i in range(k.shape[0]+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = S_q
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                print(f"    q unpad后: {q_unpad.shape}")
                print(f"    cu_seqlens_q: {cu_seqlens_q}")
                print(f"    max_seqlen_q: {max_seqlen_q}")
            else: # if attn mask is None, then do full cross-attn
                print(f"  无掩码的Flash Attention:")
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                print(f"    q_unpad: {q_unpad.shape}")
                print(f"    k_unpad: {k_unpad.shape}")
                print(f"    v_unpad: {v_unpad.shape}")
                
                cu_seqlens_q = torch.tensor([q.shape[2] * i for i in range(k.shape[0]+1)]).to(k.device, dtype=torch.int32)
                cu_seqlens_k = torch.tensor([k.shape[2] * i for i in range(k.shape[0]+1)]).to(k.device, dtype=torch.int32) # handles for the sptial dimenstion
                max_seqlen_q = q.shape[2]
                max_seqlen_k = k.shape[2] # handles for the sptial dimenstion
                print(f"    cu_seqlens_q: {cu_seqlens_q}")
                print(f"    cu_seqlens_k: {cu_seqlens_k}")
                print(f"    max_seqlen_q: {max_seqlen_q}")
                print(f"    max_seqlen_k: {max_seqlen_k}")
            
            print(f"  执行flash_attn_varlen_func")
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            print(f"  Flash Attention结果形状: {x.shape}")
            
            print(f"  重排Flash Attention输出:")
            x = rearrange(x, '(t s) h d -> t s (h d)', t=T_q).unsqueeze(dim=0) # (B, T_q, S_q, C) should be the query tokens
            print(f"  最终输出形状: {x.shape}")
        
        print(f"\n=== VideoCrossAttentionWith3DRope Forward 完成 ===")
        print(f"最终返回形状: {x.shape}")
        return x    


class DecoderVideoCrossAttention(nn.Module):
    '''
        该版本旨在处理查询(query)与视频特征之间的交叉注意力(cross-attention)
    '''
    
    def __init__(self, d_query, d_input, d_model, num_heads, attn_drop=0.0):
        super().__init__()
        # 确保模型维度能被注意力头数整除（保证每个头的维度一致）
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        assert d_query == d_model

        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_model // num_heads  # 每个注意力头的维度

        # 线性变换层：将输入映射到模型维度空间
        self.q_linear = nn.Linear(d_query, d_model)  # 查询向量线性变换
        self.k_linear = nn.Linear(d_input, d_model)  # 键向量线性变换
        self.v_linear = nn.Linear(d_input, d_model)  # 值向量线性变换
        self.attn_drop = attn_drop  # 注意力 dropout 率

    def forward(self, q, k, v, attn_mask=None, rope=None, rope_axis='time'):
        # q:    (B, N_query, C) 查询令牌 这里是
        # k:    (B, T, S, C) 视频令牌（T为时间维度，S为空间维度）
        # v:    (B, T, S, C) 视频令牌（与k形状相同）
        # attn_mask: 指示有效长度的掩码列表
        # rope: 旋转位置编码（rotary positional embedding）
        
        # 解析输入形状
        B, q_len, C = q.shape  # 查询张量形状
        Bk, T, S, C_cond = v.shape  # 值张量形状
        Bv, T, S, C_cond = k.shape  # 键张量形状
        assert B == Bk == Bv  # 确保批次大小一致

        # 对值张量进行线性变换并调整维度
        v = self.k_linear(v).view(B, T*S, self.num_heads, self.head_dim).transpose(1, 2)   # (B, 头数, 时空维度, 头维度)
        q = self.q_linear(q)  # 对查询张量进行线性变换：(B, 查询长度, 总维度)
        k = self.k_linear(k)  # 对键张量进行线性变换：(B, 时间, 空间, 总维度)
        
        # 应用旋转位置编码
        if rope is not None:  # 期望格式：(批次, 头数, 序列长度, 头维度)
            # 调整查询张量维度以适配位置编码
            q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)  # (B, 查询长度, 总维度) -> (B, 头数, 查询长度, 头维度)
            q = rope.rotate_queries_or_keys(q)  # 对查询应用旋转位置编码
            
            # 根据不同轴（时间/空间）处理键张量的位置编码
            if rope_axis == 'time':
                # 按时间轴重组键张量维度
                k = rearrange(k, "b t s (h d) -> (b s) h t d", h=self.num_heads, d=self.head_dim)  # (批次*空间, 头数, 时间, 头维度)
                k = rope.rotate_queries_or_keys(k)  # 对键应用旋转位置编码
                # 恢复键张量维度
                k = rearrange(k, "(b s) h t d -> b h (t s) d", s=S)  # (B, 头数, 时空维度, 头维度)
            elif rope_axis == 'spatial':
                # 按空间轴重组键张量维度
                k = rearrange(k, "b t s (h d) -> (b t) h s d", h=self.num_heads, d=self.head_dim)  # (批次*时间, 头数, 空间, 头维度)
                k = rope.rotate_queries_or_keys(k)  # 对键应用旋转位置编码
                # 恢复键张量维度
                k = rearrange(k, "(b t) h s d -> b h (t s) d", t=T)  # (B, 头数, 时空维度, 头维度)
            else:
                raise NotImplementedError  # 未实现的轴类型
        else:
            # 无位置编码时调整维度
            q = q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 头数, 查询长度, 头维度)
            k = k.view(B, T*S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 头数, 时空维度, 头维度)
        
        # 根据是否使用flash attention选择计算路径
        if not self.training or flash_attn_varlen_func is None:  # 不使用flash attention
            if attn_mask is not None:
                # 验证注意力掩码格式
                assert len(attn_mask.shape) == 1  # 输入应为长度列表
                assert T == max(attn_mask)  # 最大时间长度应与T一致
                
                # 生成空间维度相关的注意力掩码（深拷贝）
                spatial_attn_mask = attn_mask.clone().to(q.device)
                spatial_attn_mask *= S  # 乘以空间维度以扩展掩码
                max_len = max(spatial_attn_mask)
                
                # 生成二进制掩码矩阵（批次, 键长度）
                spatial_attn_mask = torch.arange(max_len).expand(len(spatial_attn_mask), max_len).to(q.device) < spatial_attn_mask.unsqueeze(1)
                # 调整掩码维度以适配注意力计算
                spatial_attn_mask = spatial_attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
                spatial_attn_mask = spatial_attn_mask.repeat(1, 1, q_len, 1)  # (B, 1, 查询长度, 键长度)
            
            # 计算缩放点积注意力（PyTorch原生实现）
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=spatial_attn_mask, dropout_p=self.attn_drop)
            # 调整输出维度以匹配输入格式
            x = rearrange(x, 'b h s d -> b s (h d)')
        else:  # 使用flash attention（高效实现）
            if attn_mask is not None: 
                # 验证注意力掩码格式
                assert len(attn_mask.shape) == 1  # 输入应为长度列表
                assert T == max(attn_mask)  # 最大时间长度应与T一致
                
                # 生成空间维度相关的注意力掩码
                spatial_attn_mask = attn_mask.clone()
                spatial_attn_mask *= S  # 乘以空间维度以扩展掩码
                
                # 计算掩码长度并生成填充掩码
                k_max_len = max(spatial_attn_mask)
                kv_padding_mask = torch.tensor([[True]*curr_seq_len + [False]*(k_max_len-curr_seq_len) for curr_seq_len in spatial_attn_mask]).to(k.device)
                
                # 对键和值进行去填充处理（移除无效位置）
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k.transpose(1, 2), kv_padding_mask) # k: (batch_size, seqlen_k, nheads, d)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v.transpose(1, 2), kv_padding_mask) # v: (batch_size, seqlen_v, nheads, d)
                
                # 处理查询张量维度
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = q_len
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
            else:  # 无注意力掩码时进行全交叉注意力计算
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)])
                cu_seqlens_k = torch.tensor([t_len * i for i in range(B+1)])
                max_seqlen_q = q_len
                max_seqlen_k = t_len
            
            # 使用flash attention计算注意力
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=B)
        # 返回注意力计算结果
        return x


class DecoderTextCrossAttention(nn.Module):
    '''
        This version aims to handle the cross-attn between the query and the text feature
    '''
    
    def __init__(self, d_model, num_heads, attn_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.attn_drop = attn_drop

    def forward(self, q, k, v, attn_mask=None, rope=None):
        # q:    (B, N_query, C) should be the query tokens
        # k:    (B, N_text, C) should be the text tokens
        # v:    (B, N_text, C) should be the text tokens
        # attn_mask: is list of the number which indicate the length of the key
        # rope: rotary positional embedding
        
        #ipdb.set_trace()
        B, q_len, C = q.shape
        Bk, t_len, C_cond = v.shape
        Bv, t_len, C_cond = k.shape
        assert B == Bk == Bv
        
        v = self.k_linear(v).view(B, t_len, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H_num, t_len, H_dim)
        q = self.q_linear(q) # (B, q_len, Total_H_dim)
        k = self.k_linear(k) # (B, t_len, Total_H_dim)
        
        # add the rotary pos embedding
        if rope is not None: # expect in the format of # (batch, heads, seq len, dimension of head)
            # ipdb.set_trace()
            q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim) # (B, q_len, Total_H_dim) -> (B, H_num, q_len, H_dim)
            q = rope.rotate_queries_or_keys(q)
            # ipdb.set_trace()
            k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim) # (B, t_len, Total_H_dim) -> (B, H_num, t_len, H_dim)
            k = rope.rotate_queries_or_keys(k)
        else:
            q = q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, q_len, H_dim)
            k = k.view(B, t_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, t_len, H_dim)
        
        # ipdb.set_trace() # test the text cross-attn
        if not self.training or flash_attn_varlen_func is None: # if no flash attention
            if attn_mask is not None:
                assert len(attn_mask.shape) == 1 # the input is the len
                assert t_len == max(attn_mask)
                # update the mask with the spatial (deep copy)
                attn_mask = attn_mask.clone()
                max_len = max(attn_mask)
                # attn_mask = torch.tensor(attn_mask)
                #ipdb.set_trace()
                attn_mask = torch.arange(max_len).expand(len(attn_mask), max_len).to(q.device) < attn_mask.unsqueeze(1) # (B, k_l)
                attn_mask = attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
                attn_mask = attn_mask.repeat(1, 1, q_len, 1) # (B, 1, q_l, k_l)
                #ipdb.set_trace()
            
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop)
            # x = x.transpose(1, 2)
            # x = x.reshape(B, -1, C)
            x = rearrange(x, 'b h s d -> b s (h d)')
        
        else: # use flash attention
            if attn_mask is not None: 
                assert len(attn_mask.shape) == 1 # the input is the len
                assert t_len == max(attn_mask)                
                # deep copy
                text_attn_mask = attn_mask.clone()
                # caclulate len of the mask
                k_max_len = max(text_attn_mask)
                # create the mask
                kv_padding_mask = torch.tensor([[True]*curr_seq_len + [False]*(k_max_len-curr_seq_len) for curr_seq_len in text_attn_mask]).to(k.device)
                # handle the k and v
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k.transpose(1, 2), kv_padding_mask) # k: (batch_size, seqlen_k, nheads, d)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v.transpose(1, 2), kv_padding_mask) # v: (batch_size, seqlen_v, nheads, d)
                # handle the q (B, H_num, q_len, H_dim)
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = q_len
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
            else: # if attn mask is None, then do full cross-attn
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)])
                cu_seqlens_k = torch.tensor([t_len * i for i in range(B+1)])
                max_seqlen_q = q_len
                max_seqlen_k = t_len
            
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=B)
        # ipdb.set_trace() # save the result
        return x    


class DecoderLayer(nn.Module):
    '''
        This is a version of transformer decoder which combine the DETR and Perceiver-IO
    '''
    
    def __init__(self, 
                 d_query,
                 d_input,
                 d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.0,
                 decoder_func=DecoderVideoCrossAttention):
        super().__init__()
        assert d_query == d_model
        # Init a cross-attn layer
        self.attn = decoder_func(d_query, d_input, d_model, nhead, attn_drop=dropout)
        # Implementation of Feedforward model
        self.ffn = FeedForward(d_model, dim_feedforward=dim_feedforward)
    
    def forward(self, 
                query, 
                memory,
                memory_mask: Optional[Tensor] = None,
                key_temporal_pos: Optional[Tensor] = None,
                rope_axis='time'):
        '''
            query: The query tokens (B, num_q, dim_q)
            memory: The video tokens (B, T, S, dim_video) 
            memory_mask: shoule be the mask of the video tokens
            query_pos: positional embedding for the query
        '''
        
        #ipdb.set_trace()
        query = self.attn(q=query,
                            k=memory,
                            v=memory, 
                            attn_mask=memory_mask,
                            rope=key_temporal_pos,
                            rope_axis=rope_axis) + query
        #ipdb.set_trace()
        query = self.ffn(query) + query
        return query


#### The following is from the slow-fast module
# Referemce: https://github.com/facebookresearch/SlowFast/tree/main
class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        with_pooling=True,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.with_pooling = with_pooling
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        if norm_module == nn.BatchNorm3d:
            self.norm = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        else:
            self.norm = norm_module(dim_out, eps=self.eps)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.with_pooling:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )
        else:
            self.pool_layer = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.with_pooling:
            x = self.pool_layer(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        if norm_module == nn.BatchNorm3d:
            self.a_norm = norm_module(
                num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
        else:
            self.a_norm = norm_module(dim_inner, eps=self._eps)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        if norm_module == nn.BatchNorm3d:
            self.b_norm = norm_module(
                num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
        else:
            self.b_norm = norm_module(dim_inner, eps=self._eps)
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c.final_conv = True
        if norm_module == nn.BatchNorm3d:
            self.c_norm = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.c_norm.transform_final_bn = True
        else:
            self.c_norm = norm_module(
                dim_out, eps=self._eps
            )

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_norm(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_norm(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_norm(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            if norm_module == nn.BatchNorm3d:
                self.branch1_norm = norm_module(
                    num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
                )
            else:
                self.branch1_norm = norm_module(
                    dim_out, eps=self._eps
                )
                
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = drop_path(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_norm(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        # "basic_transform": BasicTransform,
        # "x3d_transform": X3DTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class ResStage(nn.Module):
    """
    This version is modified base on the official implementation
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_size,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        dilation,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
        drop_connect_rate=0.0,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_size (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_size
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_size to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage, self).__init__()
        assert num_block_temp_kernel <= num_blocks
        self.num_blocks = num_blocks
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_size = ([temp_kernel_size] * num_blocks)[: num_block_temp_kernel] + [1] * (num_blocks - num_block_temp_kernel)
        

        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
    ):
        # ipdb.set_trace() # check init
        for i in range(self.num_blocks):
            # Retrieve the transformation function.
            trans_func = get_trans_func(trans_func_name)
            # Construct the block.
            res_block = ResBlock(
                dim_in if i == 0 else dim_out,
                dim_out,
                self.temp_kernel_size[i],
                stride if i == 0 else 1,
                trans_func,
                dim_inner,
                num_groups,
                stride_1x1=stride_1x1,
                inplace_relu=inplace_relu,
                dilation=dilation,
                norm_module=norm_module,
                block_idx=i,
                drop_connect_rate=self._drop_connect_rate,
            )
            self.add_module("res{}".format(i), res_block)

    def forward(self, inputs):

        x = inputs
        for i in range(self.num_blocks):
            m = getattr(self, "res{}".format(i))
            x = m(x)

        return x

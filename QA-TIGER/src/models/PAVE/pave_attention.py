
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
        # q:    (B, N_query, C) 查询令牌（通常是文本特征）
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
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k.transpose(1, 2), kv_padding_mask)  # 键形状: (批次, 序列长度, 头数, 维度)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v.transpose(1, 2), kv_padding_mask)  # 值形状: (批次, 序列长度, 头数, 维度)
                
                # 处理查询张量维度
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = q_len
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
            else:  # 无注意力掩码时进行全交叉注意力计算
                # 调整张量维度以适配flash attention
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)])
                cu_seqlens_k = torch.tensor([(T*S) * i for i in range(B+1)])  # 处理时空维度
                max_seqlen_q = q_len
                max_seqlen_k = (T*S)  # 处理时空维度
            
            # 使用flash attention计算注意力
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            # 恢复输出维度
            x = rearrange(x, '(b s) h d -> b s (h d)', b=B)
        # 返回注意力计算结果
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

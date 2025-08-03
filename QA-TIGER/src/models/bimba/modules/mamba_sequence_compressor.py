import torch
from torch import nn

# Assuming Mamba is in a file named mamba_simple.py in the same directory
from .mamba_simple import Mamba

# --- Copied from mamba_compressor.py to make this file self-contained ---

class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        expand=2,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand

        dim = d_model * expand
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.in_proj = nn.Linear(d_model, dim, bias=True)

        self.out_proj = nn.Linear(dim, d_model, bias=True)

    def forward(self, x):
        x = self.in_proj(x)

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.out_proj(x)
        return x


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        layer_idx,
        use_norm=True,
        use_res=True,
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
        mixer_type="mamba",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_norm = use_norm
        self.use_res = use_res
        if use_norm:
            self.norm = MambaRMSNorm(d_model)
        if mixer_type == "mamba":
            self.mixer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba=bimamba,
            )
        elif mixer_type == "attention":
            self.mixer = Attention(d_model=d_model, expand=expand)

    def forward(self, hidden_states):
        residual = hidden_states
        if self.use_norm:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        if self.use_res:
            hidden_states = residual + hidden_states
        return hidden_states

# --- New Module Implementation ---

class MambaSequenceCompressor(nn.Module):
    """
    使用 Mamba 块将一个长的上下文序列 (context_sequence) 的信息
    压缩并注入到一个短的查询序列 (query_sequence) 中。

    - Context 输入: (B, T, D) - 长序列，例如文本或时间序列特征
    - Query 输入: (B, N, D) - 短序列，需要被更新的查询令牌
    - 输出: (B, N, D) - 更新后的查询令牌
    """
    def __init__(
        self,
        d_model,
        n_layer,
        use_norm=True,
        use_res=True,
        fp32=True,
        query_pos="inter",
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
        multi_scale=True,
        mixer_type="mamba",
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.fp32 = fp32
        self.query_pos = query_pos
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    idx,
                    use_norm=use_norm,
                    use_res=use_res,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba=bimamba,
                    mixer_type=mixer_type,
                )
                for idx in range(n_layer)
            ]
        )

        if fp32:
            self.layers.to(torch.float32)

    def forward(self, context_sequence: torch.Tensor, query_sequence: torch.Tensor) -> torch.Tensor:
        """
        前向传播.

        Args:
            context_sequence: 上下文信息源，形状为 (B, T, D).
            query_sequence: 需要被更新的查询令牌，形状为 (B, N, D).

        Returns:
            更新后的查询令牌，形状为 (B, N, D).
        """
        b, t, c = context_sequence.shape
        n_query = query_sequence.shape[1]

        # 逐层处理
        for mixer_block in self.layers:
            # 1. 令牌融合 (与原版逻辑相同)
            if self.query_pos == "right":
                combined_tokens = torch.cat((context_sequence, query_sequence), dim=1)
            elif self.query_pos == "inter":
                # 创建交错插入的掩码和容器
                combined_len = context_sequence.shape[1] + query_sequence.shape[1]
                combined_tokens = torch.zeros(b, combined_len, c).to(query_sequence.device, dtype=query_sequence.dtype)
                
                mask = torch.zeros(combined_len, dtype=bool, device=query_sequence.device)
                indices = torch.linspace(0, combined_len - 1, n_query + 1, dtype=int)[1:]
                mask[indices] = True
                
                combined_tokens[:, mask] = query_sequence
                combined_tokens[:, ~mask] = context_sequence
            else:
                raise ValueError(f"Unknown query_pos: {self.query_pos}")

            # 2. 核心计算 (与原版逻辑相同)
            if self.fp32:
                dtype_prev = combined_tokens.dtype
                combined_tokens = combined_tokens.to(torch.float32)
            
            combined_tokens = mixer_block(combined_tokens)
            
            if self.fp32:
                combined_tokens = combined_tokens.to(dtype_prev)

            # 3. 令牌分离 (与原版逻辑相同)
            if self.query_pos == "right":
                query_sequence = combined_tokens[:, -n_query:, :]
                context_sequence = combined_tokens[:, :-n_query, :]
            elif self.query_pos == "inter":
                query_sequence = combined_tokens[:, mask]
                context_sequence = combined_tokens[:, ~mask]

            # 4. 多尺度处理 (修改点)
            # 原版是针对 (B,F,H,W,C) 的 F 维度降采样
            # 这里我们直接对 (B,T,D) 的 T 维度进行降采样
            if self.multi_scale and context_sequence.shape[1] > 1:
                context_sequence = context_sequence[:, ::2, :]
        
        return query_sequence

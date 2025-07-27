import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, n_position=1024):
        super(PositionalEmbedding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model))

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                    torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class TransformerLayer(nn.Module):
    def __init__(self, device, d_model, d_ff, patch_nums, patch_size, dynamic, factorized, layer_number, batch_norm):
        super(TransformerLayer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.dynamic = dynamic
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number
        self.batch_norm = batch_norm


        ##intra_patch_attention
        self.embeddings_generator = nn.ModuleList([nn.Sequential(*[
            nn.Linear(512, self.d_model)]) for _ in range(self.patch_nums)])
        self.intra_d_model = self.d_model
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_d_model, factorized=factorized)
        self.weights_generator_distinct = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=16,
                                                          factorized=factorized, number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(self.intra_d_model, self.intra_d_model, mem_dim=None,
                                                        factorized=False, number_of_weights=2)

        # 添加线性层来调整intra attention输出的维度
        # self.intra_projection = nn.Linear(self.d_model, self.d_model)
        # 添加可学习的序列长度变换层
        # self.sequence_adapter = nn.ModuleDict({
        #     'compress': nn.Linear(self.d_model, self.d_model),  # 用于压缩长序列
        #     'expand': nn.Linear(self.d_model, self.d_model),    # 用于扩展短序列
        # })



        ##inter_patch_attention
        self.stride = patch_size
        # patch_num = int((context_window - cut_size) / self.stride + 1)

        self.inter_d_model = self.d_model * self.patch_size
        ##inter_embedding
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)
        # Positional encoding
        self.W_pos = positional_encoding(pe='zeros', learn_pe=True, q_len=self.patch_nums, d_model=self.inter_d_model)
        n_heads = self.d_model
        d_k = self.inter_d_model // n_heads
        d_v = self.inter_d_model // n_heads
        self.inter_patch_attention = Inter_Patch_Attention(self.inter_d_model, self.inter_d_model, n_heads, d_k, d_v, attn_dropout=0,
                                          proj_dropout=0.1, res_attention=False)


        ##Normalization
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.d_model), Transpose(1,2))
        self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(self.d_model), Transpose(1,2))

        ##FFN
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_ff, bias=True),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.Linear(self.d_ff, self.d_model, bias=True))

    def forward(self, x, query):
        # 去掉num_nodes维度后的处理
        # x: [batch_size, temporal_length, d_model]
        # query: [batch_size, query_length, 1536]

        new_x = x
        # print(f"Input x shape: {x.shape}")
        # print(f"Input query shape: {query.shape}")
        batch_size = x.size(0)
        intra_out_concat = None

        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        ####intra Attention#####
        for i in range(self.patch_nums):
            # 提取patch: [batch_size, patch_size, d_model]
            t = x[:, i * self.patch_size:(i + 1) * self.patch_size, :]
            # print(f"Patch {i} shape: {t.shape}")
            # query embedding: [batch_size, query_length, d_model]
            intra_emb = self.embeddings_generator[i](query)
            intra_emb = torch.cat([intra_emb, t], dim=1)
            # print(f"Intra embedding shape for patch {i}: {intra_emb.shape}")
            # 拼接: [batch_size, query_length + patch_size, d_model]
            # t = torch.cat([intra_emb, t], dim=1)
            # print(f"Concatenated shape for patch {i}: {t.shape}")


            # out, attention = self.intra_patch_attention(intra_emb, t, t, weights_distinct, biases_distinct, weights_shared,biases_shared)
            out, attention = self.intra_patch_attention(t, intra_emb, intra_emb, weights_distinct, biases_distinct, weights_shared,biases_shared)
            # print(f"Output shape for patch {i}: {out.shape}")
            if intra_out_concat is None:
                intra_out_concat = out
            else:
                intra_out_concat = torch.cat([intra_out_concat, out], dim=1)

            # print(f"intra_out_concat shape for patch {i}: {intra_out_concat.shape}")

        # 重新组织intra_out_concat的维度
        # 当前: intra_out_concat [batch_size, patch_nums * query_length, d_model]
        # 需要: [batch_size, patch_nums * patch_size, d_model]
        
        # 将intra_out_concat重新reshape为每个patch的输出
        # print(f"intra_out_concat.size: {intra_out_concat.size(1)}")
        query_length = intra_out_concat.size(1) // self.patch_nums  # 每个patch的query长度
        # print(f"patcnh_nums: {self.patch_nums}, patch_size: {self.patch_size}")
        # print(f"Query length for intra attention: {query_length}")
        intra_out_concat = intra_out_concat.view(batch_size, self.patch_nums, query_length, self.d_model)
        # print(f"Reshaped intra_out_concat: {intra_out_concat.shape}")
        # 处理query_length和patch_size不匹配的问题
        # if query_length != self.patch_size:
        #     if query_length > self.patch_size:
        #         # 对每个patch独立进行平均池化，保持特征维度不变
        #         # 重新组织为 [batch_size * patch_nums, query_length, d_model]
        #         intra_reshaped = intra_out_concat.view(batch_size * self.patch_nums, query_length, self.d_model)
        #         # 转置为 [batch_size * patch_nums, d_model, query_length] 用于1D池化
        #         intra_reshaped = intra_reshaped.transpose(1, 2)
        #         # 应用自适应平均池化
        #         intra_pooled = F.adaptive_avg_pool1d(intra_reshaped, self.patch_size)
        #         # 转置回 [batch_size * patch_nums, patch_size, d_model]
        #         intra_pooled = intra_pooled.transpose(1, 2)
        #         # 重新组织为 [batch_size, patch_nums, patch_size, d_model]
        #         intra_out_concat = intra_pooled.view(batch_size, self.patch_nums, self.patch_size, self.d_model)
        #     elif query_length < self.patch_size:
        #         # 对每个patch独立进行线性插值上采样
        #         # 重新组织为 [batch_size * patch_nums, query_length, d_model]
        #         intra_reshaped = intra_out_concat.view(batch_size * self.patch_nums, query_length, self.d_model)
        #         # 转置为 [batch_size * patch_nums, d_model, query_length] 用于1D插值
        #         intra_reshaped = intra_reshaped.transpose(1, 2)
        #         # 使用线性插值进行上采样，保持特征的连续性
        #         intra_upsampled = F.interpolate(intra_reshaped, size=self.patch_size, mode='linear', align_corners=False)
        #         # 转置回 [batch_size * patch_nums, patch_size, d_model]
        #         intra_upsampled = intra_upsampled.transpose(1, 2)
        #         # 重新组织为 [batch_size, patch_nums, patch_size, d_model]
        #         intra_out_concat = intra_upsampled.view(batch_size, self.patch_nums, self.patch_size, self.d_model)
        #
        
        # 重新组织为 [batch_size, patch_nums * patch_size, d_model]
        intra_out_concat = intra_out_concat.contiguous().view(batch_size, self.patch_nums * self.patch_size, self.d_model)
        
        # 应用投影层来确保特征对齐
        # intra_out_concat = self.intra_projection(intra_out_concat)

        ####inter Attention######
        # 重新使用原始x进行inter attention
        # print(f"New x shape before unfold: {new_x.shape}")
        x_inter = new_x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # [b x patch_num x d_model x patch_len]
        b, patch_num, dim, patch_len = x_inter.shape
        # print(f"x_inter shape after unfold: {x_inter.shape}")
        # 重组为 [batch_size, patch_num, d_model * patch_len]
        x_inter = x_inter.reshape(b, patch_num, dim * patch_len)
        # print(f"x_inter shape after reshape: {x_inter.shape}")
        x_inter = self.emb_linear(x_inter)
        x_inter = self.dropout(x_inter + self.W_pos)
        # print(f"x_inter shape after dropout: {x_inter.shape}")
        inter_out, attention = self.inter_patch_attention(Q=x_inter, K=x_inter, V=x_inter)  # [b, patch_num, inter_d_model]
        
        # 重组回原始形状: [batch_size, patch_nums * patch_size, d_model]
        inter_out = inter_out.reshape(b, patch_num, self.patch_size, self.d_model)
        inter_out = inter_out.reshape(b, patch_num * self.patch_size, self.d_model)
        # print(f"inter_out shape after final reshape: {inter_out.shape}")
        # 现在可以将intra和inter attention的输出都加回到原始输入
        # print(f"Final shapes - new_x: {new_x.shape}, intra_out_concat: {intra_out_concat.shape}, inter_out: {inter_out.shape}")
        out = new_x + intra_out_concat + inter_out

        if self.batch_norm:
            out = self.norm_attn(out)
        
        ##FFN
        out = self.dropout(out)
        out = self.ff(out) + out
        
        if self.batch_norm:
            out = self.norm_ffn(out)
            
        return out, attention



class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super(CustomLinear, self).__init__()
        self.factorized = factorized

    def forward(self, input, weight, bias):
        # 去掉num_nodes维度后的处理
        # input: [batch_size, seq_length, d_model]
        # weight: [d_model, d_model] 权重张量
        # bias: [d_model] 偏置张量
        
        return torch.matmul(input, weight) + bias


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super(Intra_Patch_Attention, self).__init__()
        self.head = 2

        if d_model % self.head != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        # 去掉num_nodes维度后的处理
        # query: [batch_size, query_length, d_model]
        # key/value: [batch_size, key_length, d_model]
        
        batch_size = query.shape[0]

        # 权重变换
        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        
        # 多头分割
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)  # [batch_size * heads, query_length, head_size]
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)      # [batch_size * heads, key_length, head_size]
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)  # [batch_size * heads, key_length, head_size]

        # 调整维度用于注意力计算
        # query: [batch_size * heads, query_length, head_size]
        # key: [batch_size * heads, head_size, key_length] 
        # value: [batch_size * heads, key_length, head_size]
        key = key.transpose(-2, -1)  # [batch_size * heads, head_size, key_length]

        # 注意力计算
        attention = torch.matmul(query, key)  # [batch_size * heads, query_length, key_length]
        attention /= (self.head_size ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        # 加权求和
        x = torch.matmul(attention, value)  # [batch_size * heads, query_length, head_size]
        
        # 合并多头
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)  # [batch_size, query_length, d_model]

        # 后续处理
        # weights_shared和biases_shared是列表，包含多个权重
        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        
        return x, attention


class Inter_Patch_Attention(nn.Module):
    def __init__(self, d_model, out_dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                  res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, out_dim), nn.Dropout(proj_dropout))


    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s    : [bs x n_heads x q_len x d_k]  此处的q_len为patch_num
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3,
                                                                               1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, Q.shape[1],
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, factorized, number_of_weights=4):
        super(WeightGenerator, self).__init__()
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.factorized = factorized
        self.out_dim = out_dim
        
        if self.factorized:
            # 单节点处理，去掉num_nodes维度
            self.memory = nn.Parameter(torch.randn(mem_dim), requires_grad=True)
            self.generator = nn.Sequential(*[
                nn.Linear(mem_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 100)
            ])

            self.mem_dim = 10
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.Q = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim), requires_grad=True) for _ in
                 range(number_of_weights)])
        else:
            self.P = nn.ParameterList(
                [nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True) for _ in range(number_of_weights)])
            self.B = nn.ParameterList(
                [nn.Parameter(torch.Tensor(out_dim), requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            # 单节点处理
            memory = self.generator(self.memory.unsqueeze(0))  # [1, 100]
            bias = [torch.matmul(memory, self.B[i]).squeeze(0) for i in range(self.number_of_weights)]  # [out_dim]
            memory = memory.view(self.mem_dim, self.mem_dim)  # [mem_dim, mem_dim]
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]  # [in_dim, out_dim]
            return weights, bias
        else:
            return self.P, self.B



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

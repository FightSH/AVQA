# 该脚本实现信息聚合器，用于多模态特征融合

import math
import ipdb

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import VideoCrossAttentionWith3DRope, DecoderLayer, ResNetBasicStem, ResStage
from einops import rearrange, repeat


class LayerNorm2d(nn.LayerNorm):
    """
    2D层归一化
    参考ConvNext的层归一化实现: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    输入格式: (B, C, H, W)
    """
    def forward(self, x):
        # 将通道维度移动到最后: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # 应用层归一化
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 恢复原始维度顺序: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm3d(nn.LayerNorm):
    """
    3D层归一化，用于视频特征
    参考ConvNext的层归一化实现
    输入格式: (B, C, T, H, W)，其中T为时间维度
    """
    def forward(self, x):
        # 将通道维度移动到最后: (B, C, T, H, W) -> (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1) 
        # 应用层归一化
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 恢复原始维度顺序: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3) 
        return x


class Qwen2RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）实现
    这部分代码来自Qwen2-VL，用于为注意力机制提供位置信息
    
    Args:
        dim: 嵌入维度
        max_position_embeddings: 最大位置编码长度
        base: 频率基数
        device: 计算设备
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算逆频率，用于生成正弦和余弦位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 预先构建缓存以支持torch.jit.trace
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """设置余弦和正弦缓存"""
        self.max_seq_len_cached = seq_len
        # 生成位置索引
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        # 计算频率矩阵
        freqs = torch.outer(t, self.inv_freq)
        # 拼接频率矩阵以匹配论文中的排列方式
        emb = torch.cat((freqs, freqs), dim=-1)
        # 缓存余弦和正弦值
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):
        """
        前向传播
        Args:
            x: 输入张量 [bs, num_attention_heads, seq_len, head_size]
            seq_len: 序列长度
        Returns:
            余弦和正弦位置编码
        """
        # 如果序列长度超过缓存长度，重新设置缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class PAVEModuleV5(nn.Module):
    """
    PAVE模块第5版 - 多模态信息聚合器
    
    该模块包含：
    1. 多个卷积块用于特征映射
    2. 交叉注意力层用于空间信息聚合
    3. 可选的MLP输出层
    
    主要功能是将快速特征（如VideoVAE特征）和慢速特征（如LanguageBind特征）
    通过交叉注意力机制进行融合，生成统一的多模态表示。
    
    Args:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        embed_dim: 嵌入维度（可选）
        fast_input_mapping_type: 快速输入映射类型（'conv'或'linear'）
        number_of_input_mapping_layer: 输入映射层数量
        number_of_block_each_layer: 每层的块数量
        sptial_stride: 空间步长
        dim_scaling: 维度缩放因子
        query_type: 查询类型（'slow_feat'或'learnable'）
        chunks_number: 块数量
        number_of_query: 查询数量
        query_input_dim: 查询输入维度
        cross_attn_hidden_dim: 交叉注意力隐藏维度
        num_cross_attn_head: 交叉注意力头数
        num_cross_attn_layer: 交叉注意力层数
        use_3d_rope: 是否使用3D旋转位置编码
        use_output_mlp: 是否使用输出MLP
        use_dropout: 是否使用dropout
        dropout_rate: dropout率
        use_output_norm: 是否使用输出归一化
        train_addition_start_end_tokens: 是否训练额外的开始结束标记
        mlp_depth: MLP深度
        use_slow_feat_before_mlp: 是否在MLP前使用慢速特征
    """
    def __init__(
        self,
        input_dim,                       # 输入特征维度
        output_dim,                      # 输出维度
        embed_dim=None,                  # 嵌入维度
   
        fast_input_mapping_type='conv',  # 快速输入映射类型：基于卷积的VideoVAE特征，基于线性的LanguageBind特征
        number_of_input_mapping_layer=5, # 卷积层数量
        number_of_block_each_layer=[1, 2, 2, 2, 2], # 每层的块数量
        sptial_stride=[1, 1, 1, 1, 1],  # 空间步长
        dim_scaling=[1, 1, 1, 1, 1],    # 维度缩放
   
        query_type='slow_feat',          # 查询类型
        chunks_number=32,                # 使用可学习标记的块数量
        number_of_query=196,             # 查询数量
        query_input_dim=896,             # 使用慢速特征作为查询时的维度
        cross_attn_hidden_dim=512,       # 交叉注意力隐藏维度
        num_cross_attn_head=4,           # 注意力头数
        num_cross_attn_layer=1,          # 交叉注意力层数
        use_3d_rope=True,                # 是否使用3D旋转位置编码
        
        use_output_mlp=True,             # 是否使用输出MLP
        use_dropout=True,                # 其他配置
        dropout_rate=0.1,                # dropout率
        use_output_norm=False,           # 是否使用输出归一化
        train_addition_start_end_tokens=False, # 是否训练额外的开始结束标记
        mlp_depth=2,                     # MLP深度
        use_slow_feat_before_mlp=False   # 是否在MLP前使用慢速特征
        
    ):
        super().__init__()
        # 超参数设置
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp_depth = mlp_depth
        self.act = nn.SiLU(inplace=True)  # Swish激活函数
        self.use_slow_feat_before_mlp = use_slow_feat_before_mlp
        
        # 快速输入映射配置
        self.fast_input_mapping_type = fast_input_mapping_type
        if self.fast_input_mapping_type == 'conv':
            # 初始化卷积层（将空间维度下采样2x2，将维度从4增加到16）
            self.input_mapping = nn.ModuleList()
            self.embed_dim = embed_dim if embed_dim is not None else input_dim * 4
            curr_dim = input_dim
            
            # 构建多层输入映射网络
            for i in range(number_of_input_mapping_layer):
                curr_spatial_stride = sptial_stride[i]
                curr_dim_scaling = dim_scaling[i]
                curr_stage_block_num = number_of_block_each_layer[i]
                
                if i == 0:
                    # 第一层：基础stem层
                    self.input_mapping.append(
                        ResNetBasicStem(curr_dim,
                                        self.embed_dim,
                                        kernel = [1, 1, 1],     # 卷积核大小
                                        stride = [1, 1, 1],     # 步长
                                        padding = [0, 0, 0],    # 填充
                                        with_pooling=False,     # 不使用池化
                                        norm_module=LayerNorm3d, # 使用3D层归一化
                                        )
                    )
                    curr_dim = self.embed_dim
                else:
                    # 后续层：ResNet阶段
                    self.input_mapping.append(
                        ResStage(dim_in=curr_dim,
                                    dim_out=curr_dim * curr_dim_scaling,
                                    stride=curr_spatial_stride,
                                    temp_kernel_size=3,      # 时间卷积核大小
                                    num_blocks=curr_stage_block_num,
                                    dim_inner=curr_dim,
                                    num_groups=1,
                                    num_block_temp_kernel=curr_stage_block_num,
                                    dilation=1 if i < number_of_input_mapping_layer-1 else 2, # 最后一层使用膨胀卷积
                                    trans_func_name="bottleneck_transform",
                                    norm_module=LayerNorm3d,
                                    )
                    )
                    curr_dim *= curr_dim_scaling    
            input_mapping_out_channels = curr_dim
            
        elif self.fast_input_mapping_type == 'linear':
            # 线性映射（用于LanguageBind等特征）
            assert embed_dim is not None
            self.embed_dim = embed_dim
            self.input_mapping = nn.Linear(self.input_dim, self.embed_dim)
            input_mapping_out_channels = self.embed_dim
        else:
            raise NotImplementedError
        
        # 查询配置
        self.use_slow_as_query = True
        self.query_input_dim = query_input_dim
        self.cross_attn_hidden_dim = cross_attn_hidden_dim
        
        # 定义查询的输入映射
        self.query_type = query_type
        if self.query_type == 'slow_feat':
            # 使用慢速特征作为查询
            self.query_input_mapping = nn.Linear(self.query_input_dim, self.cross_attn_hidden_dim)
        elif self.query_type == 'learnable':
            # 使用可学习的查询标记
            # 由于添加了3D位置编码，因此不应在此处重复查询标记
            self.learnable_query = nn.Parameter(torch.randn(chunks_number, number_of_query, cross_attn_hidden_dim))
        else:
            raise NotImplementedError
        
        # 时间嵌入配置
        self.use_3d_rope = use_3d_rope
        if self.use_3d_rope:
            # 初始化3D旋转位置编码
            self.temporal_embedding = Qwen2RotaryEmbedding(self.cross_attn_hidden_dim // num_cross_attn_head)
        else:
            self.temporal_embedding = None
            
        # 交叉注意力层
        self.cross_attn_layers = nn.ModuleList()
        for i in range(num_cross_attn_layer):
            self.cross_attn_layers.append(
                DecoderLayer(self.cross_attn_hidden_dim,    # 查询维度
                            input_mapping_out_channels,      # 键值维度
                            self.cross_attn_hidden_dim,      # 输出维度
                            num_cross_attn_head,             # 注意力头数
                            dim_feedforward=2*self.cross_attn_hidden_dim, # 前馈网络维度
                            dropout=0.0,                     # dropout率
                            decoder_func=VideoCrossAttentionWith3DRope) # 使用带3D RoPE的视频交叉注意力
            )        

        # 输出MLP配置
        self.use_output_mlp = use_output_mlp
        if self.use_output_mlp:
            mlp_input_dim = self.cross_attn_hidden_dim
            mlp_hidden_dim = mlp_input_dim * 2
            # 构建MLP层
            modules = [nn.Linear(mlp_input_dim, mlp_hidden_dim)]
            for _ in range(1, self.mlp_depth):
                modules.append(nn.GELU())  # GELU激活函数
                modules.append(nn.Linear(mlp_hidden_dim, self.output_dim))
            self.output_mapping = nn.Sequential(*modules)
        else:
            self.output_mapping = None
            # 确保维度匹配
            assert self.cross_attn_hidden_dim == output_dim
        
        # Dropout配置
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.output_dropout = nn.Dropout(p=dropout_rate)
        else:
            self.output_dropout = None
            
        # 层归一化配置
        self.use_output_norm = use_output_norm
        if self.use_output_norm:
            self.output_norm = nn.LayerNorm(self.output_dim)
        else:
            self.output_norm = None
        
        self.module_dtype = None
        
        # 额外的开始结束标记配置
        self.train_addition_start_end_tokens = train_addition_start_end_tokens
        if self.train_addition_start_end_tokens:
            # 训练额外的4个标记来表示视频和图像特征的开始和结束
            # self.start_end_tokens[0]: 视频开始
            # self.start_end_tokens[1]: 视频结束
            # self.start_end_tokens[2]: 图像开始
            # self.start_end_tokens[3]: 图像结束
            self.start_end_tokens = nn.Parameter(torch.randn(4, self.output_dim))
        else:
            self.start_end_tokens = None

        # 权重初始化
        self.apply(self.__init_weights__)
        
        # 将输出归一化的gamma初始化为零（残差连接策略）
        if self.output_norm is not None:
            nn.init.zeros_(self.output_norm.weight)
            nn.init.zeros_(self.output_norm.bias)
        

    def __init_weights__(self, module,):
        """
        权重初始化函数
        注意：此初始化不会影响SSM模块的初始化
        """
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)  # 将线性层偏置初始化为零

    @property
    def dtype(self):
        """
        获取模块数据类型的便捷方法
        """
        if self.module_dtype is None:
            self.module_dtype = list(set(p.dtype for p in self.parameters()))[0]
        return self.module_dtype
    
    def forward(self, x, frame_num, 
                q_text_embeds=None, 
                q_text_nums=None, 
                chunk_num=None,
                slow_feats=None):
        """
        前向传播函数
        
        Args:
            x: 快速特征 (B, T, H, W, C) - 主要的视频特征
            frame_num: 每个样本的帧数 (B,) - 用于掩码的长度
            q_text_embeds: 文本嵌入（暂未使用）
            q_text_nums: 文本数量（暂未使用）
            chunk_num: 块数量
            slow_feats: 慢速特征列表 - 用作查询的特征，格式为 [torch.Size([32, 196, 896]), ...]
            
        Returns:
            聚合后的多模态特征
            
        数据流：
        1. 快速特征通过卷积网络映射：(B,T,H,W,4) -> (B,T,H,W,256) -> ... -> (B,T,H/2,W/2,512)
        2. 慢速特征作为查询：(T,196,896) -> (T,196,512)
        3. 通过交叉注意力融合：query与key-value特征交互
        4. 经过MLP和归一化输出最终特征
        """
        # 数据类型转换
        x = x.to(self.dtype)
        if slow_feats is not None:
            if isinstance(slow_feats, list):
                slow_feats = [ele.to(self.dtype) for ele in slow_feats]
            else:
                slow_feats = slow_feats.to(self.dtype)
        
        all_features = []
        
        # 对每个样本分别处理，将每个块转换为固定数量的标记
        for curr_feat, curr_query, curr_len in zip(x, slow_feats, frame_num):
            # 准备查询特征
            if self.query_type == 'slow_feat':
                # 使用慢速特征作为查询，通过线性层映射到交叉注意力维度
                curr_query = self.query_input_mapping(curr_query) # (T, S, C)
            elif self.query_type == 'learnable':
                # 使用可学习的查询标记
                curr_query = self.learnable_query # (T, S, C)
            else:
                raise NotImplementedError
                
            # 根据实际长度裁剪特征
            curr_feat = curr_feat[:curr_len] # (T, H, W, C) 例如：torch.Size([184, 28, 28, 4])
            
            # 快速特征映射
            if self.fast_input_mapping_type == 'conv':
                # 卷积映射：重排维度并通过卷积网络
                curr_feat = rearrange(curr_feat, "t h w c -> c t h w")  # 时间优先格式转为通道优先
                curr_feat = curr_feat.unsqueeze(dim=0)  # 添加批次维度
                
                # 通过多层卷积网络处理
                # [1, 4, 53, 28, 28] -> [1, 256, 53, 28, 28] -> ... -> [1, 512, 53, 14, 14]
                for layer in self.input_mapping:
                    curr_feat = layer(curr_feat)
            else:
                # 线性映射
                curr_feat = self.input_mapping(curr_feat)
                curr_feat = rearrange(curr_feat, "t h w c -> c t h w")
                curr_feat = curr_feat.unsqueeze(dim=0) # 保持 1, c, t, h, w 的维度格式
                
            # 时间维度上采样（如果当前特征长度小于所需块数）
            T = curr_feat.shape[2]
            if T < chunk_num:
                print(f'当前特征时间维度: {T}, 上采样到: {chunk_num}')
                temporal_scale_factor = chunk_num / T
                # 使用三线性插值进行时间维度上采样
                upsampling = nn.Upsample(scale_factor=(temporal_scale_factor, 1, 1), mode='trilinear', align_corners=True)
                curr_feat = upsampling(curr_feat)
            
            # 重新排列维度为交叉注意力所需格式
            curr_feat = rearrange(curr_feat, "b c t h w -> b t (h w) c")  # 空间维度展平

            # 交叉注意力处理
            output = curr_query.unsqueeze(dim=0)  # 添加批次维度：T_q, C, D -> 1, T_q, C, D 
            for layer in self.cross_attn_layers:
                # 执行交叉注意力：查询来自慢速特征，键值来自快速特征
                output = layer(output,       # 查询：1, T_q, C, D 
                    curr_feat,               # 键值：b t (h w) c
                    memory_mask=None,        # 无记忆掩码
                    key_temporal_pos=self.temporal_embedding,  # 时间位置编码
                    rope_axis='spatial')     # RoPE应用于空间维度
            
            # 重塑输出格式
            output = output.view(-1, output.shape[-1])  # 展平为2D
            output = output.unsqueeze(dim=0)  # 添加批次维度
            all_features.append(output)
        
        # 合并所有样本的特征
        output = torch.cat(all_features, dim=0)
        
        # 输出映射层
        if self.output_mapping is not None:
            output = self.output_mapping(output)
            
        # Dropout正则化
        if self.output_dropout is not None:
            output = self.output_dropout(output)
            
        # 输出归一化
        if self.output_norm is not None:
            output = self.output_norm(output)

        return output
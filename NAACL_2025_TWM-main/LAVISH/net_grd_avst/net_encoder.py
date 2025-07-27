import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from visual_net import resnet18

from ipdb import set_trace
import timm
from torch.distributions.normal import Normal
from einops import rearrange, repeat
from audio_layers import Transformer_Layer
from audio_others import SparseDispatcher, FourierLayer, series_decomp_multi, MLP

class VisualAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None,
                 use_bn=True, use_gate=True):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = opt.is_multimodal
        self.opt = opt

        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor

            self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
            self.gate_av = nn.Parameter(torch.zeros(1))

            ### <------

            self.activation = nn.ReLU(inplace=True)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)

            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group,
                                          bias=False)

            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group,
                                        bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt.is_before_layernorm:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt.is_post_layernorm:
                self.ln_post = nn.LayerNorm(output_dim)
        ### <---------

        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)

            if use_bn:
                self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise NotImplementedError

    def forward(self, x, vis_token=None):
        if self.adapter_kind == "bottleneck" and self.is_multimodal:

            ### -------> high dim att
            rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))

            att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

            att_v2tk = F.softmax(att_v2tk, dim=-1)
            rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0, 2, 1))

            rep_token = rep_token + rep_token_res

            att_tk2x = torch.bmm(x.squeeze(-1).permute(0, 2, 1), rep_token.permute(0, 2, 1))

            att_tk2x = F.softmax(att_tk2x, dim=-1)
            x_res = torch.bmm(att_tk2x, rep_token).permute(0, 2, 1).unsqueeze(-1)

            x = x + self.gate_av * x_res.contiguous()

            ### <----------
            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            ## <----

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "bottleneck":

            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)


        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C'))
                output = rearrange(output, 'N L C -> N C L')

        if self.opt.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(-1)

        if self.gate is not None:
            output = self.gate * output

        return output


def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0

    return out_match, batch_labels


# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature


class AVQA_Fusion_Net(nn.Module):

    def __init__(self, opt):
        super(AVQA_Fusion_Net, self).__init__()

        self.opt = opt

        # for features
        self.fc_a1 = nn.Linear(128, 512)
        self.fc_a2 = nn.Linear(512, 512)

        self.fc_a1_pure = nn.Linear(128, 512)
        self.fc_a2_pure = nn.Linear(512, 512)
        # self.visual_net = resnet18(pretrained=True)

        self.fc_v = nn.Linear(2048, 512)
        # self.fc_v = nn.Linear(1536, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.fc = nn.Linear(1024, 512)
        self.fc_aq = nn.Linear(512, 512)
        self.fc_vq = nn.Linear(512, 512)

        self.linear11 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(512, 512)

        self.linear21 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(512)

        self.attn_a = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(512, 4, dropout=0.1)

        # question
        self.question_encoder = QstEncoder(93, 512, 512, 1, 512)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(512, 42)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl = nn.Linear(1024, 512)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()

        self.yb_fc_v = nn.Linear(1536, 512)
        self.yb_fc_a = nn.Linear(1536, 512)

        # self.resnet = timm.create_model('resnet18', pretrained=True)
        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)

        ### ------------> for swin
        hidden_list = []
        down_in_dim = []
        down_out_dim = []
        for idx_layer, my_blk in enumerate(self.swin.layers):
            if not isinstance(my_blk.downsample, nn.Identity):
                down_in_dim.append(my_blk.downsample.reduction.in_features)
                down_out_dim.append(my_blk.downsample.reduction.out_features)

            for blk in my_blk.blocks:
                hidden_d_size = blk.norm1.normalized_shape[0]
                hidden_list.append(hidden_d_size)
        ### <--------------

        self.audio_adapter_blocks_p1 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                          dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                          use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
            for i in range(len(hidden_list))])

        self.vis_adapter_blocks_p1 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                          dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                          use_bn=self.opt.is_bn, use_gate=True)
            for i in range(len(hidden_list))])

        self.audio_adapter_blocks_p2 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                          dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                          use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
            for i in range(len(hidden_list))])

        self.vis_adapter_blocks_p2 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",
                          dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt,
                          use_bn=self.opt.is_bn, use_gate=True)
            for i in range(len(hidden_list))])


    def frozen_model(self):
        for p in self.parameters():
            p.requires_grad = False
        return


    def forward(self, audio, visual_posi):
        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        bs, t, c, h, w = visual_posi.shape
        print(f"AVQA FUSION audio shape: ", {audio.shape})
        print(f"AVQA FUSION visual_posi shape: ", {visual_posi.shape})
        audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)
        audio = rearrange(audio, 'b t c w h -> (b t) c w h')

        ###### ---------------->
        f_a = self.swin.patch_embed(audio)

        visual_posi = rearrange(visual_posi, 'b t c w h -> (b t) c w h')
        f_v = self.swin.patch_embed(visual_posi)
        # f_v_neg = self.swin.patch_embed(visual_nega)
        idx_layer = 0
        multi_scale = []

        idx_block = 0

        for _, my_blk in enumerate(self.swin.layers):

            for blk in my_blk.blocks:
                f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                  f_v.permute(0, 2, 1).unsqueeze(-1))
                f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                f_a.permute(0, 2, 1).unsqueeze(-1))

                f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
                f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                f_a = f_a + blk.drop_path1(blk.norm1(blk._attn(f_a)))
                f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                  f_v.permute(0, 2, 1).unsqueeze(-1))
                f_v_res = self.vis_adapter_blocks_p2[idx_layer](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                f_a.permute(0, 2, 1).unsqueeze(-1))

                f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
                f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                f_a = f_a + blk.drop_path2(blk.norm2(blk.mlp(f_a)))
                f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                idx_layer = idx_layer + 1
            #####
            f_v = my_blk.downsample(f_v)
            f_a = my_blk.downsample(f_a)

        f_v = self.swin.norm(f_v)
        f_a = self.swin.norm(f_a)

        return f_v, f_a

    def qst_feature_forward(self, question):
        ## question features
        qst_feature = self.question_encoder(question)
        return qst_feature




class AMS(nn.Module):
    """
    AMS (Adaptive Multi-scale) 模型类
    结合了多尺度Transformer专家网络和门控机制，用于自适应选择最相关的尺度进行处理
    """

    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1,
                 batch_norm=False):
        """
        初始化AMS模型

        参数:
            input_size (int): 输入特征的维度大小
            output_size (int): 输出特征的维度大小
            num_experts (int): 专家网络的数量（对应不同的patch_size）
            device: 计算设备
            num_nodes (int): 节点数量，默认为1
            d_model (int): Transformer模型的隐藏层维度
            d_ff (int): Transformer前馈网络的维度
            dynamic (bool): 是否使用动态机制
            patch_size (list): 不同尺度的patch大小列表
            noisy_gating (bool): 是否使用噪声门控机制
            k (int): 每次前向传播中激活的专家数量
            layer_number (int): Transformer层数
            residual_connection (int): 是否使用残差连接
            batch_norm (bool): 是否使用批归一化
        """
        super(AMS, self).__init__()
        self.num_experts = num_experts  # 专家网络总数
        self.output_size = output_size  # 输出维度
        self.input_size = input_size  # 输入维度
        self.k = k  # 每次激活的专家数

        # 起始线性层：将多节点特征映射到单节点
        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        # 趋势分解模型：用于分离时间序列的趋势和残差成分
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        # 创建多个专家网络，每个对应不同的patch尺寸
        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)  # 计算patch数量
            # 为每个patch尺寸创建对应的Transformer层
            self.experts.append(Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff,
                                                  dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                                                  patch_size=patch, factorized=True, layer_number=layer_number,
                                                  batch_norm=batch_norm))

        # 门控网络：决定每个输入应该分配给哪些专家
        # self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Linear(input_size, num_experts)  # 噪声权重
        self.w_gate = nn.Linear(input_size, num_experts)  # 门控权重

        self.residual_connection = residual_connection  # 残差连接标志
        # 最终的MLP层：将输出映射到目标维度
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating  # 是否使用噪声门控
        self.softplus = nn.Softplus()  # Softplus激活函数
        self.softmax = nn.Softmax(1)  # Softmax函数
        # 注册缓冲区：用于噪声分布的均值和标准差
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)  # 确保激活专家数不超过总专家数

    def cv_squared(self, x):
        """
        计算变异系数的平方（方差/均值^2），用于衡量负载平衡

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 变异系数的平方
        """
        eps = 1e-10  # 防止除零错误的小值
        if x.shape[0] == 1:  # 如果只有一个样本，返回0
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """
        计算每个专家的负载（被分配样本的数量）

        参数:
            gates (Tensor): 门控权重矩阵

        返回:
            Tensor: 每个专家的负载
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        计算每个专家在top-k中的概率（噪声门控的核心计算）

        参数:
            clean_values (Tensor): 干净的logits值
            noisy_values (Tensor): 加入噪声的logits值
            noise_stddev (Tensor): 噪声标准差
            noisy_top_values (Tensor): 带噪声的top值

        返回:
            Tensor: 每个专家在top-k中的概率
        """
        batch = clean_values.size(0)  # 批次大小
        m = noisy_top_values.size(1)  # 专家数量
        top_values_flat = noisy_top_values.flatten()  # 展平top值

        # 计算阈值位置
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)  # 判断是否在top-k中
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        # 计算正态分布的累积分布函数
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)  # 根据是否在top-k中选择概率
        return prob

    def trend_decompose(self, x):
        """
        对输入进行趋势分解处理

        参数:
            x (Tensor): 输入张量，形状为[B, T, N, D]

        返回:
            Tensor: 趋势分解后的结果
        """
        x = x[:, :, :, 0]  # 取第一个特征维度
        _, trend = self.trend_model(x)  # 获取趋势成分
        return x + trend  # 返回原始值加趋势

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        带噪声的top-k门控机制

        参数:
            x (Tensor): 输入特征
            train (bool): 是否处于训练模式
            noise_epsilon (float): 噪声epsilon值

        返回:
            gates (Tensor): 门控权重
            load (Tensor): 专家负载
        """
        x = self.start_linear(x).squeeze(-1)  # 线性变换并压缩最后一维

        # 计算干净的logits
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            # 计算噪声标准差
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            # 添加噪声得到带噪声的logits
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 计算topk + 1，用于噪声门控
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        # 获取top-k的logits和索引
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)  # softmax归一化

        # 创建稀疏门控矩阵
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 计算专家负载
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, v_q, loss_coef=1e-2):
        """
        前向传播函数

        参数:
            x (Tensor): 输入特征张量
            v_q: 查询向量
            loss_coef (float): 平衡损失系数

        返回:
            output (Tensor): 输出特征
            balance_loss (Tensor): 平衡损失
        """
        new_x = self.trend_decompose(x)  # 趋势分解

        # 多尺度路由：获取门控权重和专家负载
        gates, load = self.noisy_top_k_gating(new_x, self.training)

        # 计算负载平衡损失
        importance = gates.sum(0)  # 每个专家的重要性
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)  # 平衡损失
        balance_loss *= loss_coef  # 乘以损失系数

        # 分发器：将输入分发给对应的专家
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        # 各专家网络处理
        expert_outputs = [self.experts[i](expert_inputs[i], v_q)[0] for i in range(self.num_experts)]

        # 合并专家输出
        output = dispatcher.combine(expert_outputs)

        # 残差连接
        if self.residual_connection:
            output = output + x

        return output, balance_loss


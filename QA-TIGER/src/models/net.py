import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict
from .MCCD.layer import MCCD_MLP  # 导入MCCD的MLP模块

from .encoders import CLIP_TEncoder  # 导入CLIP文本编码器
from .modules import (
    Projection, QstGrounding,  # 导入自定义模块：投影、问题定位
    TempMoE, AVQCrossAttn,  # 导入自定义模块：时序混合专家、音视问交叉注意力
    PatchSelecter  # 导入自定义模块：Patch选择器
)
# from .TWM.net_encoders import AMS  # 导入自适应多尺度稀疏混合专家模型


# 定义QA-TIGER模型类，继承自nn.Module
class QA_TIGER(nn.Module):
    def __init__(self,
                 d_model: int = 512,          # 模型内部的主要特征维度
                 video_dim: int = 512,        # 输入视频特征的原始维度
                 patch_dim: int = 768,        # 输入patch特征的原始维度
                 audio_dim: int = 128,        # 输入音频特征的原始维度
                 topK: int = 3,               # TempMoE模块中的topK参数
                 num_experts: int = 10,       # TempMoE模块中的专家数量
                 late_fusion: bool = False,   # 是否使用后期融合策略 (当前模型结构中未直接体现)
                 nce_loss: bool = False,      # 是否使用NCE损失 (当前模型结构中未直接体现)
                 encoder_type: str = 'ViT-L/14@336px', # CLIP文本编码器的类型
                 mccd=None,
                 **kwargs
    ):
        super(QA_TIGER, self).__init__()

        self.nce_loss = nce_loss  # 存储nce_loss标志
        self.late_fusion = late_fusion  # 存储late_fusion标志
        self.mccd = mccd
        

        # 定义各种输入特征的投影层，将它们投影到统一的d_model维度
        self.audio_proj = Projection(audio_dim, d_model)  # 音频特征投影
        self.video_proj = Projection(video_dim, d_model)  # 视频特征投影
        self.patch_proj = Projection(patch_dim, d_model)  # Patch特征投影
        self.words_proj = Projection(video_dim, d_model)  # 词语特征投影 (维度与video_dim一致，可能笔误或特定设计)
        self.quest_proj = Projection(video_dim, d_model)  # 问题特征投影 (维度与video_dim一致，可能笔误或特定设计)

        # 初始化CLIP文本编码器用于编码问题文本
        self.quest_encoder = CLIP_TEncoder(encoder_type)
        self.quest_encoder.freeze()  # 冻结文本编码器的参数，不参与训练

        # self.a_attn = AVQCrossAttn(d_model, 8)  # 音频-注意力模块
        # self.v_attn = AVQCrossAttn(d_model, 8)  # 视频-注意力模块
        self.a_attn = nn.MultiheadAttention(d_model, 8 )
        self.v_attn = nn.MultiheadAttention(d_model, 8 )
        
        # 定义模型的核心组件
        self.crs_attn = AVQCrossAttn(d_model, 8)  # 音频-视频-问题交叉注意力模块
        self.patch_selecter = PatchSelecter(d_model, 8)  # Patch选择模块
        self.quest_grounding = QstGrounding(d_model, 8)  # 问题定位模块
        # 音频时序混合专家模块，用于聚合音频时序特征
        self.at_aggregator = TempMoE(d_model, 8, topK=topK, n_experts=num_experts)
        # 视频时序混合专家模块，用于聚合视频和patch的时序特征
        self.vt_aggregator = TempMoE(d_model, 8, topK=topK, n_experts=num_experts, vis_branch=True)

        self.head_act = nn.ReLU()  # 最终输出前的激活函数
        self.dropout = nn.Dropout(0.1)  # Dropout层，防止过拟合
        self.head = nn.Linear(d_model, 42)  # 最终的分类头，输出答案类别的logits (42是答案类别数)

        # 初始化权重
        self.audio_proj.apply(self.init_weight)
        self.video_proj.apply(self.init_weight)
        self.words_proj.apply(self.init_weight)
        self.quest_proj.apply(self.init_weight)
        self.patch_proj.apply(self.init_weight)
        self.head.apply(self.init_weight)

        # 引入MCCD模块
        # bias learner
        if mccd is not None and mccd['flag'] is True:
            if mccd['bias_learner']['q_bias']:
                self.q_bias = MCCD_MLP(dimensions=mccd['mlp']['dimensions'])
            if mccd['bias_learner']['a_bias']:
                self.a_bias = MCCD_MLP(dimensions=mccd['mlp']['dimensions'])
            if mccd['bias_learner']['v_bias']:
                self.v_bias = MCCD_MLP(dimensions=mccd['mlp']['dimensions'])




    # 权重初始化方法
    def init_weight(self, m):
        if isinstance(m, nn.Linear):  # 如果是线性层
            nn.init.kaiming_normal_(m.weight)  # 使用Kaiming正态分布初始化权重
            if m.bias is not None:  # 如果存在偏置项
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0

    # 子前向传播函数，用于处理不同来源（例如，正样本、负样本）的输入数据
    def sub_forward(self,
                    reshaped_data: Dict[str, Tensor],  # 输入数据字典
                    prefix: str = ''  # 前缀，用于区分不同来源的数据 (例如 'n_' 表示负样本)
    ):
        # 从数据字典中提取问题、音频、视频和patch特征
        quest = reshaped_data[f'{prefix}quest']
        audio = reshaped_data[f'{prefix}audio']
        video = reshaped_data[f'{prefix}video']
        patch = reshaped_data[f'{prefix}patch'] if f'{prefix}patch' in reshaped_data else None

        # 根据前缀处理问题特征
        if prefix == 'n_':  # 如果是负样本，则问题和词语特征设为None
            words = None
            quest = None
        else:  # 如果是正样本或其他
            # 检查问题特征的数据类型
            if quest.dtype in (torch.float32, torch.float64):
                # 如果问题已经是编码好的特征 (float类型)，则直接使用，并移除多余的维度
                quest = quest.squeeze(1)
                words = None  # 此时没有原始词语序列
            else:
                # 如果问题是原始token ID (int或long类型)，则通过文本编码器进行编码
                quest, words = self.quest_encoder(quest)  # quest是[CLS]特征，words是词语序列特征
                quest = quest.squeeze(1)  # 移除多余的维度

        return quest, words, audio, video, patch

    # 模型的主前向传播函数
    def forward(self, reshaped_data: Dict[str, Tensor]):
        '''
            输入张量的预期形状:
            input audio shape:      [B, T, AC] (批量大小, 时序长度, 音频特征维度)
            input video shape:      [B, T, VC] (批量大小, 时序长度, 视频特征维度) (注释中pos_frames更复杂，但代码中video是2D的)
            input patch shape:      [B, T, P, PC] (批量大小, 时序长度, Patch数量, Patch特征维度)
            input question shape:   [B, D] or [B, SeqLen] (批量大小, 问题特征维度 或 批量大小, 问题序列长度)
        '''
        return_dict = {}  # 用于存储返回结果的字典

        # 调用sub_forward获取处理后的问题、词语、音频、视频和patch特征 (此处prefix为空，处理正样本)
        quest, words, audio, video, patch = self.sub_forward(reshaped_data, prefix='')

        # 特征投影: 将各种模态的特征投影到统一的d_model维度
        audio = self.audio_proj(audio)  # [B, T, D]
        # print(f'audio shape: {audio.shape}')
        video = self.video_proj(video)  # [B, T, D]
        # print(f'video shape: {video.shape}')
        words = self.words_proj(words)  # [B, 77, D] (假设CLIP词序列长度为77)
        quest = self.quest_proj(quest)  # [B, D]
        # print(f'quest shape: {quest.shape}')
        patch = self.patch_proj(patch)  # [B, T, P, D]


        # MCCD模块
        if self.mccd is not None and self.mccd['flag'] is True:
            q_bias_logits, a_bias_logits, v_bias_logits = None, None, None
            if self.mccd['bias_learner']['q_bias']:

                q_bias_logits = self.get_bias_classifier_logits_q(quest)
                # print(f'q_bias_logits shape: {q_bias_logits.shape}')
            if self.mccd['bias_learner']['a_bias']:
                a_bias_logits_pooled = audio.mean(dim=1)
                a_bias_logits = self.get_bias_classifier_logits_a(a_bias_logits_pooled)
                # print(f'a_bias_logits shape: {a_bias_logits.shape}')
            if self.mccd['bias_learner']['v_bias']:
                v_bias_logits_pooled = video.mean(dim=1)
                v_bias_logits = self.get_bias_classifier_logits_v(v_bias_logits_pooled)
                # print(f'v_bias_logits shape: {v_bias_logits.shape}')




        # 增加一个自注意力模块
        # audio,temp = self.a_attn(audio, audio, words)
        # video,temp = self.v_attn(video, video, words)
        audio = audio.transpose(0, 1)  # [T, B, D]
        audio, _ = self.a_attn(audio, audio, audio)
        audio = audio.transpose(0, 1)  # [B, T, D]

        video = video.transpose(0, 1)
        video, _ = self.v_attn(video, video, video)
        video = video.transpose(0, 1)



        # 多模态交互与融合
        # 1. 音频-视频-问题交叉注意力
        audio, video = self.crs_attn(audio, video, words)  # 输出增强后的音频和视频特征: [B, T, D], [B, T, D]
        # 2. Patch选择与融合
        patch = self.patch_selecter(patch, audio, video)  # 基于音频和视频上下文选择并融合patch特征: [B, T, D]
        # 3. 音频时序特征聚合 (基于问题)
        a_global = self.at_aggregator(quest, audio)  # 输出全局音频表征: [B, D]
        # 4. 视频和patch时序特征聚合 (基于问题)
        ap_global, vp_global = self.vt_aggregator(quest, video, patch)  # 输出全局音频-patch和视频-patch表征: [B, D], [B, D]
        # 5. 问题引导的多模态特征融合 (第一层融合视觉相关的全局特征)
        fusion = self.quest_grounding(quest, [ap_global, vp_global])  # [B, D]
        # 6. 问题引导的多模态特征融合 (第二层融合第一层结果和全局音频特征)
        fusion = self.quest_grounding(quest, [fusion.unsqueeze(1), a_global])  # [B, D]

        # 分类头
        fusion = self.head_act(fusion)  # ReLU激活
        output = self.head(fusion)  # 线性层输出最终的分类logits: [B, num_answers]
        return_dict.update({'out': output})  # 将输出添加到返回字典中




        

        # print(f'output shape: {output.shape}')  # 输出形状: [B, num_answers]
        # return return_dict
        return {
            'out': output,
            'fusion_logits': output,
            'q_bias_logits': q_bias_logits,
            'a_bias_logits': a_bias_logits,
            'v_bias_logits': v_bias_logits
        }



    def get_bias_classifier_logits_q(self, inputs):
        '''
        获取问题偏置分类器的logits
        参数:
            inputs: VisualBert的文本输入
        返回:
            问题偏置的logits
        '''
        # que_emb = self.visual_bert(
        #     input_ids=inputs['input_ids'],
        #     position_ids=inputs['position_ids'],
        #     attention_mask=inputs['attention_mask']
        # )
        #que_emb = grad_mul_const(que_emb.pooler_output, 0.0)  # 梯度乘以常数(已注释)
        q_bias_logits = self.q_bias(inputs)  # 通过MLP获取问题偏置logits
        return q_bias_logits

    def get_bias_classifier_logits_a(self, inputs):
        '''
        获取音频偏置分类器的logits
        参数:
            inputs: 音频嵌入表示
        返回:
            音频偏置的logits
        '''
        # audio_emb = self.visual_bert(inputs_embeds=inputs)  # 将音频嵌入输入到VisualBert
        #audio_emb = grad_mul_const(audio_emb.pooler_output, 0.0)  # 梯度乘以常数(已注释)
        a_bias_logits = self.a_bias(inputs)  # 通过MLP获取音频偏置logits
        return a_bias_logits

    def get_bias_classifier_logits_v(self, inputs):
        '''
        获取视频偏置分类器的logits
        参数:
            inputs: 视频嵌入表示
        返回:
            视频偏置的logits
        '''
        # video_emb = self.visual_bert(inputs_embeds=inputs)  # 将视频嵌入输入到VisualBert
        #video_emb = grad_mul_const(video_emb.pooler_output, 0.0)  # 梯度乘以常数(已注释)
        v_bias_logits = self.v_bias(inputs)  # 通过MLP获取视频偏置logits
        return v_bias_logits


        

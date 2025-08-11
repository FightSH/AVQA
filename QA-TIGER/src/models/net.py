import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict

from .TWM.batch_alvs_inter import iterative_sampling_vectorized
from .TWM.batch_alvs_inter import iterative_sampling
from .MCCD.layer import MCCD_MLP  # 导入MCCD的MLP模块
from .TWM.net_encoders import AMS
from .encoders import(CLIP_TEncoder,Hug_Clip_TEncoder,SigLIP_TEncoder,SigLIP2_TEncoder)   # 导入CLIP文本编码器
# from .mamba.modules.mamba_compressor import MambaCompressor
from .mamba_vision.video_mamba_integration import VideoMambaAdapter, MambaEnhancedQATiger
from .mamba_vision.mamba_aggregator import create_mamba_aggregator
from .modules import (
    Projection, QstGrounding,  # 导入自定义模块：投影、问题定位
    TempMoE, AVQCrossAttn,  # 导入自定义模块：时序混合专家、音视问交叉注意力
    PatchSelecter,  # 导入自定义模块：Patch选择器
    
)
from .align_mamba import AlignmentController  # 导入对齐控制器





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
                 use_ams: bool = False,       # 是否使用自适应多尺度稀疏混合专家 (AMS)
                 use_mamba: bool = False,     # 是否使用mamba模块
                 use_video_mamba: bool = False, # 是否使用VideoMamba模块
                 mamba_config: dict = None,   # VideoMamba配置
                 use_mamba_aggregator: bool = False, # 是否使用VideoMamba聚合器
                 mamba_aggregator_config: dict = None, # VideoMamba聚合器配置
                 use_unified_aggregator: bool = True, # 是否使用统一序列聚合器
                 use_alignment: bool = False, # 是否使用跨模态对齐功能
                 alignment_config: dict = None, # 对齐配置参数
                 mccd=None,
                 **kwargs
    ):
        super(QA_TIGER, self).__init__()

        # 保存配置参数
        self.mccd = mccd
        self.use_ams = use_ams
        self.use_mamba = use_mamba
        self.use_video_mamba = use_video_mamba
        self.use_mamba_aggregator = use_mamba_aggregator
        self.use_unified_aggregator = use_unified_aggregator
        self.use_alignment = use_alignment
        
        # 配置验证和对齐模块初始化
        if use_alignment:
            alignment_config = alignment_config or {}
            # 设置默认配置
            default_alignment_config = {
                'enabled': True,
                'strategy': 'reverse',
                'lambda_align': 0.1,
                'ot_eps': 1e-8,
                'mmd_sigma': 1.0,
                'patch_alignment': True,
                'debug_mode': False,
                'memory_efficient': False,
            }
            # 合并用户配置和默认配置
            for key, default_value in default_alignment_config.items():
                alignment_config.setdefault(key, default_value)
            
            # 验证配置并初始化对齐控制器
            try:
                self.alignment_controller = AlignmentController(alignment_config)
            except Exception as e:
                raise ValueError(f"对齐配置无效: {str(e)}")
        else:
            self.alignment_controller = None



        # 定义各种输入特征的投影层，将它们投影到统一的d_model维度
        self.audio_proj = Projection(audio_dim, d_model)  # 音频特征投影
        self.video_proj = Projection(video_dim, d_model)  # 视频特征投影
        self.patch_proj = Projection(patch_dim, d_model)  # Patch特征投影
        self.words_proj = Projection(video_dim, d_model)  # 词语特征投影
        self.quest_proj = Projection(video_dim, d_model)  # 问题特征投影

        # VideoMamba模块
        if use_video_mamba:
            mamba_config = mamba_config or {}
            # 提取VideoMambaAdapter需要的参数
            adapter_params = {
                'd_model': d_model,
                'video_dim': 512,
                'audio_dim': 512,
                'mamba_hidden_dim': mamba_config.get('mamba_hidden_dim', d_model // 2),
                'depths': mamba_config.get('depths', [2, 2, 8, 2]),
                'num_heads': mamba_config.get('num_heads', [4, 8, 16, 32]),
                'drop_path_rate': mamba_config.get('drop_path_rate', 0.1),
                'layer_scale': mamba_config.get('layer_scale', 1e-6),
                'causal': mamba_config.get('causal', False),
            }
            self.video_mamba_adapter = VideoMambaAdapter(**adapter_params)
            
            # VideoMamba特征融合层
            self.mamba_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        

        # 初始化CLIP文本编码器用于编码问题文本
        if encoder_type == 'ViT-L/14@336px':
            self.quest_encoder = CLIP_TEncoder(encoder_type)
        if encoder_type == 'openai/clip-vit-large-patch14':
            self.quest_encoder = Hug_Clip_TEncoder(encoder_type)
        if encoder_type == 'google/siglip-so400m-patch14-384':
            self.quest_encoder = SigLIP_TEncoder(model_name= encoder_type)
        if encoder_type == 'google/siglip2-so400m-patch14-384':
            self.quest_encoder = SigLIP2_TEncoder(model_name= encoder_type)

        self.quest_encoder.freeze()  # 冻结文本编码器的参数，不参与训练


        # self.temporal_aggregator = PAVEModuleV5(input_dim=d_model, output_dim=d_model, embed_dim=512)
        # self.temporal_aggregator2 = PAVEModuleV5(input_dim=d_model, output_dim=d_model, embed_dim=512)

        
        # 定义模型的核心组件
        self.crs_attn = AVQCrossAttn(d_model, 8)  # 音频-视频-问题交叉注意力模块
        self.patch_selecter = PatchSelecter(d_model, 8)  # Patch选择模块
        self.quest_grounding = QstGrounding(d_model, 8)  # 问题定位模块
        
        # 聚合器选择：统一序列聚合器 vs VideoMamba聚合器 vs 传统TempMoE
        if use_unified_aggregator:
            # 使用新的统一序列聚合器
            mamba_agg_config = mamba_aggregator_config or {}
            
            # 统一聚合器 - 处理所有模态
            self.unified_aggregator = create_mamba_aggregator(
                aggregator_type="unified_single",
                d_model=d_model,
                mamba_config=mamba_agg_config
            )
            
            # 不需要单独的at_aggregator和vt_aggregator
            self.at_aggregator = None
            self.vt_aggregator = None
            
        elif use_mamba_aggregator:
            # 使用VideoMamba聚合器
            mamba_agg_config = mamba_aggregator_config or {}
            
            # 音频聚合器
            self.at_aggregator = create_mamba_aggregator(
                aggregator_type="single",
                d_model=d_model,
                mamba_config=mamba_agg_config
            )
            
            # 视频+patch聚合器
            self.vt_aggregator = create_mamba_aggregator(
                aggregator_type="dual",
                d_model=d_model,
                mamba_config=mamba_agg_config
            )
            
            self.unified_aggregator = None
        else:
            # 使用传统TempMoE聚合器
            # 音频时序混合专家模块，用于聚合音频时序特征
            self.at_aggregator = TempMoE(d_model, 8, topK=topK, n_experts=num_experts)
            # 视频时序混合专家模块，用于聚合视频和patch的时序特征
            self.vt_aggregator = TempMoE(d_model, 8, topK=topK, n_experts=num_experts, vis_branch=True)
            
            self.unified_aggregator = None

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

        # # 引入AMS
        # if self.use_ams:
        #      self.ams =AMS(
        #     input_size=512,
        #     output_size=512,
        #     seq_lenth=60,
        #     num_experts=4,
        #     d_model=512,
        #     d_ff=64,
        #     patch_size=[15, 10, 6, 3],
        #     k=4)

        # if use_mamba:
            # self.video_mamba = MambaCompressor(d_model=512, n_layer=1)
            # self.patch_mamba = MambaCompressor(d_model=512, n_layer=1)

        # self.feature_adjuster = FeatureAdjuster()



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
        # logger.debug(f'input reshaped_data keys: {reshaped_data.keys()}')  # 打印输入数据的键

        # 调用sub_forward获取处理后的问题、词语、音频、视频和patch特征 (此处prefix为空，处理正样本)
        quest, words, audio, video, patch = self.sub_forward(reshaped_data, prefix='')

        # audio = audio[:, ::2, :]      # shape: [B, T//2, D]
        # video = video[:, ::2, :]      # shape: [B, T//2, D]
        # patch = patch[:, ::2, :, :]   # shape: [B, T//2, P, D]
        # audio = audio.repeat_interleave(2, dim=1)
        
        # Projection
        audio = self.audio_proj(audio) # [B, T, D]
        video = self.video_proj(video) # [B, T, D]
        words = self.words_proj(words) # [B, 77, D]
        quest = self.quest_proj(quest) # [B, D]
        patch = self.patch_proj(patch) # [B, T, P, D]

        # VideoMamba增强处理
        if self.use_video_mamba:
            # 获取原始输入用于VideoMamba处理
            original_video = video  # [B, T, video_dim]
            original_audio = audio  # [B, T, audio_dim]
            
            # 通过VideoMamba获取增强特征
            mamba_features = self.video_mamba_adapter(original_video, original_audio)  # [B, T, d_model]
            
            # 融合原始投影特征和Mamba增强特征
            # 音频增强
            audio_combined = torch.cat([audio, mamba_features], dim=-1)  # [B, T, 2*d_model]
            audio = self.mamba_fusion(audio_combined)  # [B, T, d_model]
            
            # 视频增强
            video_combined = torch.cat([video, mamba_features], dim=-1)  # [B, T, 2*d_model]
            video = self.mamba_fusion(video_combined)  # [B, T, d_model]
            
            # Patch增强 - 广播Mamba特征到所有patch
            # B, T, P, D = patch.shape
            # mamba_expanded = mamba_features.unsqueeze(2).expand(B, T, P, D)  # [B, T, P, d_model]
            # patch_combined = torch.cat([patch, mamba_expanded], dim=-1)  # [B, T, P, 2*d_model]
            # patch_combined_flat = patch_combined.view(B * T * P, -1)
            # patch_enhanced_flat = self.mamba_fusion(patch_combined_flat)
            # patch = patch_enhanced_flat.view(B, T, P, D)  # [B, T, P, d_model]

        # audio,video = self.feature_adjuster(audio, video)

        # 对齐处理（新增）
        alignment_loss = torch.tensor(0.0, device=audio.device)
        alignment_debug_info = None
        if self.use_alignment and self.alignment_controller is not None:
            try:
                # print(f"alignbefore audio shape: {audio.shape}")
                # print(f"alignbefore video shape: {video.shape}")
                # print(f"alignbefore words shape: {words.shape}")
                # print(f"alignbefore quest shape: {quest.shape}")
                # print(f"alignbefore patch shape: {patch.shape}")
                audio, video, words, quest, patch, alignment_loss, alignment_debug_info = \
                    self.alignment_controller(audio, video, words, quest, patch)
                
                # print(f"align audio shape: {audio.shape}")
                # print(f"align video shape: {video.shape}")
                # print(f"align words shape: {words.shape}")
                # print(f"align quest shape: {quest.shape}")
                # print(f"align patch shape: {patch.shape}")
                # print(f"alignment_loss: {alignment_loss}")
            except Exception as e:
                # 如果对齐失败，记录错误但继续执行
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"对齐处理失败，继续使用原始特征: {str(e)}")
                alignment_loss = torch.tensor(0.0, device=audio.device)

        q_bias_logits, a_bias_logits, v_bias_logits = None, None, None
        # MCCD模块
        if self.mccd is not None and self.mccd['flag'] is True:
            if self.mccd['bias_learner']['q_bias']:
                q_bias_logits = self.get_bias_classifier_logits_q(quest)
                # logger.debug(f'q_bias_logits shape: {q_bias_logits.shape}')
            if self.mccd['bias_learner']['a_bias']:
                a_bias_logits_pooled = audio.mean(dim=1)
                a_bias_logits = self.get_bias_classifier_logits_a(a_bias_logits_pooled)
                # logger.debug(f'a_bias_logits shape: {a_bias_logits.shape}')
            if self.mccd['bias_learner']['v_bias']:
                v_bias_logits_pooled = video.mean(dim=1)
                v_bias_logits = self.get_bias_classifier_logits_v(v_bias_logits_pooled)
                # logger.debug(f'v_bias_logits shape: {v_bias_logits.shape}')




        # frame_num = torch.tensor([60], device='cuda:0')
        # fast = audio.unsqueeze(2).unsqueeze(3)
        # video = video.unsqueeze(2)
        # video = video + self.temporal_aggregator(fast,frame_num=frame_num,chunk_num=60,slow_feats=video)
        # video = video.squeeze(2)
        #
        #
        #
        # patch = patch + self.temporal_aggregator2(fast,frame_num=frame_num,chunk_num=60,slow_feats=patch)
        
        # logger.debug(f'patchOr shape: {patchOr.shape}')  # 输出patchOr的形状
        # logger.debug(f'融合后patchOr shape: {fusion_patch.shape}')
        if self.use_ams:
            # 打印形状信息以便调试
            # print(f"video shape: {video.shape}")
            # print(f"quest shape: {quest.shape}")
            # print(f"audio shape: {audio.shape}")
            indices_batch = iterative_sampling(video, quest, 11, 8, 0.8, 0.2, 15)
            # print(f"indices_batch type: {type(indices_batch)}, length: {len(indices_batch)}")
            # if len(indices_batch) > 0:
            #     print(f"indices_batch[0] type: {type(indices_batch[0])}, length: {len(indices_batch[0])}")
            
            indices_tensor = torch.tensor(indices_batch)  # [batch_size, target_frames]
            # print(f"indices_tensor shape: {indices_tensor.shape}")
            
            batch_indices = torch.arange(video.size(0)).unsqueeze(1)  # [batch_size, 1]
            # print(f"batch_indices shape: {batch_indices.shape}")
            
            selected_video = video[batch_indices, indices_tensor]  # [batch_size, target_frames, feature_dim]
            # print(f"selected_video shape: {selected_video.shape}")
            # print(f"audio shape: {audio.shape}")
            
            audio,_ = self.ams(audio,selected_video)
            # video=selected_video
            # patch=video[batch_indices, indices_tensor]





        # 多模态交互与融合
        # 1. 音频-视频-问题交叉注意力
        audio, video = self.crs_attn(audio, video, words)  # 输出增强后的音频和视频特征: [B, T, D], [B, T, D]
        # 2. Patch选择与融合
        patch = self.patch_selecter(patch, audio, video)  # 基于音频和视频上下文选择并融合patch特征: [B, T, D]

        # 3. 时序特征聚合 (基于问题)
        if self.use_unified_aggregator:
            # 使用新的统一序列聚合器
            # patch_selecter返回的是list [audio_patch, video_patch]，我们需要将其合并
            if isinstance(patch, list):
                # 如果patch是list，取平均或选择一个
                patch_unified = (patch[0] + patch[1]) / 2  # [B, T, D] 简单平均
            else:
                patch_unified = patch  # [B, T, D]
            
            # 统一聚合器处理所有模态
            unified_global = self.unified_aggregator(quest, video, audio, patch_unified)  # [B, 1, D]
            # print(f"unified_global shape: {unified_global.shape}")
            unified_global = unified_global.squeeze(1)
            
            # print(f"unified_global shape: {unified_global.shape}")
            fusion = self.head_act(unified_global)  # ReLU激活
            output = self.head(fusion)  # 线性层输出最终的分类logits: [B, num_answers]
            return_dict.update({'out': output})  # 将输出添加到返回字典中
            return {
                'out': output,
                'fusion_logits': output,
                'alignment_loss': alignment_loss,
                'alignment_debug_info': alignment_debug_info,
                'q_bias_logits': q_bias_logits,
                'a_bias_logits': a_bias_logits,
                'v_bias_logits': v_bias_logits
            }


            # 为了兼容后续的quest_grounding，我们需要创建三个特征
            # 这里我们将统一的全局特征复制三份，但通过不同的线性层处理以产生差异
            # if not hasattr(self, 'unified_split_heads'):
            #     # 动态创建分割头（只在第一次使用时创建）
            #     self.unified_split_heads = nn.ModuleList([
            #         nn.Linear(d_model, d_model),
            #         nn.Linear(d_model, d_model),
            #         nn.Linear(d_model, d_model)
            #     ]).to(unified_global.device)
            
            # unified_squeezed = unified_global.squeeze(1)  # [B, D]
            # a_global = self.unified_split_heads[0](unified_squeezed).unsqueeze(1)  # [B, 1, D]
            # ap_global = self.unified_split_heads[1](unified_squeezed).unsqueeze(1)  # [B, 1, D]
            # vp_global = self.unified_split_heads[2](unified_squeezed).unsqueeze(1)  # [B, 1, D]
            
        elif self.use_mamba_aggregator:
            # 使用VideoMamba聚合器
            a_global = self.at_aggregator(quest, audio=audio)  # 输出全局音频表征: [B, 1, D]
            # 4. 视频和patch时序特征聚合 (基于问题)
            ap_global, vp_global = self.vt_aggregator(quest, video, patch)  # 输出全局表征: [B, 1, D], [B, 1, D]
            
            # 保持[B, 1, D]格式以匹配quest_grounding的期望输入
            # quest_grounding期望3维输入: [B, Seq, D]
        else:
            # 使用传统TempMoE聚合器
            a_global = self.at_aggregator(quest, audio)  # 输出全局音频表征: [B, 1, D]
            # 4. 视频和patch时序特征聚合 (基于问题)
            ap_global, vp_global = self.vt_aggregator(quest, video, patch)  # 输出全局音频-patch和视频-patch表征: [B, 1, D], [B, 1, D]
            
            # 保持[B, 1, D]格式以匹配quest_grounding的期望输入

        # 5. 问题引导的多模态特征融合 (第一层融合视觉相关的全局特征)
        # ap_global和vp_global都是[B, 1, D]格式，符���quest_grounding的期望
        fusion = self.quest_grounding(quest, [ap_global, vp_global])  # [B, D]
        # 6. 问题引导的多模态特征融合 (第二层融合第一层结果和全局音频特征)
        # fusion是[B, D]，需要unsqueeze；a_global是[B, 1, D]，直接使用
        fusion = self.quest_grounding(quest, [fusion.unsqueeze(1), a_global])  # [B, D]
        # 分类头
        fusion = self.head_act(fusion)  # ReLU激活
        output = self.head(fusion)  # 线性层输出最终的分类logits: [B, num_answers]
        return_dict.update({'out': output})  # 将输出添加到返回字典中




        

        # logger.debug(f'output shape: {output.shape}')  # 输出形状: [B, num_answers]
        # return return_dict
        return {
            'out': output,
            'fusion_logits': output,
            'alignment_loss': alignment_loss,
            'alignment_debug_info': alignment_debug_info,
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


        

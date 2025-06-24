# 该脚本实现PAVE的结构，基于Qwen2语言模型

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from libs.model.pave_arch import PAVEMetaModel, PAVEMetaForCausalLM


class PAVEQwen2Config(Qwen2Config):
    """
    PAVE模型配置类，继承自Qwen2配置
    用于定义PAVE模型的配置参数
    """
    model_type = "pave_qwen2"


class PAVEQwen2Model(PAVEMetaModel, Qwen2Model):
    """
    PAVE Qwen2模型类
    继承自PAVEMetaModel（提供多模态功能）和Qwen2Model（提供语言模型基础）
    """
    config_class = PAVEQwen2Config

    def __init__(self, config: Qwen2Config):
        super(PAVEQwen2Model, self).__init__(config)
        

class PAVEQwen2ForCausalLM(Qwen2ForCausalLM, PAVEMetaForCausalLM):
    """
    PAVE Qwen2因果语言模型
    用于生成任务，结合了视频理解和语言生成能力
    
    主要功能：
    1. 处理多模态输入（视频、图像、文本）
    2. 通过交叉注意力机制融合不同模态特征
    3. 生成基于多模态理解的文本回答
    """
    config_class = PAVEQwen2Config

    def __init__(self, config):
        # 初始化语言模型基础组件
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = PAVEQwen2Model(config)  # 核心模型
        self.vocab_size = config.vocab_size  # 词汇表大小
        # 语言模型头部，用于生成词汇概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_model(self):
        """获取核心模型实例"""
        return self.model

    def forward(
        self,
        # === 文本输入参数 ===
        input_ids: torch.LongTensor = None,           # 输入文本的token ID
        labels: Optional[torch.LongTensor] = None,    # 训练时的标签
        
        # === 视频特征参数（快速路径）===
        video_feats: Optional[torch.FloatTensor] = None,     # 视频特征 (B, C, T, H, W)
        video_feat_fps: Optional[torch.FloatTensor] = None,  # 视频帧率
        feat_frame_nums: Optional[torch.FloatTensor] = None, # 每个样本的帧数
        
        # === 图像特征参数（慢速路径）===
        images: Optional[torch.FloatTensor] = None,          # 图像特征列表
        image_sizes: Optional[List[List[int]]] = None,       # 图像尺寸信息
        modalities: Optional[List[str]] = ["video"],         # 模态类型列表
        
        # === 标准Transformer参数 ===
        attention_mask: Optional[torch.Tensor] = None,      # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,    # 位置编码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 缓存的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,   # 直接输入的嵌入
        use_cache: Optional[bool] = None,                    # 是否使用缓存
        output_attentions: Optional[bool] = None,           # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,        # 是否输出隐藏状态
        return_dict: Optional[bool] = None,                 # 是否返回字典格式
        
        # === PAVE特定参数 ===
        video_metas = None,                                  # 视频元数据
        question_ids = None,                                 # 问题token ID
        question_lens = None,                               # 问题长度
        dpo_forward = False,                                # 是否为DPO前向传播
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        PAVE模型的前向传播函数
        
        数据流程：
        1. 如果inputs_embeds为None，说明是训练或推理的第一次前向传播
        2. 调用prepare_inputs_labels_for_multimodal处理多模态输入
        3. 执行标准的语言模型前向传播
        4. 检查并处理NaN损失
        
        Args:
            input_ids: 文本token ID，形状 (B, seq_len)
            video_feats: 视频特征，形状 (B, C, T, H, W)
            images: 图像特征列表，每个元素形状 (T, patch_num, dim)
            labels: 训练标签，形状 (B, seq_len)
            其他参数: 标准Transformer参数
            
        Returns:
            训练时返回损失，推理时返回logits和labels
        """

        # 第一步：多模态输入预处理
        if inputs_embeds is None:  # 训练或推理的第一次前向传播
            (
                input_ids,           # 更新后的输入ID
                position_ids,        # 位置编码
                attention_mask,      # 注意力掩码
                past_key_values,     # 键值对缓存
                inputs_embeds,       # 融合后的多模态嵌入
                labels              # 更新后的标签
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                video_feats,                    # 快速路径特征
                video_feat_fps=video_feat_fps,
                feat_frame_nums=feat_frame_nums,
                question_ids=question_ids,
                question_lens=question_lens,
                images=images,                  # 慢速路径特征
                image_sizes=image_sizes,
                modalities=modalities,
                video_metas=video_metas,
            )

        # 第二步：执行语言模型前向传播
        if not dpo_forward:  # 标准前向传播（训练/推理）
            loss = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,     # 融合了视频和图像特征的嵌入
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            
            # 第三步：NaN损失检查和处理
            if loss.loss is not None and torch.isnan(loss.loss):
                print('检测到NaN损失，批次视频信息:', video_metas)
                import time
                # 保存出错的视频元数据用于调试
                torch.save(video_metas, 'error_' + str(time.time()) + '.pt')
                raise NotImplementedError
            
            return loss
            
        else:  # DPO（Direct Preference Optimization）前向传播
            # 设置输出配置
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict            
                
            # 获取模型输出（不计算损失）
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            # 通过语言模型头部生成logits
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels            

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,              # 输入token
        video_feats: Optional[torch.Tensor] = None,         # 视频特征
        video_feat_fps: Optional[torch.FloatTensor] = None, # 视频帧率
        feat_frame_nums: Optional[torch.FloatTensor] = None, # 帧数

        images: Optional[torch.FloatTensor] = None,         # 图像特征
        image_sizes: Optional[List[List[int]]] = None,      # 图像尺寸
        modalities: Optional[List[str]] = ["video"],        # 模态类型
        
        question_ids = None,        # 问题ID
        question_lens = None,       # 问题长度
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        推理生成函数
        
        处理流程：
        1. 提取位置编码和注意力掩码
        2. 如果有视频或图像输入，调用多模态预处理
        3. 否则直接使用文本嵌入
        4. 调用父类的生成方法
        
        Args:
            inputs: 输入token
            video_feats: 视频特征
            images: 图像特征
            其他参数: 生成相关参数
            
        Returns:
            生成的文本token序列
        """
        # 提取生成参数
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` 参数暂不支持")

        # 处理多模态输入
        if video_feats is not None or images is not None:  # 如果有视觉输入
            (
                inputs,              # 更新后的输入
                position_ids,        # 位置编码
                attention_mask,      # 注意力掩码
                _,                   # 键值对缓存（推理时为None）
                inputs_embeds,       # 多模态嵌入
                _                    # 标签（推理时为None）
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,               # past_key_values
                None,               # labels
                video_feats,
                video_feat_fps=video_feat_fps,
                feat_frame_nums=feat_frame_nums,
                question_ids=question_ids,
                question_lens=question_lens,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
            )
        else:
            # 纯文本输入，直接获取词嵌入
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类生成方法
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.half(),  # 转换为半精度
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        为生成准备输入
        处理视频和图像输入的传递
        
        Args:
            input_ids: 输入token ID
            past_key_values: 缓存的键值对
            inputs_embeds: 输入嵌入
            **kwargs: 其他参数，包括videos和video_sizes
            
        Returns:
            为生成准备好的输入字典
        """
        # 提取视频相关参数
        videos = kwargs.pop("videos", None)
        video_sizes = kwargs.pop("video_sizes", None)
        
        # 调用父类方法准备基础输入
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        
        # 添加视频相关输入
        if videos is not None:
            inputs['videos'] = videos
        if video_sizes is not None:
            inputs['video_sizes'] = video_sizes
        return inputs


# 注册PAVE Qwen2配置和模型
AutoConfig.register("pave_qwen2", PAVEQwen2Config)
AutoModelForCausalLM.register(PAVEQwen2Config, PAVEQwen2ForCausalLM)

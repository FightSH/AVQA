# This script holds the implementation of the PAVE model.
# 该脚本包含了 PAVE 模型的实现。

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import math
import ipdb
import logging
from einops import rearrange, repeat

from .multimodal_encoder.builder import build_temporal_aggregator, build_video_tower, build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libs.mm_utils import get_anyres_image_grid_shape, split_list_lengths
from libs.utils.train_utils import rank0_print


def get_weight(weights, keyword):
    """一个辅助函数，用于从预训练模型的 state_dict 中提取特定模块的权重。"""
    # {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
    # 遍历权重字典，如果键（k）中包含指定的关键字（keyword），
    # 就将关键字及其前面的部分（如 'model.vision_tower.'）去掉，
    # 然后将剩余的键和对应的值（v）存入新的字典中。
    return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}


class PAVEMetaModel:
    """
    PAVE 模型的元类（Meta Model）。
    这个类主要负责定义和初始化模型的各个组件，如视觉塔、视频塔等。
    它不包含完整的 forward 逻辑，更像是一个模型结构的容器。
    """

    def __init__(self, config):
        super(PAVEMetaModel, self).__init__(config)
        print("开始初始化PAVE元模型")

        # The Fast-Path (快速路径，通常用于处理视频流)
        if hasattr(config, "mm_video_tower"):
            print("检测到mm_video_tower配置，初始化快速路径组件")
            #build the encoder (MAE + diffusion)
            # 构建视频编码器（例如，结合了 MAE 和 diffusion 的模型）
            print("构建视频塔，delay_load=True")
            self.video_tower = build_video_tower(config, delay_load=True) # delay_load=True 表示稍后才会真正加载权重
            #TODO: build the compresser (SSM) We may not need to instantiate it now
            #TODO: 构建压缩器 (SSM)，这里可能暂时不需要实例化
            # 构建时间聚合器，用于沿时间维度聚合视频帧特征
            print("构建时间聚合器")
            self.temporal_aggregator = build_temporal_aggregator(config)  # 第一次构建
        else:
            print("未检测到mm_video_tower配置，跳过快速路径初始化")

        # The Slow-Path (慢速路径，通常用于处理静态图像或关键帧)
        if hasattr(config, "mm_vision_tower"):
            print("检测到mm_vision_tower配置，初始化慢速路径组件")
            # delay_load 表示是否延迟加载模型权重，默认为 False
            delay_load = getattr(config, "delay_load", False)
            print(f"构建视觉塔，delay_load={delay_load}")
            # 构建视觉塔（用于处理图像）
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            # self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            # 构建多模态投影器，将视觉特征投影到与文本特征相同的维度空间
            print("构建多模态投影器")
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            # 如果 patch 合并类型包含 "unpad"，则需要一个额外的可学习参数作为图像换行符
            mm_patch_merge_type = getattr(config, "mm_patch_merge_type", "")
            if "unpad" in mm_patch_merge_type:
                print(f"检测到unpad类型patch合并({mm_patch_merge_type})，初始化image_newline参数")
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
            else:
                print(f"patch合并类型为{mm_patch_merge_type}，不需要image_newline参数")
        else:
            print("未检测到mm_vision_tower配置，跳过慢速路径初始化")
        
        print("PAVE元模型初始化完成")

    def get_video_tower(self):
        """获取视频塔实例。处理 FSDP（Fully Sharded Data Parallel）可能将其包装在列表中的情况。"""
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            print("视频塔被FSDP包装在列表中，提取第一个元素")
            video_tower = video_tower[0]
        return video_tower

    def get_vision_tower(self):
        """获取视觉塔实例。同样处理 FSDP 的情况。"""
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            print("视觉塔被FSDP包装在列表中，提取第一个元素")
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """
        初始化模型的视觉相关模块。
        这个函数负责构建模块、加载预训练权重等。
        """
        print("开始初始化视觉模块")
        
        #### The Fast-path Init ### (快速路径初始化)
        video_tower = model_args.video_tower # 从模型参数中获取视频塔的配置或路径
        pretrain_temporal_aggregator = model_args.pretrain_temporal_aggregator # 获取预训练的时间聚合器权重路径
        self.config.mm_video_tower = video_tower # 更新模型配置
        
        print(f"快速路径配置: video_tower={video_tower}, pretrain_temporal_aggregator={pretrain_temporal_aggregator}")

        ### init and load the pretrained video tower backbone (初始化并加载预训练的视频塔骨干网络)
        if self.get_video_tower() is None:
            print("视频塔未初始化，开始构建")
            # 如果视频塔尚未初始化，则根据配置构建它
            video_tower = build_video_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                print("使用FSDP，将视频塔包装在列表中")
                # 如果使用 FSDP，将模型包装在列表中
                self.video_tower = [video_tower]
            else:
                print("不使用FSDP，直接设置视频塔")
                self.video_tower = video_tower
        else:
            print("视频塔已存在，加载预训练权重")
            # 如果视频塔已存在，则获取其实例
            if fsdp is not None and len(fsdp) > 0:
                video_tower = self.video_tower[0]
            else:
                video_tower = self.video_tower
            # load the checkpoint (for step 1 and step 2)
            # 加载预训练权重（用于训练的第一步和第二步）
            video_tower.load_model()

        ### build the temporal aggregator again (再次构建时间聚合器)
        self.config.temporal_aggregator_type = getattr(model_args, 'temporal_aggregator_type', 'ssm')
        print(f"时间聚合器类型: {self.config.temporal_aggregator_type}")
        
        if getattr(self, 'temporal_aggregator', None) is None:
            print("时间聚合器未初始化，开始构建")
            # 如果时间聚合器未初始化，则构建它
            self.temporal_aggregator = build_temporal_aggregator(model_args)  # 第二次构建
        else:
            print("时间聚合器已存在，解冻参数以进行训练")
            # In case it is frozen by LoRA
            # 如果它因为 LoRA 等技术被冻结了，需要解冻以进行训练
            for p in self.temporal_aggregator.parameters():
                p.requires_grad = True

        # load the existing temporal aggregator checkpoint (for step 2 training)
        # 加载已有的时间聚合器权重（用于训练的第二步）
        if pretrain_temporal_aggregator is not None:
            print(f"加载预训练时间聚合器权重: {pretrain_temporal_aggregator}")
            temporal_aggregator_weights = torch.load(pretrain_temporal_aggregator, map_location='cpu')
            # ipdb.set_trace()
            def get_w(weights, keyword, module):
                """一个内部函数，用于加载权重并处理可能不匹配的键。"""
                print(f"提取关键字'{keyword}'的权重")
                # 提取特定模块的权重
                temp = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                
                # handle the running means (处理 state_dict 中的 running_mean/var 等)
                module_state_dict = module.state_dict()
                # check the loading (检查加载情况)
                if len(module_state_dict) >= len(temp):
                    # 找出当前模块有但加载权重中没有的键（可能需要重新初始化）
                    missed_key = [ele for ele in module_state_dict if ele not in temp]
                    
                    if missed_key:
                        print(f"发现缺失的权重键: {missed_key}")
                    
                    # hacky way to get the data type (用一种取巧的方式获取权重的数据类型)
                    data_type = None
                    for key in temp:
                        data_type = temp[key].dtype
                        break
                    
                    # 对缺失的键，使用当前模块的初始值，并转换到正确的类型
                    for key in missed_key:
                        print(f'参数 {key} 被重新初始化')
                        temp[key] = module_state_dict[key].to(data_type)  
                return temp
            
            # 加载时间聚合器的权重
            loaded_weights = get_w(temporal_aggregator_weights, 'temporal_aggregator', self.temporal_aggregator)
            self.temporal_aggregator.load_state_dict(loaded_weights)
            print("时间聚合器权重加载完成")
        else:
            print("未提供预训练时间聚合器权重，使用随机初始化")

        ### The Slow-path Init. This section should be skipped ### (慢速路径初始化。此部分应被跳过)
        # ipdb.set_trace() # check this part is skipped (检查这部分代码是否按预期被跳过)
        if hasattr(model_args, 'vision_tower') and self.get_vision_tower() is None:
            print('重新加载/重新初始化慢速路径中的vision_tower')
            # 这部分逻辑用于初始化慢速路径（图像处理），但在当前设置下可能不会执行
            vision_tower = model_args.vision_tower
            mm_vision_select_layer = model_args.mm_vision_select_layer
            mm_vision_select_feature = model_args.mm_vision_select_feature
            pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter     # check whether we train the adapter in the step1 training
            mm_patch_merge_type = model_args.mm_patch_merge_type
            pretrain_vision_modules = model_args.pretrain_vision_modules     # Considering the case we directly load the whole vision module from other model
            # ipdb.set_trace() # check pretrain_vision_modules

            self.config.mm_vision_tower = vision_tower
            self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

            # Load the vision backbone (Image Backbone)
            # 加载视觉骨干网络（图像骨干）
            if self.get_vision_tower() is None:
                vision_tower = build_vision_tower(model_args)
                # vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
                # for k, v in vision_resampler.config.items():
                #     setattr(self.config, k, v)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                    # self.vision_resampler = [vision_resampler]
                else:
                    self.vision_tower = vision_tower
                    # self.vision_resampler = vision_resampler
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                    # vision_resampler = self.vision_resampler[0]
                else:
                    vision_tower = self.vision_tower
                    # vision_resampler = self.vision_resampler
                # if pretrain_vision_modules is not None: # if we has the pretrain model then further delay the loading
                vision_tower.load_model()

            self.config.use_mm_proj = True
            self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
            # self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
            self.config.mm_vision_select_layer = mm_vision_select_layer
            self.config.mm_vision_select_feature = mm_vision_select_feature
            self.config.mm_patch_merge_type = mm_patch_merge_type
            
            # 如果配置了要添加快速视频流的token，则初始化一个可学习的 `faster_token`
            if not hasattr(self.config, 'add_faster_video'):
                if hasattr(model_args, 'add_faster_video')  and model_args.add_faster_video:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.faster_token = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            
            # 初始化多模态投影器
            if getattr(self, "mm_projector", None) is None:
                self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

                if "unpad" in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
            else:
                # In case it is frozen by LoRA
                # 如果被 LoRA 冻结，解冻参数
                for p in self.mm_projector.parameters():
                    p.requires_grad = True

            # load the adaptor (加载适配器权重)
            if pretrain_mm_mlp_adapter is not None:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

                incompatible_keys = self.mm_projector.load_state_dict(get_weight(mm_projector_weights, "mm_projector"))
                print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
                incompatible_keys = self.vision_resampler.load_state_dict(get_weight(mm_projector_weights, "vision_resampler"), strict=False)
                print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            
            # ipdb.set_trace()
            # load the vision backbone, adaptor and the self.image_newline
            # 加载视觉骨干网络、适配器和 image_newline
            if pretrain_vision_modules is not None:
                assert pretrain_mm_mlp_adapter is None, "You give the pretrain_mm_mlp_adapter and pretrain_vision_modules at the same time"
                # load the full model (加载完整的视觉模块权重)
                whole_vision_weights = torch.load(pretrain_vision_modules, map_location="cpu")
                
                # load the backbone #'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight' =>  vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight
                # 加载骨干网络权重，注意这里对权重键（key）进行了处理，以匹配当前模型的结构
                incompatible_keys = self.vision_tower.load_state_dict({'.'.join(k.split(".")[2:]): v for k, v in whole_vision_weights.items() if "vision_tower" in k})
                print(f"ReLoaded vision_tower weights from {pretrain_vision_modules}. Incompatible keys: {incompatible_keys}")
                
                # load the adaptor (加载适配器)
                incompatible_keys = self.mm_projector.load_state_dict(get_weight(whole_vision_weights, "mm_projector"))
                print(f"Loaded mm projector weights from {pretrain_vision_modules}. Incompatible keys: {incompatible_keys}")
                
                # load the newline (加载 image_newline 权重)
                self.image_newline.load_state_dict(whole_vision_weights['model.image_newline'])
                print(f'Loaded image_newline weights from {pretrain_vision_modules}.')

        ## handle other config (处理其他配置)
        self.config.mm_newline_position = model_args.mm_newline_position
        self.config.feat_combine_method = model_args.feat_combine_method
        self.config.train_addition_start_end_tokens = model_args.train_addition_start_end_tokens
        
        print(f"其他配置设置完成: mm_newline_position={self.config.mm_newline_position}, "
                   f"feat_combine_method={self.config.feat_combine_method}, "
                   f"train_addition_start_end_tokens={self.config.train_addition_start_end_tokens}")
        print("视觉模块初始化完成")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.
    对一个经过填充和缩放的图像张量进行“去填充”操作，恢复其原始宽高比。

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
                           图像张量，格式应为 CxHxW。
    original_size (tuple): The original size of PIL image (width, height).
                           原始 PIL 图像的尺寸 (宽度, 高度)。

    Returns:
    torch.Tensor: The unpadded image tensor.
                  去填充后的图像张量。
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        # 如果原始图像更“宽”，说明填充是在高度上
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        # 如果原始图像更“高”，说明填充是在宽度上
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class PAVEMetaForCausalLM(ABC):
    """
    一个抽象基类（ABC），为因果语言模型（CausalLM）集成 PAVE 模型。
    它定义了处理多模态输入的通用接口和核心逻辑。
    """

    @abstractmethod
    def get_model(self):
        """抽象方法，子类必须实现，用于返回底层的语言模型实例。"""
        pass

    def get_video_tower(self):
        """获取视频塔。"""
        return self.get_model().get_video_tower()

    def get_vision_tower(self):
        """获取视觉塔。"""
        return self.get_model().get_vision_tower()

    def encode_videos(self, 
                      video_feats, 
                      q_text_embeds=None, 
                      video_feat_fps=None, 
                      feat_frame_nums=None,
                      q_text_nums=None,
                      chunk_num=None,
                      slow_feats=None):
        """
        编码视频特征。这是"快速路径"的核心。
        """
        print("开始编码视频特征（快速路径）")
        print(f"输入形状: video_feats={video_feats.shape if video_feats is not None else None}, "
                    f"feat_frame_nums={feat_frame_nums}, chunk_num={chunk_num}")
        
        # Using the question text embedding in the diffusion module of the OpenSora
        # we add one extra layer in temporal_aggregator to keep all trainable params in temporal_aggregator
        # 如果时间聚合器中包含 diffusion_mlp，则使用问题文本嵌入作为 diffusion 模块的条件输入
        if hasattr(self.get_model().temporal_aggregator, 'diffusion_mlp') and self.get_model().temporal_aggregator.diffusion_mlp is not None:
            print("检测到diffusion_mlp，使用问题嵌入作为条件输入")
            assert self.get_model().get_video_tower().opensora_diffusion is not None , "The MLP defined but the diffusion is not used"
            # 将问题嵌入转换为 MLP 期望的数据类型
            q_text_embeds = q_text_embeds.to(dtype=self.get_model().temporal_aggregator.diffusion_mlp.layers[0].weight.dtype)
            # 通过 MLP 处理问题嵌入
            diffusion_text_embedding = self.get_model().temporal_aggregator.diffusion_mlp(q_text_embeds)
            print(f"diffusion文本嵌入形状: {diffusion_text_embedding.shape}")
        else:
            print("未使用diffusion条件输入")
            diffusion_text_embedding = None

        # encode video feature (编码视频特征)
        print("直接使用传入的视频特征（跳过视频塔编码）")
        video_features = video_feats
        new_frame_num = feat_frame_nums        
        # 调整维度顺序以匹配时间聚合器的输入要求
        video_features = video_features.permute([0,2,3,4,1]) # [B, C, T, H, W] -> [B, T, H, W, C]
        
        # **这里是使用 temporal_aggregator 处理特征的核心位置**
        if hasattr(self.get_model().temporal_aggregator, 'use_slow_as_query') and self.get_model().temporal_aggregator.use_slow_as_query:
            print("时间聚合器使用慢速特征作为查询")
            video_features = self.get_model().temporal_aggregator(video_features, new_frame_num, 
                                                                q_text_embeds=q_text_embeds, 
                                                                q_text_nums=q_text_nums,
                                                                chunk_num=chunk_num,
                                                                slow_feats=slow_feats)  # 🔥 核心处理位置1
        else:
            print("时间聚合器正常模式（不使用慢速特征作为查询）")
            video_features = self.get_model().temporal_aggregator(video_features, new_frame_num, 
                                                                q_text_embeds=q_text_embeds, 
                                                                q_text_nums=q_text_nums,
                                                                chunk_num=chunk_num)  # 🔥 核心处理位置2
        
        # handle output feature number for the case with the cross-attn 
        print("处理时间聚合器输出的token数量")
        if hasattr(self.get_model().temporal_aggregator, 'use_query_tokens') and self.get_model().temporal_aggregator.use_query_tokens: # v2 version
            print(f"V2版本：使用固定查询token数({self.get_model().temporal_aggregator.num_query_tokens})")
            new_frame_num = torch.tensor([self.get_model().temporal_aggregator.num_query_tokens] * video_features.shape[0]).to(video_features.device)
        if type(self.get_model().temporal_aggregator).__name__ == 'SSMTemporalAggregatorV4': # for v4 version, although this may not neccessary
            print(f"V4版本：token数为查询数×chunk数({self.get_model().temporal_aggregator.number_of_query}×{chunk_num})")
            new_frame_num = torch.tensor([self.get_model().temporal_aggregator.number_of_query*chunk_num] * video_features.shape[0]).to(video_features.device)
        
        print(f"视频编码完成，输出形状: {video_features.shape}, 新帧数: {new_frame_num}")
        return video_features, new_frame_num

    def encode_images(self, images, return_feat_before_mlp=False):
        """编码图像。这是"慢速路径"的核心。"""
        print("开始编码图像特征（慢速路径）")
        print(f"输入图像形状: {images.shape}, return_feat_before_mlp: {return_feat_before_mlp}")
        
        # 使用视觉塔提取特征
        image_features_before_mlp = self.get_model().get_vision_tower()(images)
        print(f"视觉塔输出特征形状: {image_features_before_mlp.shape}")
        
        # 确保数据类型正确，以便进行后续计算
        image_features_before_mlp = image_features_before_mlp.to(dtype=self.dtype) # update the data type for eval
        # 将提取的特征通过多模态投影器（mm_projector），对齐到语言模型空间
        image_features = self.get_model().mm_projector(image_features_before_mlp)
        print(f"投影器输出特征形状: {image_features.shape}")
        
        if return_feat_before_mlp:
            print("返回投影前后的特征")
            return image_features, image_features_before_mlp
        else:
            print("只返回投影后的特征")
            return image_features

    def get_2dPool(self, image_feature, stride=2):
        """对图像特征进行 2D 池化操作，以减少空间维度（token 数量）。"""
        print(f"开始2D池化，输入形状: {image_feature.shape}, stride: {stride}")
        
        height = width = self.get_vision_tower().num_patches_per_side # 获取 patch 的边长
        num_frames, num_tokens, num_dim = image_feature.shape
        print(f"池化参数: height={height}, width={width}, frames={num_frames}")
        
        # 将扁平化的 patch 特征恢复为 2D 网格形状
        image_feature = image_feature.view(num_frames, height, width, -1)
        # 调整维度顺序以适应 PyTorch 的池化函数 (N, C, H, W)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        pool_mode = self.config.mm_spatial_pool_mode
        print(f"使用{pool_mode}池化模式")
        
        if pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif pool_mode == "bilinear":
            # 使用双线性插值进行下采样
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            print(f"双线性插值目标形状: {scaled_shape}")
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {pool_mode}")
        
        # 将维度顺序恢复，并再次扁平化
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        
        print(f"2D池化完成，输出形状: {image_feature.shape}")
        return image_feature

    def add_token_per_grid(self, image_feature):
        """为每个网格（grid）添加一个特殊 token（如换行符）。"""
        resize_h = int(math.sqrt(image_feature.shape[1])) # 计算网格的高度
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]
        
        # 恢复网格形状
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        # 在每个网格的末尾拼接上 image_newline token
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # ... (注释掉的代码块) ...
        # 扁平化并调整维度顺序，返回最终结果
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        """为每一帧（frame）添加一个特殊 token。"""
        # 调整维度顺序
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        # 在每帧特征的末尾拼接上 image_newline token
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # 恢复原始维度顺序
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_image_features(self, images, image_sizes, modalities, return_feat_before_mlp=False):
        '''
            This function is for encode the image feature.
            该函数用于编码图像特征。
            images: list[tensor]: shape of the tensor is torch.Size([32, 3, 384, 384]), len of list = batchsize
                                  一个张量列表，每个张量代表一批帧，形状为 [帧数, 通道, 高, 宽]
            image_sizes: list[int]: represent the H*W*C of the origin video frames
                                    原始视频帧的尺寸信息
            modalities: list[string]: string should be "video"
                                     模态类型列表，指示每个输入是 "video" 还是 "image"
        '''
        print("开始准备图像特征")
        print(f"输入参数: images类型={type(images)}, image_sizes={image_sizes}, modalities={modalities}")
        
        if images is None: # We do not have image as input (如果没有图像输入，直接返回 None)
            print("没有图像输入，返回None")
            return None
        
        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                print("输入为图像列表，确保每个元素至少是4维")
                # 确保每个图像/视频帧张量至少是4维的
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            # 找出批次中哪些是视频
            video_idx_in_batch = []
            for idx, modality in enumerate(modalities):
                if modality == "video":
                    video_idx_in_batch.append(idx)
            print(f"批次中视频索引: {video_idx_in_batch}")

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            # 将所有图像/帧拼接成一个大的批次，以便一次性送入编码器
            concat_images = torch.cat([image for image in images_list], dim=0) # e.g., torch.Size([64, 3, 384, 384])
            # 记录每个原始图像/视频的帧数，以便后续拆分
            split_sizes = [image.shape[0] for image in images_list] # e.g., [32, 32]
            print(f"拼接后图像形状: {concat_images.shape}, 拆分大小: {split_sizes}")
            
            if return_feat_before_mlp:
                print("编码图像特征（返回投影前后特征）")
                # 编码拼接后的大批次图像，并获取投影前后的特征
                encoded_image_features, encoded_image_features_before_mlp = self.encode_images(concat_images, return_feat_before_mlp=return_feat_before_mlp)
                
                # 将编码后的特征按原始大小拆分回列表
                encoded_image_features = torch.split(encoded_image_features, split_sizes) 
                encoded_image_features_before_mlp = torch.split(encoded_image_features_before_mlp, split_sizes) 
                image_features = []
                image_features_before_mlp = []
                # 遍历每个样本的特征
                for idx, (image_feat, image_feat_before_mlp) in enumerate(zip(encoded_image_features, encoded_image_features_before_mlp)):
                    if idx in video_idx_in_batch: # 如果是视频
                        print(f"样本{idx}是视频，应用2D池化")
                        # 对视频帧特征进行 2D 池化，减少 token 数量
                        image_features.append(self.get_2dPool(image_feat))
                        image_features_before_mlp.append(self.get_2dPool(image_feat_before_mlp))
                    else: # 如果是单张图像
                        print(f"样本{idx}是图像，直接使用特征")
                        image_features.append(image_feat)
                        image_features_before_mlp.append(image_feat_before_mlp)
                
                print("图像特征准备完成（包含投影前特征）")
                return image_features, video_idx_in_batch, image_features_before_mlp
            else:
                print("编码图像特征（仅返回投影后特征）")
                # 只获取投影后的特征
                encoded_image_features = self.encode_images(concat_images, return_feat_before_mlp=return_feat_before_mlp)
                
                # 将编码后的特征按原始大小拆分
                encoded_image_features = torch.split(encoded_image_features, split_sizes) 
                image_features = []
                for idx, image_feat in enumerate(encoded_image_features):
                    if idx in video_idx_in_batch: # 如果是视频
                        print(f"样本{idx}是视频，应用2D池化")
                        # 应用 2D 池化
                        image_features.append(self.get_2dPool(image_feat))
                    else: # 如果是图像
                        print(f"样本{idx}是图像，直接使用特征")
                        image_features.append(image_feat)
                
                print("图像特征准备完成（仅投影后特征）")
                return image_features, video_idx_in_batch
        else:
            print("不支持的输入格式")
            raise NotImplementedError

    def post_processing_of_image_feature(self, image_features, video_idx_in_batch):
        '''
            This function is for some post-processing of the image feature, 
            like flatten and adding special tokens
            此函数用于图像特征的后处理，例如扁平化和添加特殊 token。
        '''
        print("开始图像特征后处理")
        
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat") # 'spatial_unpad'
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square") # 'anyres_max_9'
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token") # image:'one_token', video: 'no_token'
        
        print(f"后处理配置: patch_merge_type={mm_patch_merge_type}, "
                    f"aspect_ratio={image_aspect_ratio}, newline_position={mm_newline_position}")

        if mm_patch_merge_type == "flat":
            print("使用flat模式：直接扁平化帧和patch维度")
            # 直接将帧和 patch 维度扁平化
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"): # INTO HERE (进入此分支)
            print("使用spatial模式进行后处理")
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_idx in video_idx_in_batch:  # video operations (视频操作)
                    print(f"处理视频{image_idx}，newline_position={mm_newline_position}")
                    
                    if mm_newline_position == "grid":
                        print("按网格添加token")
                        # Grid-wise: 按网格添加 token
                        image_feature = self.add_token_per_grid(image_feature)
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        print("按帧添加token")
                        # Frame-wise: 按帧添加 token
                        image_feature = self.add_token_per_frame(image_feature) # e.g., [32, 169, 896] -> [32, 170, 896]
                        new_image_features.append(image_feature.flatten(0, 1)) # 扁平化
                    elif mm_newline_position == "one_token":
                        print("添加单个token")
                        # one-token: 只在所有特征的最后添加一个 token
                        image_feature = image_feature.flatten(0, 1)
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        new_image_features.append(image_feature)      
                    elif mm_newline_position == "no_token":
                        print("不添加特殊token，直接扁平化")
                        # 不添加任何特殊 token，直接扁平化
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        print(f"不支持的newline_position: {mm_newline_position}")
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                else:  # single image operations (单图操作)
                    print(f"处理单图{image_idx}（未实现）")
                    raise NotImplementedError
            image_features = new_image_features
        else:
            print(f"不支持的patch_merge_type: {mm_patch_merge_type}")
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        
        print("图像特征后处理完成")
        return image_features

    def prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                video_feats,
                video_feat_fps=None,
                feat_frame_nums=None,
                question_ids=None,
                question_lens=None,
                # for the image frames (用于图像帧的参数)
                images=None,
                image_sizes=None,
                modalities=None,
                video_metas=None,
            ):
        """
        核心函数：准备多模态输入的 input_embeds 和 labels。
        它将文本、图像、视频特征整合在一起，生成最终送入语言模型的输入。
        
        attention_mask的作用：
        1. 标识哪些位置是有效token，哪些是padding
        2. 在多模态融合后重新计算mask，确保视觉token被正确关注
        3. 防止模型关注到padding位置，提高训练和推理效率
        """
        print("开始准备多模态输入")
        print(f"输入参数: input_ids.shape={input_ids.shape if input_ids is not None else None}, "
                    f"attention_mask.shape={attention_mask.shape if attention_mask is not None else None}, "
                    f"video_feats.shape={video_feats.shape if video_feats is not None else None}, "
                    f"images类型={type(images)}, modalities={modalities}")
        
        video_tower = self.get_video_tower()
        vision_tower = self.get_vision_tower()
        
        # 如果没有视觉模块，或者没有视觉输入，或者在推理（只输入一个 token），则直接返回文本输入
        if (video_tower is None and vision_tower is None) or (video_feats is None and images is None) or input_ids.shape[1] == 1:
            print("跳过多模态处理：无视觉模块或视觉输入，或处于推理模式")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # embed the question id using our LLM
        # 使用 LLM 的 token_embedding 层将问题 ID 转换为嵌入
        if question_ids is not None:
            print("将问题ID转换为嵌入")
            question_embeds = self.get_model().embed_tokens(question_ids).detach()
        else:
            print("无问题ID输入")
            question_embeds = None
        
        # figure out the chunk size (确定视频块的数量)
        if images is not None:
            chunk_num = images[0].shape[0]
            print(f"检测到chunk数量: {chunk_num}")
        else:
            chunk_num = None
            print("无chunk信息")
            
        # special control for the slow feature (对慢速路径特征的特殊控制)
        if hasattr(self.get_model(), 'temporal_aggregator') and getattr(self.get_model().temporal_aggregator, 'use_slow_feat_before_mlp', False): 
            print("启用use_slow_feat_before_mlp模式")
            assert images is not None and (-1 not in image_sizes) # 如果使用 MLP 前的慢速特征，当前只支持原始视频输入
            self.use_slow_feat_before_mlp = True
        else:
            print("未启用use_slow_feat_before_mlp模式")
            self.use_slow_feat_before_mlp = False

        # get the image feature (The slow feature) (获取图像特征，即慢速特征)
        if images is not None and (-1 not in image_sizes):
            print("处理原始图像输入")
            if self.use_slow_feat_before_mlp:
                # 准备图像特征，并返回投影前后的特征
                image_features, video_idx_in_batch, image_features_before_mlp = self.prepare_image_features(images, image_sizes, modalities, return_feat_before_mlp=self.use_slow_feat_before_mlp)
            else:
                # 只准备投影后的特征
                image_features, video_idx_in_batch = self.prepare_image_features(images, image_sizes, modalities)
        else: # the image feature is loaded (图像特征是预先加载的)
            print("使用预加载的图像特征")
            assert sum(image_sizes) == -len(image_sizes) # 确认所有 image_sizes 都是 -1
            image_features = images
            video_idx_in_batch = []
            for idx, modality in enumerate(modalities):
                if modality == "video":
                    video_idx_in_batch.append(idx)
        
        # get the video feature (The fast feature) (获取视频特征，即快速特征)
        if video_feats is None:
            print("无视频特征输入")
            video_features, new_frame_num = None, None
        else:
            print("编码视频特征")
            video_features, new_frame_num = self.encode_videos(video_feats, 
                                                            q_text_embeds=question_embeds,
                                                            video_feat_fps=video_feat_fps, 
                                                            feat_frame_nums=feat_frame_nums,
                                                            q_text_nums=question_lens,
                                                            chunk_num=chunk_num,
                                                            slow_feats=image_features if not self.use_slow_feat_before_mlp else image_features_before_mlp)

        # add up the video and image features. (合并视频和图像特征)
        feat_combine_method = getattr(self.config, 'feat_combine_method', 'concat')
        print(f"使用特征合并方法: {feat_combine_method}")
        
        if video_features is not None and feat_combine_method == 'add':
            print("使用add方法合并快速和慢速特征")
            # 如果是 'add' 方法，将快速特征和慢速特征相加
            assert image_features[0].shape[1]*image_features[0].shape[0] == video_features.shape[1]
            
            updated_image_feat = []
            for curr_video_feat, curr_image_feat in zip(video_features, image_features):
                curr_video_feat = rearrange(curr_video_feat, "(k s) d -> k s d", k=chunk_num)  # 调整快速特征形状以匹配慢速特征
                updated_image_feat.append(curr_video_feat + curr_image_feat)
            image_features = updated_image_feat
        
        # proprocessing of the image feature (图像特征的后处理)
        if images is not None:
            print("对图像特征进行后处理")
            image_features = self.post_processing_of_image_feature(image_features, video_idx_in_batch)
        
        # Combine image and video feature, and update the new_frame_num (合并图像和视频特征，并更新帧数)
        train_addition_start_end_tokens = getattr(self.config, 'train_addition_start_end_tokens', False)
        print(f"train_addition_start_end_tokens: {train_addition_start_end_tokens}")
        
        if train_addition_start_end_tokens:
            assert self.get_model().temporal_aggregator.start_end_tokens is not None
            start_end_token_set = self.get_model().temporal_aggregator.start_end_tokens
        
        if video_features is not None:
            if feat_combine_method == 'concat':
                print("使用concat方法合并快速和慢速特征")
                # 'concat' 方法：将慢速特征和快速特征在 token 维度上拼接
                image_features = torch.cat([ele.unsqueeze(dim=0) for ele in image_features], dim=0)
                video_features = torch.cat([image_features, video_features], dim=-2)
                new_frame_num += image_features.shape[1]
            elif feat_combine_method == 'interleave':
                print("使用interleave方法交错合并特征")
                # 'interleave' 方法：交错合并快速和慢速特征
                interleaved_feat = []
                interleaved_frame_num = []
                frame_number_per_video = images[0].shape[0] # 每段视频的帧数
                print(f"每段视频帧数: {frame_number_per_video}")
                
                for curr_img_feat, curr_vid_feat, curr_vid_feat_len in zip(image_features, video_features, new_frame_num):
                    #### handle the image features (处理图像特征)
                    total_image_tokens = curr_img_feat.shape[0]
                    assert total_image_tokens % frame_number_per_video == 0
                    tokens_per_frame = total_image_tokens // frame_number_per_video
                    image_feat_split_sizes = [tokens_per_frame for i in range(frame_number_per_video)]
                    splited_image_feat = torch.split(curr_img_feat, image_feat_split_sizes) # 按帧拆分慢速特征
                    
                    #### handle the video features (处理视频特征)
                    updated_video_feat = curr_vid_feat[:curr_vid_feat_len]
                    video_feat_split_sizes = split_list_lengths(curr_vid_feat_len, frame_number_per_video)
                    splited_video_feat = torch.split(updated_video_feat, video_feat_split_sizes) # 按帧拆分快速特征
                    
                    #### combine the feature (合并特征)
                    combined_feat = []
                    for i_f, v_f in zip(splited_image_feat, splited_video_feat):
                        if train_addition_start_end_tokens:
                            print("添加起止token")
                            # 如果使用额外的起止 token，按特定顺序拼接
                            combined_feat.append(start_end_token_set[0].unsqueeze(dim=0))
                            combined_feat.append(v_f)
                            combined_feat.append(start_end_token_set[1].unsqueeze(dim=0))
                            combined_feat.append(start_end_token_set[2].unsqueeze(dim=0))
                            combined_feat.append(i_f)
                            combined_feat.append(start_end_token_set[3].unsqueeze(dim=0))
                        else:
                            # 否则直接拼接快速和慢速特征
                            combined_feat.append(v_f)
                            combined_feat.append(i_f)
                    combined_feat = torch.cat(combined_feat)
                    interleaved_feat.append(combined_feat)
                    interleaved_frame_num.append(combined_feat.shape[0])
                new_frame_num = interleaved_frame_num
                video_features = interleaved_feat
            elif feat_combine_method == 'add':
                print("add方法已在前面处理")
                # 'add' 方法已在前面处理，这里更新帧数
                video_features = image_features
                new_frame_num = [ele.shape[0] for ele in image_features]
            else:
                print(f"不支持的特征合并方法: {feat_combine_method}")
                raise NotImplementedError
        else: # IF we are not using the fast path (如果不使用快速路径)
            print("不使用快速路径，直接使用慢速特征")
            video_features = image_features
            for ele in video_features:
                ele.requires_grad = True # 确保特征有梯度
            new_frame_num = [ele.shape[0] for ele in image_features]
        
        print("开始构建最终的输入序列")
        
        # Let's just add dummy tensors if they do not exist...
        # 为 None 的张量创建虚拟张量，以简化后续处理
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        print("处理attention_mask:")
        if attention_mask is None:
            print("attention_mask为None，根据input_ids创建全1的mask")
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            print(f"创建的attention_mask形状: {attention_mask.shape}")
        else:
            print(f"使用输入的attention_mask，形状: {attention_mask.shape}")
            attention_mask = attention_mask.bool()
        
        print("attention_mask的作用解释:")
        print("1. 原始作用：标识input_ids中哪些位置是真实token(1)，哪些是padding(0)")
        print("2. 在文本处理中：防止模型关注padding位置")
        print("3. 在多模态处理中：需要重新计算以包含视觉token的位置")
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # 使用 attention_mask 去除填充部分，得到可变长度的序列
        print("使用attention_mask去除padding，获得每个样本的有效序列:")
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # 打印去除padding后的序列长度
        for i, (ids, mask) in enumerate(zip(input_ids, [cur_attention_mask for cur_attention_mask in attention_mask])):
            print(f"样本{i}: 原始长度={mask.shape[0]}, 有效长度={ids.shape[0]}, padding比例={1-ids.shape[0]/mask.shape[0]:.2%}")

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # 遍历批次中的每一个样本
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # determine how many image we have (确定当前样本有多少个图像/视频占位符)
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # 如果没有图像，直接处理文本（虽然这里的逻辑似乎不完整，cur_image_features[0:0] 是空张量）
                cur_image_features = video_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # 找到所有图像占位符的位置
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            # 根据图像占位符的位置，将文本和标签切分成段
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            
            # 将切分后的文本 ID 转换为嵌入
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # merge the text embedding with the image embedding
            # 将文本嵌入和视觉特征嵌入交错合并
            for i in range(num_images + 1):
                # 添加一段文本嵌入
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # 添加一段视觉特征嵌入
                    cur_image_features = video_features[cur_image_idx]
                    cur_feature_len = new_frame_num[cur_image_idx]
                    cur_image_features = cur_image_features[:cur_feature_len] # (T, C) or (T, H*W, C)
                    if len(cur_image_features.shape) == 3:
                        cur_image_features = cur_image_features.view(-1, cur_image_features.shape[-1])
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    # 视觉特征对应的标签应被忽略，不计入损失
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            
            # 将所有片段拼接成一个完整的序列
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # 因为加入了视觉特征，序列可能变得很长，需要截断到模型能接受的最大长度
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            print(f"截断序列到最大长度: {tokenizer_model_max_length}")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them by padding
        # 将批次内所有可变长度的序列填充到相同的最大长度
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        print(f"重新构建attention_mask用于多模态输入:")
        print(f"批次大小: {batch_size}, 最大序列长度: {max_len}")
        
        # 初始化填充后的张量
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        print("重新构建的attention_mask作用:")
        print("1. 新的mask标识融合后序列中的有效位置（包括文本和视觉token）")
        print("2. 0表示padding位置，1表示有效的token位置")
        print("3. 确保Transformer只关注有意义的文本和视觉内容")
        
        # 遍历每个样本并进行填充
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')
            
            print(f"样本{i}: 有效长度={cur_len}, 需要padding长度={max_len-cur_len}, padding方向={padding_side}")
            
            if padding_side == "left":
                # 左填充
                print(f"  执行左填充，有效token位置: [{max_len-cur_len}:{max_len}]")
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True  # 左填充时，右侧为有效位置
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # 右填充 (默认)
                print(f"  执行右填充，有效token位置: [0:{cur_len}]")
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True  # 右填充时，左侧为有效位置
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # 将列表堆叠成一个批次张量
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        print(f"最终attention_mask统计:")
        for i in range(batch_size):
            valid_tokens = attention_mask[i].sum().item()
            print(f"  样本{i}: 有效token数={valid_tokens}/{max_len}, 比例={valid_tokens/max_len:.2%}")

        # match the output with the input (匹配输出格式和原始输入)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            print("原始attention_mask为None，返回None")
            attention_mask = None
        else:
            print("转换attention_mask数据类型以匹配原始输入")
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        print("attention_mask处理完成，总结其关键作用:")
        print("1. 输入阶段：识别原始文本序列中的有效token")
        print("2. 处理阶段：用于去除padding，获得真实的可变长度序列")
        print("3. 融合阶段：重新计算以包含插入的视觉token位置")
        print("4. 输出阶段：指导Transformer模型正确执行attention计算")
        print("5. 最终效果：确保模型只关注有意义的文本和视觉内容，忽略padding")
        
        # 最终返回的是处理好的 embedding 和相关张量，input_ids 返回 None，因为已经被 embedding 替代
        print("多模态输入准备完成")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """初始化与视觉相关的 tokenizer 设置，主要是添加新的特殊 token。"""
        print("开始初始化视觉tokenizer")
        
        # use additional image token between the image and the text
        if model_args.mm_use_im_patch_token:
            print("添加图像patch token")
            # 添加表示图像 patch 的 token
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            print("添加图像/视频开始和结束token")
            # 添加表示图像/视频开始和结束的 token
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer)) # 调整 embedding 矩阵大小以容纳新 token

            if num_new_tokens > 0:
                print(f"添加了{num_new_tokens}个新token，初始化其embedding")
                # 将新 token 的 embedding 初始化为所有旧 token embedding 的平均值
                input_embeddings_avg = self.get_input_embeddings().weight.data[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = self.get_input_embeddings().weight.data[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                self.get_input_embeddings().weight.data[-num_new_tokens:] = input_embeddings_avg
                self.get_input_embeddings().weight.data[-num_new_tokens:] = output_embeddings_avg # Bug? Should be output_embeddings_avg?

            # 根据配置决定是否需要训练这些新的 token embedding
            if model_args.tune_temporal_aggregator or model_args.tune_addition_token_embeddings:
                print("解冻新token的embedding参数")
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True # 解冻输入 embedding 层
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False # 保持输出 embedding 层（语言模型头）冻结

            if model_args.pretrain_temporal_aggregator:
                print("从预训练权重加载新token的embedding")
                # 如果有预训练权重，则加载新 token 的 embedding
                mm_projector_weights = torch.load(model_args.pretrain_temporal_aggregator, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if self.get_input_embeddings().weight.data.shape == embed_tokens_weight.shape:
                    self.get_input_embeddings().weight.data[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    self.get_input_embeddings().weight.data[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {self.get_input_embeddings().weight.data.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            # freeze the input and output tokens
            # 如果只使用 patch token 而不使用 start/end token，且需要微调聚合器，则冻结 token embedding
            if model_args.tune_temporal_aggregator:
                print("冻结输入和输出token embedding")
                self.get_input_embeddings().requires_grad_(False)
                self.get_output_embeddings().requires_grad_(False)       

        ### handle the special case which the len(tokenizer) != self.get_input_embeddings().weight.data.shape[0]
        # 处理特殊情况：tokenizer 的词汇表大小与 embedding 矩阵大小不一致，进行调整
        if len(tokenizer) != self.get_input_embeddings().weight.data.shape[0]:
            print(f"Tokenizer词汇表大小({len(tokenizer)})与embedding矩阵大小({self.get_input_embeddings().weight.data.shape[0]})不一致，进行调整")
            self.resize_token_embeddings(len(tokenizer))
        
        print("视觉tokenizer初始化完成")
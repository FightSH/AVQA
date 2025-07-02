# This script contains all the implementation for the training dataset.
# 该脚本包含了所有关于训练数据集的实现。

import os
import copy
from dataclasses import dataclass, field
import json
import time
from typing import  Any, Dict, Optional, Sequence, List, Tuple, Union
import ipdb
from PIL import Image
import numpy as np
import math
import random
from copy import deepcopy

from einops import rearrange, repeat
import torch
from torch.utils.data import Dataset
import transformers

from ..constants import IGNORE_INDEX, VID_EXTENSIONS, IMG_EXTENSIONS
from ..utils.train_utils import DataArguments, rank0_print
from .image_dataset import preprocess_multimodal, preprocess, preprocess_qwen, preprocess_video_multimodal
from .video_loading_utils import read_video, temporal_random_crop, get_transforms_video, fps_base_temporal_sampling, frame_base_temporal_sampling, process_video_with_decord, process_video_with_tv
from .dataset_utils import load_data


def load_images_from_folder(folder_path):
    """从文件夹加载一系列图像帧，并（如果需要）将它们保存为 .npy 文件以加速后续加载。"""
    # ipdb.set_trace()
    # try loading the npy files (尝试加载 .npy 文件)
    npy_file_path = os.path.join(folder_path, 'stacked_images.npy')
    if os.path.exists(npy_file_path):
        images_array = np.load(npy_file_path)
        return images_array
    
    # if the npy files not exist, list all the files (如果 .npy 文件不存在，列出文件夹中所有文件)
    all_files_under_folder = os.listdir(folder_path)
    
    # filter and sort the file along the temporal (按时间顺序（文件名中的数字）过滤和排序 jpg 文件)
    jpg_files = sorted(
        [f for f in all_files_under_folder if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )    
    
    # load the image (加载图像)
    images = []
    for filename in jpg_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure all images are RGB (确保所有图像都是 RGB 格式)
        img_array = np.array(img)
        images.append(img_array)
    
    # handle the special case (处理特殊情况，确保帧数固定为 32)
    if len(images) > 32:
        # 如果帧数超过 32，进行截断
        print(folder_path, len(images))
        images = images[:32]
    elif len(images) < 32:
        # 如果帧数不足 32，用最后一帧进行填充
        gap = 32 - len(images)
        print(folder_path, len(images))
        images = images + [images[-1]] * gap
    
    # stack the image (将图像列表堆叠成一个 numpy 数组)
    if images:
        # Convert list of images to a 4D NumPy array (将图像列表转换为 4D NumPy 数组)
        images_array = np.stack(images, axis=0)  # Shape: (number_of_images, H, W, C)
        # save the npy file (保存 .npy 文件)
        # ipdb.set_trace()
        save_file_name = os.path.join(folder_path, "stacked_images.npy")
        print('stacked numpy array is saved to:', save_file_name)
        np.save(save_file_name, images_array)
        return images_array
    else:
        print("No JPG images found in the specified folder.")
        return None


def load_dense_frame_feature(video_feat_file_path, exclude_languagebind_cls_token):
    """加载密集的帧级别特征（例如，从 LanguageBind 提取的特征）。"""
    # ipdb.set_trace() # check the loading
    # load the feature (加载特征)
    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu')) # e.g., torch.Size([280, 5, 1024]) T, C, D / torch.Size([32, 196, 1024]) / torch.Size([280, 4, 1024])
    # exclude the cls tokens (排除 CLS token)
    if exclude_languagebind_cls_token:
        video_feat = video_feat[:, 1:,]
    # reshape (重塑形状)
    S = video_feat.shape[1] # S 是 patch 的数量
    assert int(math.sqrt(S)) ** 2 == S # assert is a square (断言 patch 数量是平方数，可以构成 HxW 网格)
    W = H = int(math.sqrt(S))
    # 将形状从 (T, S, C) 调整为 (C, T, H, W)
    video_feat = rearrange(video_feat, 't (h w) c -> c t h w', h = H) # video_feat should be in the shape of (C, T, H, W)
    video_feat_fps = 2 # 假设特征的帧率为 2
    feat_frame_num = video_feat.shape[1] # 获取帧数
    
    return video_feat, video_feat_fps, feat_frame_num


def load_exo_feature(fast_feat_type, video_feat_file_path):
    """加载 'exo' (egocentric, 第一人称视角) 类型的特征。"""
    # video_folder = '/'.join(video_feat_file_path.split('/')[:-1])
    video_folder = video_feat_file_path # we assume it given a folder (假设给定的是一个文件夹路径)
    # list all the .pt files, the content should be in the shape of torch.Size([32, 196, 1152])
    # 列出所有 .pt 文件，其内容应为形状 [32, 196, 1152]
    all_files = os.listdir(video_folder)
    
    if fast_feat_type == 'exo_random': # randomly pick the video and update hte list (随机选择视频片段)
        # ipdb.set_trace()
        random_num = random.randint(1, len(all_files)) # Step 2: Pick a random number k between 1 and 4 (随机选择 1 到 k 个片段)
        all_files = random.sample(all_files, random_num)

    # concat the feature in the first dimension. (在第一个维度上拼接特征)
    video_feat = []
    for curr_pt_file in all_files:
        curr_feature = torch.load(os.path.join(video_folder, curr_pt_file)).unsqueeze(dim=0)
        video_feat.append(curr_feature)
        
    # concat the feature and merge the temporal dimension torch.Size([4, 32, 196, 1152])
    # 拼接特征并合并时间维度
    video_feat = torch.cat(video_feat, dim=0)
    V, T, S, D = video_feat.shape # V: 视频片段数, T: 每片段帧数, S: 每帧 patch 数, D: 特征维度
    video_feat = video_feat.permute([1,0,2,3]).reshape(-1, S, D) # 合并 V 和 T 维度，得到 [128, 196, 1152]
    # change the shape to (C, T, H, W) (将形状调整为 C, T, H, W)
    video_feat = video_feat.view(-1, 14, 14, D).permute([3, 0, 1, 2]) # T, S, C -> T, H, W, C -> C, T, H, W
    
    video_feat_fps = 1 # 假设帧率为 1
    feat_frame_num = video_feat.shape[1]
    
    return video_feat, video_feat_fps, feat_frame_num


def load_audio_feature(video_feat_file_path):
    """加载音频特征。"""
    # load the feature (加载特征)
    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu')) #torch.Size([19474, 768]) T, C
    if len(video_feat.shape) == 3: # special case for the model languagebind feature: torch.Size([32, 593, 1024]) (处理 LanguageBind 特征的特殊情况)
        # ipdb.set_trace() # check the loading, check the view
        feat_dim = video_feat.shape[-1]
        video_feat = video_feat.view(-1, feat_dim) # (T*S, C)
    # special handle for the grad (特殊处理梯度)
    if video_feat.requires_grad:
        video_feat.requires_grad = False
    
    video_feat = video_feat.permute([1,0]) # (C, T)
    # unsqueeze dim (增加维度以匹配 C, T, H, W 格式)
    video_feat = video_feat.unsqueeze(dim=-1).unsqueeze(dim=-1) # (C, T, 1, 1) (C, T, H, W)
    # reshape
    video_feat_fps = 100 / 16 # 计算音频特征的等效帧率
    feat_frame_num = video_feat.shape[1]
    
    return video_feat, video_feat_fps, feat_frame_num


class LazySupervisedVideoDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""
    """用于视频监督微调的数据集类，采用"懒加载"模式。"""

    def __init__(self, 
                 anno_path: str,            # path to the instruction annotation json file (指令标注 json 文件的路径)
                 fast_path_mapping_path: str,    # path to the mapping between the video id and the video feature path (视频ID到视频特征路径的映射文件)
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedVideoDataset, self).__init__()
        print("=== LazySupervisedVideoDataset 初始化开始 ===")
        # handle the hyper (处理超参数)
        print("\n第一步：设置基础参数")
        self.tokenizer = tokenizer
        self.data_root = data_args.data_root
        self.data_args = data_args
        self.prepare_qid = data_args.prepare_qid # 是否准备问题ID
        self.data_sample_ratio = data_args.data_sample_ratio # 数据集采样比例
        

        # handle the video features (处理视频特征)
        print("\n第二步：配置视频特征路径")
        self.use_fast = data_args.use_fast # 是否使用快速路径（原始视频）
        self.use_fast_feat = data_args.use_fast_feat # 是否使用快速路径（预提取特征）
        
        print(f"  - use_fast (原始视频): {self.use_fast}")
        print(f"  - use_fast_feat (预提取特征): {self.use_fast_feat}")
        
        # handle for the image feature (处理图像特征)
        print("\n第三步：配置图像特征路径")
        self.use_slow = data_args.use_slow # 是否使用慢速路径（原始帧）
        self.use_slow_feat = data_args.use_slow_feat # 是否使用慢速路径（预提取特征）
        slow_path_mapping_path = data_args.slow_path_mapping_path
        self.slow_path_data_root = data_args.slow_path_data_root
        
        print(f"  - use_slow (原始帧): {self.use_slow}")
        print(f"  - use_slow_feat (预提取特征): {self.use_slow_feat}")
        print(f"  - slow_path_mapping_path: {slow_path_mapping_path}")
        print(f"  - slow_path_data_root: {self.slow_path_data_root}")
        
        # handle the fast feature type and the tokens (处理快速特征的类型和token)
        print("\n第四步：配置特征类型参数")
        self.fast_feat_type = data_args.fast_feat_type
        self.exclude_languagebind_cls_token = data_args.exclude_languagebind_cls_token
        
        print(f"  - fast_feat_type: {self.fast_feat_type}")
        print(f"  - exclude_languagebind_cls_token: {self.exclude_languagebind_cls_token}")
        
        # the method for loading the video (视频加载的后端库)
        print("\n第五步：配置视频加载后端")
        self.video_loading_backbone = data_args.video_loading_backbone
        
        print(f"  - video_loading_backbone: {self.video_loading_backbone}")
        
        # handle for the second side channel (处理第二路辅助信道，如音频或第一人称视角)
        print("\n第六步：配置第二路辅助信道")
        self.use_second_sides = data_args.use_second_sides
        self.second_sides_type = data_args.second_sides_type
        self.second_sides_data_root = data_args.second_sides_data_root
        
        print(f"  - use_second_sides: {self.use_second_sides}")
        print(f"  - second_sides_type: {self.second_sides_type}")
        print(f"  - second_sides_data_root: {self.second_sides_data_root}")
        
        # assertion about the video path (关于视频路径的断言，确保配置是合法的)
        print("\n第七步：验证配置合法性")
        fast_assertion = (self.data_args.use_fast == True and self.data_args.use_fast_feat == False) or \
                        (self.data_args.use_fast == False and self.data_args.use_fast_feat == True) or \
                        (self.data_args.use_fast == False and self.data_args.use_fast_feat == False)
        
        slow_assertion = (self.data_args.use_slow == True and self.data_args.use_slow_feat == False) or \
                        (self.data_args.use_slow == False and self.data_args.use_slow_feat == True) or \
                        (self.data_args.use_slow == False and self.data_args.use_slow_feat == False)
        
        print(f"  - 快速路径配置合法性: {fast_assertion}")
        print(f"  - 慢速路径配置合法性: {slow_assertion}")
        
        assert fast_assertion, "快速路径配置冲突：use_fast和use_fast_feat不能同时为True"
        assert slow_assertion, "慢速路径配置冲突：use_slow和use_slow_feat不能同时为True"
        
        # process the self.data_sample_ratio (处理数据集采样比例)
        print("\n第八步：处理采样比例配置")
        if self.data_sample_ratio is not None:
            print(f"  - 原始采样比例配置: {self.data_sample_ratio}")
            self.data_sample_ratio = self.data_sample_ratio.split(',')
            self.data_sample_ratio = [float(ele) for ele in self.data_sample_ratio]
            print(f"  - 解析后采样比例: {self.data_sample_ratio}")
            print(f"  - slow_path_mapping_path长度: {len(slow_path_mapping_path) if isinstance(slow_path_mapping_path, list) else 1}")
            assert len(self.data_sample_ratio) == len(slow_path_mapping_path), "采样比例数量与数据路径数量不匹配"
        else:
            print(f"  - 不使用采样比例，使用全部数据")
        
        # 预过滤掉一些有问题的视频ID
        print("\n第九步：设置预过滤视频ID列表")
        prefilter_video_ids = ['009151_009200/1057949419', '077251_077300/14911927', # for step1 training
                               '00013654', # for step2 sharegptvideo traning
                               ] 
        print(f"  - 预过滤视频ID: {prefilter_video_ids}")
        
        # handle the special case for multiple dataset (处理多个数据集的情况)
        print("\n第十步：加载和处理数据集")
        if isinstance(anno_path, list):
            print(f"  检测到多个数据集配置")
            print(f"  - 数据集数量: {len(anno_path)}")
            print(f"  - 标注文件: {anno_path}")
            print(f"  - 快速路径映射: {fast_path_mapping_path}")
            print(f"  - 数据根目录: {self.data_root}")
            
            assert isinstance(fast_path_mapping_path, list), "多数据集模式下fast_path_mapping_path必须是列表"
            assert isinstance(self.data_root, list), "多数据集模式下data_root必须是列表"
            
            if self.use_slow or self.use_slow_feat: # use raw video frames (如果使用慢速路径)
                print(f"  使用慢速路径，验证慢速路径配置...")
                assert isinstance(slow_path_mapping_path, list), "使用慢速路径时slow_path_mapping_path必须是列表"
                assert isinstance(self.slow_path_data_root, list), "使用慢速路径时slow_path_data_root必须是列表"
                
                expected_len = len(anno_path)
                actual_lens = [len(fast_path_mapping_path), len(self.data_root), 
                             len(slow_path_mapping_path), len(self.slow_path_data_root)]
                print(f"  配置长度验证: 期望{expected_len}, 实际{actual_lens}")
                assert len(anno_path) == len(fast_path_mapping_path) == len(self.data_root) == len(slow_path_mapping_path) == len(self.slow_path_data_root), "多数据集配置长度不一致"

                # load annotation (加载标注)
                print(f"  开始加载多个数据集的标注...")
                all_filtered_anno = []
                # 遍历所有数据集的配置
                for i, (curr_anno_path, curr_fast_path_mapping_path, curr_data_root, curr_slow_path_mapping_path, curr_slow_path_data_root) in \
                    enumerate(zip(anno_path, fast_path_mapping_path, self.data_root, slow_path_mapping_path, self.slow_path_data_root)):
                    print(f"    处理数据集 {i+1}/{len(anno_path)}:")
                    print(f"      - 标注文件: {curr_anno_path}")
                    print(f"      - 快速路径映射: {curr_fast_path_mapping_path}")
                    print(f"      - 数据根目录: {curr_data_root}")
                    print(f"      - 慢速路径映射: {curr_slow_path_mapping_path}")
                    print(f"      - 慢速数据根目录: {curr_slow_path_data_root}")
                    
                    # 加载并过滤当前数据集的标注
                    curr_anno = load_annotation_and_filter(curr_anno_path, curr_fast_path_mapping_path, curr_data_root, 
                                                        prefilter_video_ids=prefilter_video_ids,
                                                        slow_path_mapping_path=curr_slow_path_mapping_path,
                                                        slow_path_data_root=curr_slow_path_data_root,
                                                        second_side_channels_root=self.second_sides_data_root if self.use_second_sides else None)
                    print(f"      - 加载完成，原始数量: {len(curr_anno)}")
                    
                    if self.data_sample_ratio is not None:
                        # 如果设置了采样比例，则按比例采样
                        curr_ratio = self.data_sample_ratio[i]
                        curr_selected_len = int(curr_ratio * len(curr_anno))
                        curr_anno = curr_anno[:curr_selected_len]
                        print(f"      - 应用采样比例 {curr_ratio}: {len(curr_anno)} 样本")
                    all_filtered_anno += curr_anno # 将处理后的标注添加到总列表中
                    print(f"      - 累计样本数: {len(all_filtered_anno)}")
            else: # 如果不使用慢速路径
                print(f"  不使用慢速路径")
                expected_len = len(anno_path)
                actual_lens = [len(fast_path_mapping_path), len(self.data_root)]
                print(f"  配置长度验证: 期望{expected_len}, 实际{actual_lens}")
                assert len(anno_path) == len(fast_path_mapping_path) == len(self.data_root), "配置长度不一致"
                
                # load annotation (加载标注)
                print(f"  开始加载多个数据集的标注...")
                all_filtered_anno = []
                for i, (curr_anno_path, curr_fast_path_mapping_path, curr_data_root) in enumerate(zip(anno_path, fast_path_mapping_path, self.data_root)):
                    print(f"    处理数据集 {i+1}/{len(anno_path)}:")
                    print(f"      - 标注文件: {curr_anno_path}")
                    print(f"      - 快速路径映射: {curr_fast_path_mapping_path}")
                    print(f"      - 数据根目录: {curr_data_root}")
                    
                    curr_anno = load_annotation_and_filter(curr_anno_path, curr_fast_path_mapping_path, curr_data_root, 
                                                        prefilter_video_ids=prefilter_video_ids)
                    print(f"      - 加载完成，原始数量: {len(curr_anno)}")
                    
                    if self.data_sample_ratio is not None:
                        curr_ratio = self.data_sample_ratio[i]
                        curr_selected_len = int(curr_ratio * len(curr_anno))
                        curr_anno = curr_anno[:curr_selected_len]
                        print(f"      - 应用采样比例 {curr_ratio}: {len(curr_anno)} 样本")
                    all_filtered_anno += curr_anno
                    print(f"      - 累计样本数: {len(all_filtered_anno)}")
        else: # 如果只处理单个数据集
            print(f"  检测到单个数据集配置")
            print(f"  - 标注文件: {anno_path}")
            print(f"  - 快速路径映射: {fast_path_mapping_path}")
            print(f"  - 数据根目录: {self.data_root}")
            print(f"  - 慢速路径映射: {slow_path_mapping_path}")
            print(f"  - 慢速数据根目录: {self.slow_path_data_root}")
            
            all_filtered_anno = load_annotation_and_filter(anno_path, fast_path_mapping_path, self.data_root, 
                                                           prefilter_video_ids=prefilter_video_ids,
                                                           slow_path_mapping_path=slow_path_mapping_path,
                                                           slow_path_data_root=self.slow_path_data_root,
                                                           second_side_channels_root=self.second_sides_data_root if self.use_second_sides else None)
            print(f"  单个数据集加载完成，样本数量: {len(all_filtered_anno)}")

        # ipdb.set_trace() # check the loaded list_data_dict
        self.list_data_dict = all_filtered_anno
        print(f"\n第十一步：数据集最终统计")
        print(f"  - 最终样本总数: {len(self.list_data_dict)}")
        if len(self.list_data_dict) > 0:
            print(f"  - 第一个样本键: {list(self.list_data_dict[0].keys())}")
            print(f"  - 示例视频ID: {self.list_data_dict[0].get('video', 'N/A')}")

        # set the transform (设置数据增强/变换)
        print(f"\n第十二步：设置数据变换")
        print(f"  - 变换名称: {self.data_args.transform_name}")
        print(f"  - 图像尺寸: {self.data_args.image_size}")
        
        self.transforms = {
            # "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(self.data_args.transform_name, self.data_args.image_size),
        }
        print(f"  - 视频变换设置完成")
        
        # handle the additional information of the feature loading and downsampling
        # 处理特征加载和下采样的额外信息
        print(f"\n第十三步：配置特征加载参数")
        if self.use_fast_feat:
            self.original_feat_fps = self.data_args.original_feat_fps # 原始特征帧率
            self.training_feat_fps = self.data_args.training_feat_fps # 训练时使用的特征帧率
            print(f"  - 原始特征帧率: {self.original_feat_fps}")
            print(f"  - 训练特征帧率: {self.training_feat_fps}")
        else:
            print(f"  - 不使用快速特征，跳过特征帧率配置")
        
        print("=== LazySupervisedVideoDataset 初始化完成 ===\n")

    def __len__(self):
        return len(self.list_data_dict)

    def get_type(self, path):
        """根据文件扩展名判断是视频还是图像。"""
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """获取单个数据样本，包含重试机制以处理 I/O 错误。"""
        print(f"\n=== 开始获取样本 {i} ===")
        print(f"数据集总长度: {len(self.list_data_dict)}")
        
        # TODO: define number of retries somewhere else
        num_base_retries = 3 # 基础重试次数
        num_final_retries = 300 # 最终重试次数（已注释掉）

        # try the current sample first (首先尝试获取当前样本)
        print(f"第一阶段：尝试获取当前样本 {i}，最多重试 {num_base_retries} 次")
        for attempt_idx in range(num_base_retries):
            try:
                print(f"  尝试 #{attempt_idx + 1}: 调用 _get_item({i})")
                sample = self._get_item(i)
                print(f"  成功获取样本 {i}")
                print(f"=== 样本 {i} 获取完成 ===\n")
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue (如果是云盘问题，暂停1秒)
                print(f"  [Try #{attempt_idx + 1}] 获取样本 {i} 失败. 异常:", e)
                print(f"  暂停1秒后重试...")
                time.sleep(1)

        # try other samples, in case it is file corruption issue (如果可能是文件损坏问题，尝试获取其他样本)
        print(f"第二阶段：当前样本失败，尝试获取其他样本，最多重试 {num_base_retries} 次")
        for attempt_idx in range(num_base_retries):
            try:
                # 获取下一个样本
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                print(f"  尝试 #{attempt_idx + 1}: 调用 _get_item({next_index}) (替代样本)")
                sample = self._get_item(next_index)
                print(f"  成功获取替代样本 {next_index}")
                print(f"=== 样本 {next_index} (替代 {i}) 获取完成 ===\n")
                return sample
            except Exception as e:
                # no need to sleep
                print(f"  [Try other #{attempt_idx + 1}] 获取样本 {next_index} 失败. 异常:", e)

        # 最后再尝试一次当前样本，如果再失败就抛出异常
        print(f"第三阶段：最后一次尝试获取原始样本 {i}")
        try:
            print(f"  最终尝试: 调用 _get_item({i})")
            sample = self._get_item(i)
            print(f"  最终成功获取样本 {i}")
            print(f"=== 样本 {i} 获取完成 ===\n")
            return sample
        except Exception as e:
            print(f"  最终尝试失败，抛出异常: {e}")
            print(f"=== 样本 {i} 获取彻底失败 ===\n")
            raise e

    def _get_item(self, i):
        """实际获取和处理单个数据样本的逻辑。"""
        print(f"    → _get_item({i}) 开始执行")
        
        sources = self.list_data_dict[i]
        print(f"    → 获取源数据: {list(sources.keys()) if isinstance(sources, dict) else 'list'}")
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        print(f"    → 检查数据类型...")
        if 'video' in sources[0]:
            print(f"    → 检测到视频数据，video_id: {sources[0].get('video', 'N/A')}")
            
            # 特征加载部分
            print(f"    → 开始加载特征...")
            print(f"    → use_fast: {self.use_fast}, use_fast_feat: {self.use_fast_feat}")
            
            if self.use_fast: # This is for a special test which directly load the video frames as the video feature
                print(f"    → 使用原始视频作为快速特征（未实现）")
                raise NotImplementedError
            elif self.use_fast_feat: # This is for loading the fast features
                print(f"    → 加载预提取的快速特征")
                video_feat_file_path = self.list_data_dict[i]['feat_path']
                print(f"    → 快速特征路径: {video_feat_file_path}")
                print(f"    → 快速特征类型: {self.fast_feat_type}")
                
                if self.fast_feat_type in ['languagebind', 'languagebind_14x14', 'internvideo2', 'siglip']:
                    print(f"    → 加载密集帧特征 ({self.fast_feat_type})")
                    video_feat, video_feat_fps, feat_frame_num = load_dense_frame_feature(video_feat_file_path, self.exclude_languagebind_cls_token)
                    print(f"    → 快速特征形状: {video_feat.shape}, FPS: {video_feat_fps}, 帧数: {feat_frame_num}")
                elif self.fast_feat_type == 'audio':
                    print(f"    → 加载音频特征")
                    video_feat, video_feat_fps, feat_frame_num = load_audio_feature(video_feat_file_path)
                    print(f"    → 音频特征形状: {video_feat.shape}, FPS: {video_feat_fps}, 帧数: {feat_frame_num}")
                elif self.fast_feat_type == '3d_feature':
                    print(f"    → 加载3D特征")
                    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu'))
                    print(f"    → 原始3D特征形状: {video_feat.shape}")
                    B, _ , D = video_feat.shape
                    V, H, W = 32, 24, 24
                    video_feat = video_feat.view(B, V, H, W, D).squeeze(dim=0)
                    video_feat = video_feat.permute([3,0,1,2])
                    video_feat_fps = 1
                    feat_frame_num = video_feat.shape[1]
                    print(f"    → 处理后3D特征形状: {video_feat.shape}, FPS: {video_feat_fps}, 帧数: {feat_frame_num}")
                elif self.fast_feat_type in ['exo', 'exo_random']:
                    print(f"    → 加载Exo特征 ({self.fast_feat_type})")
                    video_feat, video_feat_fps, feat_frame_num = load_exo_feature(self.fast_feat_type, video_feat_file_path)
                    print(f"    → Exo特征形状: {video_feat.shape}, FPS: {video_feat_fps}, 帧数: {feat_frame_num}")
                else:
                    print(f"    → 不支持的快速特征类型: {self.fast_feat_type}")
                    raise NotImplementedError
            else:
                print(f"    → 不使用快速路径特征")
                    
            # 视频帧加载部分
            print(f"    → 开始处理视频帧...")
            print(f"    → use_slow: {self.use_slow}, use_slow_feat: {self.use_slow_feat}")
            
            if self.use_slow: # 如果使用慢速路径（原始帧）
                print(f"    → 使用慢速路径加载原始帧")
                video_file_path = self.list_data_dict[i]['video_path']
                print(f"    → 视频文件路径: {video_file_path}")
                
                if os.path.isdir(video_file_path): # is bunch of image
                    print(f"    → 检测到图像序列文件夹")
                    video = load_images_from_folder(video_file_path)
                    print(f"    → 加载的图像序列形状: {video.shape if video is not None else None}")
                    processor = self.data_args.image_processor
                    image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                    print(f"    → 预处理后图像形状: {image.shape}")
                    image = [(image, 100, "video")]
                else: # is video
                    print(f"    → 检测到视频文件")
                    print(f"    → 视频加载后端: {self.video_loading_backbone}")
                    try:
                        # load the video
                        if self.video_loading_backbone == 'decord':
                            print(f"    → 使用decord加载视频")
                            video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file_path, self.data_args)
                        else:
                            print(f"    → 使用torchvision加载视频")
                            video, video_time, frame_time, num_frames_to_sample = process_video_with_tv(video_file_path, self.data_args)
                        
                        print(f"    → 视频加载成功，帧数: {len(video)}, 视频时长: {video_time}, 采样帧数: {num_frames_to_sample}")
                        
                        # preprocess the video frames
                        processor = self.data_args.image_processor
                        image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                        print(f"    → 视频预处理完成，形状: {image.shape}")
                        
                        # prepare the video frames, the original video frame size, type of modality
                        image = [(image, video[0].size, "video")]
                        print(f"    → 原始视频帧尺寸: {video[0].size}")
                    except Exception as e:
                        print(f"    → 视频加载失败: {e}")
                        print(f"    → 尝试加载下一个样本...")
                        return self._get_item(i + 1)
            elif self.use_slow_feat: # load the video feature
                print(f"    → 加载预提取的慢速特征")
                video_file_path = self.list_data_dict[i]['video_path']
                print(f"    → 慢速特征路径: {video_file_path}")
                image = torch.load(video_file_path)
                print(f"    → 慢速特征形状: {image.shape}")
                image = [(image, -1, "video")] # use -1 to mark it as a feature instead of a raw frames
            else:
                print(f"    → 不使用慢速路径")

            # 文本处理
            print(f"    → 开始处理文本对话...")
            sources = preprocess_video_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            print(f"    → 视频多模态文本预处理完成")
        else:
            print(f"    → 检测到纯文本数据")
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # tokenize the text
        print(f"    → 开始文本tokenization...")
        print(f"    → prepare_qid: {self.prepare_qid}")
        old_data_dict = preprocess(
            sources,
            self.tokenizer,
            has_vision=('video' in self.list_data_dict[i]),
            for_video=True,
            prepare_qid=self.prepare_qid)
        
        print(f"    → tokenization完成，input_ids长度: {len(old_data_dict['input_ids'][0])}")
        
        if isinstance(i, int):
            data_dict = dict(input_ids=old_data_dict["input_ids"][0],
                             labels=old_data_dict["labels"][0])
            if 'question_ids' in old_data_dict:
                data_dict['question_ids'] = old_data_dict["question_ids"][0]
                data_dict['question_len'] = old_data_dict["question_len"]
                print(f"    → 问题ID长度: {old_data_dict['question_len']}")

        # video exist in the data
        if 'video' in self.list_data_dict[i]:
            print(f"    → 整理视频相关数据...")
            
            if self.use_fast or self.use_fast_feat:
                data_dict['video_feat'] = video_feat
                data_dict['video_feat_fps'] = video_feat_fps  
                data_dict['feat_frame_num'] = feat_frame_num
                print(f"    → 添加快速特征到数据字典")
            
            # Put the video frame information in
            if self.use_slow or self.use_slow_feat:
                data_dict["image"] = image
                print(f"    → 添加慢速特征/帧到数据字典")
            
            # put the meta information in
            data_dict['video_meta'] = self.list_data_dict[i]
            print(f"    → 添加视频元信息到数据字典")
            
        elif self.data_args.is_multimodal:
            print(f"    → 多模态模型但无视频数据（未实现）")
            raise NotImplementedError

        ### add for second sides
        if self.use_second_sides:
            print(f"    → 处理第二路辅助信道...")
            print(f"    → 第二路类型: {self.second_sides_type}")
            assert 'second_side_file_path' in self.list_data_dict[i]
            second_side_file_path = self.list_data_dict[i]['second_side_file_path']
            print(f"    → 第二路文件路径: {second_side_file_path}")
            
            if self.second_sides_type == 'audio':
                print(f"    → 加载第二路音频特征")
                second_feat, second_feat_fps, second_feat_frame_num = load_audio_feature(second_side_file_path)
            elif self.second_sides_type in ['exo', 'exo_random']: 
                print(f"    → 加载第二路Exo特征")
                second_feat, second_feat_fps, second_feat_frame_num = load_exo_feature(self.second_sides_type, second_side_file_path)
            else:
                print(f"    → 不支持的第二路类型: {self.second_sides_type}")
                raise NotImplementedError 
            
            data_dict['second_feat'] = second_feat
            data_dict['second_feat_fps'] = second_feat_fps  
            data_dict['second_feat_frame_num'] = second_feat_frame_num
            print(f"    → 第二路特征形状: {second_feat.shape}, FPS: {second_feat_fps}, 帧数: {second_feat_frame_num}")
        else:
            print(f"    → 不使用第二路辅助信道")
        
        print(f"    → _get_item({i}) 执行完成")
        print(f"    → 最终数据字典键: {list(data_dict.keys())}")
        
        return data_dict


@dataclass
class DataCollatorForSupervisedVideoDataset(object):
    """Collate examples for supervised fine-tuning."""
    """用于监督微调的数据整理器。"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """将一批样本（instances）整理成一个批次（batch）张量。"""
        # 提取 input_ids 和 labels
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # 对 input_ids 和 labels 进行填充，使批次内所有样本长度一致
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # 截断到模型的最大长度
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), # 创建 attention mask
        )

        # 整理视频特征的元信息
        if 'video_feat_fps' in instances[0]:
            batch['video_feat_fps'] = torch.tensor([ele['video_feat_fps'] for ele in instances])
        
        if 'feat_frame_num' in instances[0]:
            batch['feat_frame_nums'] = torch.tensor([ele['feat_frame_num'] for ele in instances])

        # 整理视频特征
        if 'video_feat' in instances[0]:
            video_feats = [instance['video_feat'] for instance in instances]
            if all(x is not None and x.shape == video_feats[0].shape for x in video_feats):
                # 如果所有特征形状相同，直接堆叠
                batch['video_feats'] = torch.stack(video_feats)
            else:
                # We do the padding here to accerate the training (如果形状不同，进行填充)
                all_lens = [ele['feat_frame_num'] for ele in instances]
                max_len = max(all_lens) 
                C, T, H, W = instances[0]['video_feat'].shape
                padded_tensor = torch.zeros([len(instances), C, max_len, H, W]) # (B, C, T, H, W)
                for i, (v, v_len) in enumerate(zip(video_feats, all_lens)):
                    padded_tensor[i][:, :v_len] = v
                batch['video_feats'] = padded_tensor
                
        # handle for the second sides (处理第二路辅助信道的数据)
        if 'second_feat_fps' in instances[0]:
            batch['second_feat_fps'] = torch.tensor([ele['second_feat_fps'] for ele in instances])
        
        if 'second_feat_frame_num' in instances[0]:
            batch['second_feat_frame_nums'] = torch.tensor([ele['second_feat_frame_num'] for ele in instances])

        if 'second_feat' in instances[0]:
            second_feats = [instance['second_feat'] for instance in instances]
            if all(x is not None and x.shape == second_feats[0].shape for x in second_feats):
                batch['second_feats'] = torch.stack(second_feats)
            else:
                # We do the padding here to accerate the training (进行填充)
                all_lens = [ele['second_feat_frame_num'] for ele in instances]
                max_len = max(all_lens) 
                C, T, H, W = instances[0]['second_feat'].shape
                padded_tensor = torch.zeros([len(instances), C, max_len, H, W]) # (B, C, T, H, W)
                for i, (v, v_len) in enumerate(zip(second_feats, all_lens)):
                    padded_tensor[i][:, :v_len] = v
                batch['second_feats'] = padded_tensor

        # 整理问题 ID 和长度
        if 'question_ids' in instances[0]:
            question_ids = [instance['question_ids'].squeeze(dim=0) for instance in instances]
            
            question_ids = torch.nn.utils.rnn.pad_sequence(
                question_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            question_ids = question_ids[:, :self.tokenizer.model_max_length]
            batch['question_ids'] = question_ids
            batch['question_lens'] = torch.tensor([instance['question_len'] for instance in instances])

        # 整理视频元信息
        batch['video_metas'] = [ele['video_meta'] for ele in instances]
        
        ## handle the image frames (处理图像帧)
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            
            # 展平列表，提取尺寸和模态信息
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # 直接将图像/特征列表放入批次中，后续处理
            batch["images"] = images
        return batch



def make_video_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                        data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    """创建用于监督微调的数据集和数据整理器。"""
    train_dataset = LazySupervisedVideoDataset(tokenizer=tokenizer,
                                               anno_path=data_args.annotation_path,
                                               fast_path_mapping_path=data_args.fast_path_mapping_path,
                                               data_args=data_args)
    data_collator = DataCollatorForSupervisedVideoDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load_annotation_and_filter(anno_path, fast_path_mapping_path, data_root, 
                               prefilter_video_ids=None,
                               slow_path_mapping_path=None,
                               slow_path_data_root=None,
                               second_side_channels_root=None):
    """加载标注数据，并根据特征/视频文件是否存在进行过滤。"""
    list_data_dict = load_data(anno_path) # It will be a list and each ele in the list is a dictionary (加载原始标注)
    # load the mapping (加载映射文件)
    feat_path_mapping = json.load(open(fast_path_mapping_path))
    
    if slow_path_mapping_path is not None:
        video_path_mapping = json.load(open(slow_path_mapping_path))
    else:
        video_path_mapping = None
    
    # filter the id that video not exist (过滤掉视频不存在的标注)
    filtered_anno_list = []
    remaining_video = []
    video_not_exist = []
    
    for ele in list_data_dict:    
        # decode the video id (解析视频ID)
        vid = ele['video']
        # handle the spcial case in the LLaVA-video 178K (处理 LLaVA-video 178K 中的特殊情况)
        if '/' in vid:
            vid = vid.split('/')[-1]        
        # handle the case that for the activitynet video of sharegpt-video (处理 sharegpt-video 中 activitynet 视频的情况)
        if 'Scene' in vid: 
            vid = vid[:13]
        # handle the special case for the valley and video-chatgpt dataset (处理 valley 和 video-chatgpt 数据集的特殊情况)
        if '.' in vid:
            vid = vid.split('.')[0]
        # filter some issue video for video-chatgpt dataset (过滤掉 video-chatgpt 数据集中的问题视频)
        if prefilter_video_ids is not None and vid in prefilter_video_ids:
            print(vid, 'in prefilter list, filtered.')
            video_not_exist.append(vid)
            continue        
            
        if vid in feat_path_mapping:
            # 检查快速路径文件是否存在
            feat_file_path = os.path.join(data_root, feat_path_mapping[vid])
            if not os.path.exists(feat_file_path): # filter the not exist video
                video_not_exist.append(vid)
                continue
            ele['feat_path'] = feat_file_path # 将完整路径存入标注信息
            
            # 检查慢速路径文件是否存在
            if video_path_mapping is not None:
                video_file_path = os.path.join(slow_path_data_root, video_path_mapping[vid])
                if not os.path.exists(video_file_path):
                    video_not_exist.append(vid)
                    continue
                ele['video_path'] = video_file_path
                
            # 检查第二路辅助信道文件是否存在
            if second_side_channels_root is not None:
                second_side_file_path = os.path.join(second_side_channels_root, feat_path_mapping[vid])
                if 'ego' in second_side_file_path: # 特殊处理 ego 路径
                    second_side_file_path = '/'.join(second_side_file_path.split('/')[:-1])
                    
                if not os.path.exists(second_side_file_path):
                    video_not_exist.append(vid)
                    continue
                ele['second_side_file_path'] = second_side_file_path                
                
            filtered_anno_list.append(ele)
            remaining_video.append(vid)
        else:
            video_not_exist.append(vid)
    
    # 打印过滤统计信息
    print('dataset:', anno_path, 
          'total annotation:', len(list_data_dict), 
          'remaining anno:', len(filtered_anno_list), 
          'existing video:', len(set(remaining_video)),
          'video_not_exist:', len(set(video_not_exist)))
    
    return filtered_anno_list
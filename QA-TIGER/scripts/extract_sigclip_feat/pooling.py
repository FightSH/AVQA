import os
import torch
import numpy as np
import torch.nn as nn
import math



def get_2dPool( image_feature, stride=2):
    """对图像特征进行 2D 池化操作，以减少空间维度（token 数量）。"""

    # height = width = self.get_vision_tower().num_patches_per_side  # 获取 patch 的边长
    num_frames = 60
    height = width = 27
    num_tokens = 729
    num_dim=1152

    # 将扁平化的 patch 特征恢复为 2D 网格形状
    image_feature = image_feature.view(num_frames, height, width, -1)
    # 调整维度顺序以适应 PyTorch 的池化函数 (N, C, H, W)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()


    # print(f"使用{pool_mode}池化模式")
        # 使用双线性插值进行下采样
    height, weight = image_feature.shape[2:]
    scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]

    image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')


    # 将维度顺序恢复，并再次扁平化
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)

    return image_feature

folder = '/mnt/sda/shenhao/datasets/siglip/MUSIC-AVQA/patch_features'
for filename in os.listdir(folder):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder, filename)
        try:
            arr = np.load(file_path)
            print(f"[{filename}] 原始npy shape: {arr.shape}")
            if arr.shape == (60, 196, 1152):
                print(f"[{filename}] shape 为 (60, 196, 1152)，跳过处理。")
                continue
            tensor = torch.from_numpy(arr)
            pooled = get_2dPool(tensor)
            print(f"[{filename}] 池化后 shape: {pooled.shape}")
            pooled_np = pooled.cpu().numpy()
            np.save(file_path, pooled_np)
        except Exception as e:
            print(f"处理文件 [{filename}] 时出错：{e}")
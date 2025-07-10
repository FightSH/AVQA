import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import logging
import numpy as np
import glob
from typing import Union, List
import torch.nn.functional as F
import torch.nn as nn
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Siglip2FeatureExtractor:
    def __init__(self, model_path: str = None, cache_dir: str = '/mnt/sda/shenhao/models/siglip2'):
        """
        初始化图像特征提取器
        :param model_path: 模型路径，可以是本地路径或HuggingFace模型名
        :param cache_dir: 自定义缓存目录
        """
        # 首先检查依赖


        # 使用环境变量或默认值

        self.model_path  = 'google/siglip2-so400m-patch14-384'
        # 设置缓存目录
        self.cache_dir = cache_dir

        # 显示缓存路径信息

        try:
            # 使用自定义缓存目录加载模型
            logger.info("正在加载处理器...")

            # model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384",
            #                                   cache_dir='/mnt/sda/shenhao/models/siglip2')
            # processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384",
            #                                           cache_dir='/mnt/sda/shenhao/models/siglip2', use_fast=True)

            self.processor = AutoProcessor.from_pretrained(self.model_path, cache_dir=self.cache_dir,use_fast=True)
            logger.info("处理器加载成功")

            logger.info("正在加载模型...")
            self.model = AutoModel.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            logger.info("模型加载成功")

            # 显示实际加载的模型路径
            # if hasattr(self.model.config, '_name_or_path'):
            #     logger.info(f"实际加载的模型路径: {self.model.config._name_or_path}")
            self.device = torch.device("cuda:3")

            self.model = self.model.to(self.device)
            logger.info(f"模型已加载到设备: {self.device}")
            logger.info(f"GPU名称: {torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else 'N/A'}")
            logger.info(
                f"GPU显存: {torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3:.1f}GB" if torch.cuda.is_available() else "N/A")

        except ImportError as e:
            if "sentencepiece" in str(e).lower():
                logger.error("SentencePiece 库缺失！请安装：pip install sentencepiece")
                raise ImportError(
                    "请先安装 SentencePiece 库：\n"
                    "pip install sentencepiece\n"
                    "然后重启 Python 环境。"
                ) from e
            else:
                raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error("可能的解决方案：")
            logger.error("1. 检查网络连接")
            logger.error("2. 确保有足够的磁盘空间")
            logger.error("3. 尝试清除缓存后重新下载")
            raise

    def get_2dPool(self, image_feature, stride=2):
        """对图像特征进行 2D 池化操作，以减少空间维度（token 数量）。"""
        # print(f"开始2D池化，输入形状: {image_feature.shape}, stride: {stride}")

        # height = width = self.get_vision_tower().num_patches_per_side  # 获取 patch 的边长
        num_frames = 1
        height = width = 27
        num_tokens = 729
        num_dim = 1152

        # 将扁平化的 patch 特征恢复为 2D 网格形状
        image_feature = image_feature.view(num_frames, height, width, -1)
        # 调整维度顺序以适应 PyTorch 的池化函数 (N, C, H, W)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()

        # print(f"使用{pool_mode}池化模式")
        # 使用双线性插值进行下采样
        height, weight = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
        # print(f"双线性插值目标形状: {scaled_shape}")
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        # 将维度顺序恢复，并再次扁平化
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)

        # print(f"2D池化完成，输出形状: {image_feature.shape}")
        return image_feature

    def get_image_features(self, image: Union[Image.Image, str, np.ndarray],
                           extract_patch_features: bool = False) -> np.ndarray:
        """
        提取图片特征
        :param image: 输入图片 (PIL.Image, 文件路径, 或numpy数组)
        :param extract_patch_features: 是否提取patch级别的特征，False为全局特征
        :return: 图片特征向量 (numpy.ndarray)
        """
        try:

            # 处理不同类型的输入
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    # 网络图片
                    response = requests.get(image, stream=True, timeout=10)
                    response.raise_for_status()
                    image = Image.open(response.raw)
                else:
                    # 本地文件
                    image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            # 读取本地图片
            # image = Image.open(image)
            # 打印原始图片信息
            # logger.info(f"原始图片尺寸: {image.size}")
            # logger.info(f"原始图片模式: {image.mode}")

            # 确保图片是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                # logger.info("图片已转换为RGB格式")

            # 处理图像输入
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # 打印处理后的输入张量信息
            # if 'pixel_values' in inputs:
            # logger.info(f"处理后的输入张量维度: {inputs['pixel_values'].shape}")

            # 提取特征
            with torch.no_grad():
                if extract_patch_features:
                    # 获取patch级别的特征
                    vision_outputs = self.model.vision_model(**inputs)
                    # vision_outputs.last_hidden_state 包含所有patch的特征
                    # 形状通常是 [batch_size, num_patches, hidden_size]
                    patch_features = vision_outputs.last_hidden_state



                    # L2 归一化每个patch特征
                    patch_features = F.normalize(patch_features, p=2, dim=-1)
                    # patch_features = patch_features / patch_features.norm(p=2, dim=-1, keepdim=True)

                    patch_features = self.get_2dPool(patch_features, stride=2)

                    feature_array = patch_features.cpu().numpy()


                    return feature_array
                else:
                    # 获取全局图像特征（原有方式）
                    image_features = self.model.get_image_features(**inputs)

                    # logger.info(f"提取的全局特征维度: {image_features.shape}")
                    # logger.info(f"特征数据类型: {image_features.dtype}")
                    # logger.info(f"特征范围: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")

                    # L2 归一化
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                    # logger.info(f"L2归一化后特征范围: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")
                    # logger.info(f"特征向量的L2范数: {image_features.norm(p=2, dim=-1).item():.4f}")

                    feature_array = image_features.cpu().numpy()
                    # logger.info(f"最终全局特征numpy数组维度: {feature_array.shape}")

                    return feature_array

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path, dst_clip_patch_path):
    """
    批量提取图像的全局特征和patch级别特征
    :param dir_fps_path: 输入视频帧目录路径
    :param dst_clip_path: 全局特征保存路径
    :param dst_clip_patch_path: patch特征保存路径
    """
    # 初始化特征提取器
    logger.info("初始化图像特征提取器...")
    extractor = Siglip2FeatureExtractor(model_path='google/siglip2-so400m-patch14-384',cache_dir='/mnt/sda/shenhao/models/siglip2')

    # 创建输出目录
    os.makedirs(dst_clip_path, exist_ok=True)
    os.makedirs(dst_clip_patch_path, exist_ok=True)

    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    logger.info(f"共找到 {total_nums} 个视频目录")

    for video in video_list:
        video_idx = video_idx + 1
        print(f"\n--> {video_idx}/{total_nums} 处理视频: {video}")

        # 全局特征保存路径
        save_global_file = os.path.join(dst_clip_path, video + '.npy')
        # patch特征保存路径
        save_patch_file = os.path.join(dst_clip_patch_path, video + '.npy')

        # 检查是否已经处理过
        if os.path.exists(save_global_file) and os.path.exists(save_patch_file):
            print(f"{video} 的全局特征和patch特征已存在，跳过处理!")
            continue

        # 获取视频帧列表
        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))

        if not video_img_list:
            logger.warning(f"视频 {video} 中没有找到图像文件，跳过处理")
            continue

        # logger.info(f"找到 {len(video_img_list)} 张图像")

        # 初始化特征存储
        global_features_list = []
        patch_features_list = []

        for idx, img_path in enumerate(video_img_list):
            try:
                # logger.info(f"处理图像 {idx+1}/{len(video_img_list)}: {os.path.basename(img_path)}")

                # 提取全局特征
                global_feat = extractor.get_image_features(img_path, extract_patch_features=False)
                global_features_list.append(global_feat.squeeze(0))  # 移除batch维度

                # 提取patch特征
                patch_feat = extractor.get_image_features(img_path, extract_patch_features=True)
                patch_features_list.append(patch_feat.squeeze(0))  # 移除batch维度

                # logger.info(f"全局特征形状: {global_feat.shape}, patch特征形状: {patch_feat.shape}")

            except Exception as e:
                logger.error(f"处理图像 {img_path} 时出错: {e}")
                continue

        if global_features_list and patch_features_list:
            # 转换为numpy数组并保存
            global_features_array = np.array(global_features_list)
            patch_features_array = np.array(patch_features_list)

            # 保存全局特征
            np.save(save_global_file, global_features_array)
            # logger.info(f"全局特征已保存: {save_global_file}, 形状: {global_features_array.shape}")

            # 保存patch特征
            np.save(save_patch_file, patch_features_array)
            # logger.info(f"patch特征已保存: {save_patch_file}, 形状: {patch_features_array.shape}")

            print(f"处理完成: {video_idx}/{total_nums} - 视频ID: {video}")
            print(f"全局特征形状: {global_features_array.shape}")
            print(f"patch特征形状: {patch_features_array.shape}")
        else:
            logger.warning(f"视频 {video} 没有成功提取到特征")


def batch_extract_features(dir_fps_path, output_base_dir):
    """
    便捷的批量特征提取函数
    :param dir_fps_path: 输入视频帧目录路径
    :param output_base_dir: 输出基础目录
    """
    dst_clip_path = os.path.join(output_base_dir, 'global_features')
    dst_clip_patch_path = os.path.join(output_base_dir, 'patch_features')

    logger.info(f"输入目录: {dir_fps_path}")
    logger.info(f"全局特征输出目录: {dst_clip_path}")
    logger.info(f"patch特征输出目录: {dst_clip_patch_path}")

    ImageClIP_feat_extract(dir_fps_path, dst_clip_path, dst_clip_patch_path)

# model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384",
#                                   cache_dir='/mnt/sda/shenhao/models/siglip2')
# processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384",
#                                           cache_dir='/mnt/sda/shenhao/models/siglip2', use_fast=True)
#
# model.to('cuda:3')
# device = next(model.parameters()).device
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs = processor(images=image, return_tensors="pt")
# inputs = {k: v.to(device) for k, v in inputs.items()}
#
# with torch.no_grad():
#     image_features = model.get_image_features(**inputs)
#     print(image_features.shape)  # (1, 1152)

if __name__ == "__main__":
    try:
        # 测试单张图片特征提取
        # logger.info("开始初始化特征提取器...")
        # extractor = ImageFeatureExtractor()
        # extractor = Siglip2FeatureExtractor(model_path='google/siglip2-so400m-patch14-384',
        #                         cache_dir='/mnt/sda/shenhao/models/siglip2')
        # # # 使用本地图片进行测试
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"

        # # 测试全局特征提取
        # print("\n=== 测试全局特征提取 ===")
        # global_features = extractor.get_image_features(url, extract_patch_features=False)
        # print(f"全局特征形状: {global_features.shape}")
        # print(f"全局特征类型: {type(global_features)}")
        #
        # # # 测试patch级别特征提取
        # print("\n=== 测试patch级别特征提取 ===")
        # patch_features = extractor.get_image_features(url, extract_patch_features=True)
        # print(f"patch特征形状: {patch_features.shape}")
        # print(f"patch特征类型: {type(patch_features)}")
        # print(f"patch数量: {patch_features.shape[1]}")
        # print(f"每个patch特征维度: {patch_features.shape[2]}")

        # 测试批量特征提取（如果有本地数据）
        # 示例路径，请根据实际情况修改
        dir_fps_path = "/mnt/sda/shenhao/datasets/MUSIC-AVQA/frames/"
        output_base_dir = "/mnt/sda/shenhao/datasets/siglip2/MUSIC-AVQA/"

        batch_extract_features(dir_fps_path, output_base_dir)

    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("请按照提示安装缺失的依赖库。")
    except Exception as e:
        print(f"运行出错: {e}")

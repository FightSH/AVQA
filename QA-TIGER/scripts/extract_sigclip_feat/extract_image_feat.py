import os
# 设置HuggingFace镜像端点（必须在导入transformers之前设置）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
from typing import Union, List
import logging
from huggingface_hub import snapshot_download

# 检查必需的依赖
def check_dependencies():
    """检查必需的依赖是否已安装"""
    try:
        import sentencepiece
        logger.info("SentencePiece 库已安装")
    except ImportError:
        logger.error("缺少 SentencePiece 库！请运行: pip install sentencepiece")
        raise ImportError(
            "SentencePiece 库未安装。请运行以下命令安装：\n"
            "pip install sentencepiece\n"
            "安装完成后请重启 Python 环境。"
        )

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    def __init__(self, model_path: str = None, cache_dir: str = '/mnt/sda/shenhao/models'):
        """
        初始化图像特征提取器
        :param model_path: 模型路径，可以是本地路径或HuggingFace模型名
        :param cache_dir: 自定义缓存目录
        """
        # 首先检查依赖
        check_dependencies()
        
        # 使用环境变量或默认值
        self.model_path = model_path or os.getenv(
            'SIGLIP_MODEL_PATH', 
            'google/siglip-so400m-patch14-384'  # 使用HuggingFace官方路径
        )
        
        # 设置缓存目录
        self.cache_dir = cache_dir
        
        # 显示缓存路径信息
        if not os.path.exists(self.model_path):
            # 如果是HuggingFace模型名，显示缓存路径
            from huggingface_hub import _CACHED_NO_EXIST
            from transformers.utils import TRANSFORMERS_CACHE
            effective_cache_dir = self.cache_dir or os.getenv('HF_HOME', os.getenv('TRANSFORMERS_CACHE', TRANSFORMERS_CACHE))
            logger.info(f"模型将缓存到: {effective_cache_dir}")
            logger.info(f"完整模型路径将是: {effective_cache_dir}/models--{self.model_path.replace('/', '--')}")
        
        try:
            # 使用自定义缓存目录加载模型
            logger.info("正在加载处理器...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            logger.info("处理器加载成功")
            
            logger.info("正在加载模型...")
            self.model = AutoModel.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            logger.info("模型加载成功")
            
            # 显示实际加载的模型路径
            if hasattr(self.model.config, '_name_or_path'):
                logger.info(f"实际加载的模型路径: {self.model.config._name_or_path}")
            
            # 设备管理
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info(f"模型已加载到设备: {self.device}")
            
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
            
            # 打印原始图片信息
            logger.info(f"原始图片尺寸: {image.size}")
            logger.info(f"原始图片模式: {image.mode}")
            
            # 确保图片是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info("图片已转换为RGB格式")
            
            # 处理图像输入
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # 打印处理后的输入张量信息
            if 'pixel_values' in inputs:
                logger.info(f"处理后的输入张量维度: {inputs['pixel_values'].shape}")
            
            # 提取特征
            with torch.no_grad():
                if extract_patch_features:
                    # 获取patch级别的特征
                    vision_outputs = self.model.vision_model(**inputs)
                    # vision_outputs.last_hidden_state 包含所有patch的特征
                    # 形状通常是 [batch_size, num_patches, hidden_size]
                    patch_features = vision_outputs.last_hidden_state
                    
                    logger.info(f"提取的patch特征维度: {patch_features.shape}")
                    logger.info(f"patch数量: {patch_features.shape[1]}")
                    logger.info(f"每个patch特征维度: {patch_features.shape[2]}")
                    
                    # L2 归一化每个patch特征
                    patch_features = patch_features / patch_features.norm(p=2, dim=-1, keepdim=True)
                    
                    logger.info(f"L2归一化后patch特征范围: [{patch_features.min().item():.4f}, {patch_features.max().item():.4f}]")
                    
                    feature_array = patch_features.cpu().numpy()
                    logger.info(f"最终patch特征numpy数组维度: {feature_array.shape}")
                    
                    return feature_array
                else:
                    # 获取全局图像特征（原有方式）
                    image_features = self.model.get_image_features(**inputs)
                    
                    logger.info(f"提取的全局特征维度: {image_features.shape}")
                    logger.info(f"特征数据类型: {image_features.dtype}")
                    logger.info(f"特征范围: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")
                    
                    # L2 归一化
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    logger.info(f"L2归一化后特征范围: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")
                    logger.info(f"特征向量的L2范数: {image_features.norm(p=2, dim=-1).item():.4f}")
                    
                    feature_array = image_features.cpu().numpy()
                    logger.info(f"最终全局特征numpy数组维度: {feature_array.shape}")
                    
                    return feature_array
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise

# 示例用法
if __name__ == "__main__":
    try:
        logger.info("开始初始化特征提取器...")
        extractor = ImageFeatureExtractor()
        
        # 使用本地图片进行测试
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        
        # 测试全局特征提取
        print("\n=== 测试全局特征提取 ===")
        global_features = extractor.get_image_features(url, extract_patch_features=False)
        print(f"全局特征形状: {global_features.shape}")
        print(f"全局特征类型: {type(global_features)}")
        
        # 测试patch级别特征提取
        print("\n=== 测试patch级别特征提取 ===")
        patch_features = extractor.get_image_features(url, extract_patch_features=True)
        print(f"patch特征形状: {patch_features.shape}")
        print(f"patch特征类型: {type(patch_features)}")
        print(f"patch数量: {patch_features.shape[1]}")
        print(f"每个patch特征维度: {patch_features.shape[2]}")
        
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("请按照提示安装缺失的依赖库。")
    except Exception as e:
        print(f"运行出错: {e}")
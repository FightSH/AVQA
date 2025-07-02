import os
import sys
import numpy as np
from PIL import Image
import requests
import torch
from typing import Union, List
import logging
import tempfile

# 添加ImageBind路径（如果使用本地克隆的仓库）
sys.path.append('/mnt/sda/shenhao/code/ImageBind')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    def __init__(self, model_name: str = "imagebind_huge", cache_dir: str = '/mnt/sda/shenhao/models', 
                 local_model_path: str = None, extract_patch_features: bool = False):
        """
        初始化ImageBind图像特征提取器
        :param model_name: 模型名称 ("imagebind_huge" 或其他可用模型)
        :param cache_dir: 自定义缓存目录
        :param local_model_path: 本地模型文件路径 (.pth文件)，如果指定则优先使用本地文件
        :param extract_patch_features: 是否提取patch级别特征
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_model_path = local_model_path
        self.extract_patch_features = extract_patch_features
        
        try:
            # 导入ImageBind - 修复重复导入
            try:
                from imagebind.models import imagebind_model
                from imagebind import data
                logger.info("ImageBind模块导入成功")
            except ImportError:
                logger.error("ImageBind未安装！请按照以下步骤安装：")
                logger.error("1. git clone https://github.com/facebookresearch/ImageBind.git")
                logger.error("2. cd ImageBind")
                logger.error("3. pip install -e .")
                raise ImportError("ImageBind未安装，请先安装ImageBind")
            
            # 保存数据处理模块的引用
            self.data = data
            
            # 加载模型
            if self.local_model_path and os.path.exists(self.local_model_path):
                logger.info(f"从本地路径加载ImageBind模型: {self.local_model_path}")
                # 先创建模型架构（不加载预训练权重）
                self.model = imagebind_model.imagebind_huge(pretrained=False)
                
                # 加载本地权重文件
                try:
                    checkpoint = torch.load(self.local_model_path, map_location='cpu')
                    
                    # 处理不同的checkpoint格式
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # 加载权重
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    
                    if missing_keys:
                        logger.warning(f"缺失的权重键: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"意外的权重键: {unexpected_keys}")
                    
                    logger.info("本地模型权重加载成功")
                    
                except Exception as e:
                    logger.error(f"加载本地模型权重失败: {e}")
                    logger.info("回退到在线下载预训练模型...")
                    self.model = imagebind_model.imagebind_huge(pretrained=True)
                    
            elif self.local_model_path:
                logger.warning(f"指定的本地模型文件不存在: {self.local_model_path}")
                logger.info("回退到在线下载预训练模型...")
                self.model = imagebind_model.imagebind_huge(pretrained=True)
            else:
                logger.info(f"在线加载ImageBind预训练模型: {self.model_name}")
                self.model = imagebind_model.imagebind_huge(pretrained=True)
            
            # 设备管理 - 增强设备检查
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                self.device = torch.device("cpu")
                logger.info("使用CPU")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"模型已加载到设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _register_hooks(self):
        """注册hook来获取中间层特征"""
        self.patch_features = None
        
        def hook_fn(module, input, output):
            # 保存最后一层transformer输出（在全局池化之前）
            self.patch_features = output
        
        # 为vision encoder的最后一层注册hook
        if hasattr(self.model, 'modality_encoders') and 'vision' in self.model.modality_encoders:
            vision_encoder = self.model.modality_encoders['vision']
            if hasattr(vision_encoder, 'trunk') and hasattr(vision_encoder.trunk, 'blocks'):
                # 在最后一个transformer block后注册hook
                vision_encoder.trunk.blocks[-1].register_forward_hook(hook_fn)
                logger.info("已注册patch特征提取hook")
            else:
                logger.warning("无法找到vision encoder的transformer blocks，patch特征提取可能不可用")
        else:
            logger.warning("无法找到vision encoder，patch特征提取不可用")
    
    def get_image_features(self, image: Union[Image.Image, str, np.ndarray], 
                          feature_type: str = "global") -> Union[np.ndarray, dict]:
        """
        使用ImageBind提取图片特征
        :param image: 输入图片 (PIL.Image, 文件路径, 或numpy数组)
        :param feature_type: 特征类型 ("global", "patch", "both")
        :return: 图片特征向量 (numpy.ndarray) 或包含不同类型特征的字典
        """
        temp_path = None
        try:
            # 处理不同类型的输入 - 增强错误处理
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    # 网络图片 - 增强网络请求处理
                    try:
                        response = requests.get(image, stream=True, timeout=30)
                        response.raise_for_status()
                        image = Image.open(response.raw)
                    except requests.RequestException as e:
                        logger.error(f"网络图片下载失败: {e}")
                        raise
                else:
                    # 本地文件 - 检查文件存在性
                    if not os.path.exists(image):
                        raise FileNotFoundError(f"图片文件不存在: {image}")
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
            
            # 将PIL图像转换为临时文件 - 使用更安全的临时文件处理
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image.save(tmp_file.name, 'JPEG', quality=95)
                    temp_path = tmp_file.name
            except Exception as e:
                logger.error(f"临时文件创建失败: {e}")
                raise
            
            try:
                # 使用ImageBind的数据加载和预处理
                inputs = {
                    'vision': self.data.load_and_transform_vision_data([temp_path], self.device)
                }
                
                # 打印处理后的输入张量信息
                logger.info(f"处理后的输入张量维度: {inputs['vision'].shape}")
                
                # 重置patch特征
                if self.extract_patch_features:
                    self.patch_features = None
                
                # 提取特征
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    global_features = embeddings['vision']
                
                # 处理全局特征
                if global_features.dim() > 2:
                    global_features = global_features.squeeze()
                if global_features.dim() == 1:
                    global_features = global_features.unsqueeze(0)
                
                logger.info(f"提取的全局特征维度: {global_features.shape}")
                
                # 处理patch特征
                patch_features_array = None
                if self.extract_patch_features and self.patch_features is not None:
                    patch_features = self.patch_features
                    
                    # 移除CLS token (通常是第一个token)
                    if patch_features.dim() == 3:  # [batch, seq_len, dim]
                        # 假设第一个token是CLS token
                        patch_features = patch_features[:, 1:, :]  # 移除CLS token
                    
                    # 压缩批次维度
                    if patch_features.shape[0] == 1:
                        patch_features = patch_features.squeeze(0)
                    
                    patch_features_array = patch_features.cpu().numpy()
                    
                    # 计算patch特征的空间维度
                    num_patches = patch_features_array.shape[0]
                    patch_size = int(np.sqrt(num_patches))
                    
                    logger.info(f"提取的patch特征维度: {patch_features_array.shape}")
                    logger.info(f"patch数量: {num_patches}, 推断的patch grid: {patch_size}x{patch_size}")
                
                # 全局特征归一化处理
                feature_norm = global_features.norm(p=2, dim=-1)
                logger.info(f"全局特征向量的L2范数: {feature_norm.item():.4f}")
                
                if feature_norm.item() > 1.1:
                    global_features = global_features / global_features.norm(p=2, dim=-1, keepdim=True)
                    logger.info("应用了额外的L2归一化")
                
                # 转换为numpy数组
                global_features_array = global_features.cpu().numpy()
                if global_features_array.shape[0] == 1:
                    global_features_array = global_features_array.squeeze(0)
                
                logger.info(f"最终全局特征numpy数组维度: {global_features_array.shape}")
                
                # 根据请求的特征类型返回结果
                if feature_type == "global":
                    return global_features_array
                elif feature_type == "patch" and patch_features_array is not None:
                    return patch_features_array
                elif feature_type == "both":
                    result = {"global": global_features_array}
                    if patch_features_array is not None:
                        result["patch"] = patch_features_array
                        result["patch_info"] = {
                            "num_patches": num_patches,
                            "patch_grid": f"{patch_size}x{patch_size}",
                            "feature_dim": patch_features_array.shape[-1]
                        }
                    return result
                else:
                    if feature_type == "patch" and patch_features_array is None:
                        logger.warning("请求patch特征但未启用patch特征提取，返回全局特征")
                    return global_features_array
                
            finally:
                # 清理临时文件 - 增强错误处理
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        logger.debug(f"临时文件已清理: {temp_path}")
                    except OSError as e:
                        logger.warning(f"临时文件清理失败: {e}")
                    
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 确保临时文件被清理
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise

# 示例用法
if __name__ == "__main__":
    try:
        logger.info("开始初始化ImageBind特征提取器...")
        
        # 可以指定本地模型路径和是否提取patch特征
        local_model_path = "/mnt/sda/shenhao/models/imagebind/.checkpoints/imagebind_huge.pth"
        
        # 启用patch特征提取
        extractor = ImageFeatureExtractor(
            local_model_path=local_model_path,
            extract_patch_features=True
        )
        
        # 使用本地图片进行测试或网络图片
        test_urls = [
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            # 可以添加本地文件路径进行测试
            # "/path/to/local/image.jpg"
        ]
        
        for url in test_urls:
            try:
                logger.info(f"处理图片: {url}")
                
                # 提取不同类型的特征
                print(f"\n=== 处理图片: {url} ===")
                
                # 1. 只提取全局特征
                global_features = extractor.get_image_features(url, feature_type="global")
                print(f"全局特征形状: {global_features.shape}")
                print(f"全局特征范围: [{global_features.min():.4f}, {global_features.max():.4f}]")
                
                # 2. 只提取patch特征
                patch_features = extractor.get_image_features(url, feature_type="patch")
                if patch_features is not None:
                    print(f"Patch特征形状: {patch_features.shape}")
                    print(f"Patch特征范围: [{patch_features.min():.4f}, {patch_features.max():.4f}]")
                
                # 3. 提取所有特征
                all_features = extractor.get_image_features(url, feature_type="both")
                if isinstance(all_features, dict):
                    print(f"所有特征包含: {list(all_features.keys())}")
                    if "patch_info" in all_features:
                        print(f"Patch信息: {all_features['patch_info']}")
                
            except Exception as e:
                logger.error(f"处理图片 {url} 失败: {e}")
        
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("请按照以下步骤安装ImageBind：")
        print("1. git clone https://github.com/facebookresearch/ImageBind.git")
        print("2. cd ImageBind") 
        print("3. pip install -e .")
    except Exception as e:
        print(f"运行出错: {e}")
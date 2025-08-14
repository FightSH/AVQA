"""
CA模块集成指南和示例
展示如何将CA模块集成到不同的任务中
"""

import torch
import torch.nn as nn
from ca_module_standalone import StandaloneCA
from typing import Optional, Dict, Any


class MultiModalClassifier(nn.Module):
    """
    示例1: 多模态分类任务
    使用CA模块进行视频-音频-文本融合，然后进行分类
    """
    def __init__(self, 
                 num_classes: int,
                 ca_config: Dict[str, Any] = None):
        super().__init__()
        
        # 默认CA配置
        if ca_config is None:
            ca_config = {
                'video_input_dim': 1024,
                'audio_input_dim': 1024,
                'text_input_dim': 768,  # 如果使用BERT特征
                'num_query_tokens': 32,
                'output_dim': 512
            }
        
        # CA模块
        self.ca_module = StandaloneCA(**ca_config)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(ca_config['output_dim'] * 2, 256),  # video + audio features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_features, video_features, audio_features):
        # CA模块融合
        video_output, audio_output = self.ca_module(text_features, video_features, audio_features)
        
        # 全局池化
        video_pooled = video_output.mean(dim=1)  # [batch, output_dim]
        audio_pooled = audio_output.mean(dim=1)  # [batch, output_dim]
        
        # 特征拼接
        fused_features = torch.cat([video_pooled, audio_pooled], dim=-1)
        
        # 分类
        logits = self.classifier(fused_features)
        return logits


class MultiModalQA(nn.Module):
    """
    示例2: 多模态问答任务
    使用CA模块处理问题引导的多模态理解
    """
    def __init__(self, 
                 vocab_size: int,
                 ca_config: Dict[str, Any] = None,
                 decoder_config: Dict[str, Any] = None):
        super().__init__()
        
        # 默认配置
        if ca_config is None:
            ca_config = {
                'video_input_dim': 1024,
                'audio_input_dim': 1024,
                'text_input_dim': 768,
                'num_query_tokens': 32,
                'output_dim': 768
            }
            
        if decoder_config is None:
            decoder_config = {
                'hidden_size': 768,
                'num_layers': 6,
                'num_heads': 8
            }
        
        # CA模块
        self.ca_module = StandaloneCA(**ca_config)
        
        # 简单的解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=decoder_config['hidden_size'],
                nhead=decoder_config['num_heads'],
                batch_first=True
            ),
            num_layers=decoder_config['num_layers']
        )
        
        # 输出投影
        self.output_projection = nn.Linear(decoder_config['hidden_size'], vocab_size)
        
    def forward(self, question_features, video_features, audio_features, target_sequence=None):
        # CA模块处理
        video_output, audio_output = self.ca_module(question_features, video_features, audio_features)
        
        # 拼接多模态特征作为memory
        multimodal_memory = torch.cat([video_output, audio_output], dim=1)  # [batch, 64, dim]
        
        if target_sequence is not None:
            # 训练模式：使用目标序列
            output = self.decoder(target_sequence, multimodal_memory)
            logits = self.output_projection(output)
            return logits
        else:
            # 推理模式：自回归生成
            return self.generate(multimodal_memory)
    
    def generate(self, memory, max_length=50):
        # 简化的生成逻辑
        batch_size = memory.shape[0]
        device = memory.device
        
        # 开始token
        generated = torch.zeros(batch_size, 1, memory.shape[-1]).to(device)
        
        for _ in range(max_length):
            output = self.decoder(generated, memory)
            next_token_logits = self.output_projection(output[:, -1:, :])
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # 这里需要将token转换为embedding，简化处理
            # generated = torch.cat([generated, next_token_embedding], dim=1)
            break  # 简化版本
            
        return generated


class VideoAudioRetrieval(nn.Module):
    """
    示例3: 视频-音频检索任务
    使用CA模块学习跨模态表示用于检索
    """
    def __init__(self, ca_config: Dict[str, Any] = None):
        super().__init__()
        
        if ca_config is None:
            ca_config = {
                'video_input_dim': 1024,
                'audio_input_dim': 1024,
                'text_input_dim': 768,
                'num_query_tokens': 16,  # 检索任务可以用更少的query
                'output_dim': 512
            }
        
        self.ca_module = StandaloneCA(**ca_config)
        
        # 投影到统一的嵌入空间
        self.video_projection = nn.Linear(ca_config['output_dim'], 256)
        self.audio_projection = nn.Linear(ca_config['output_dim'], 256)
        self.text_projection = nn.Linear(ca_config['text_input_dim'], 256)
        
    def encode_multimodal(self, text_features, video_features, audio_features):
        """编码多模态内容"""
        video_output, audio_output = self.ca_module(text_features, video_features, audio_features)
        
        # 池化和投影
        video_embed = self.video_projection(video_output.mean(dim=1))
        audio_embed = self.audio_projection(audio_output.mean(dim=1))
        
        # 融合视频和音频
        multimodal_embed = (video_embed + audio_embed) / 2
        return multimodal_embed
    
    def encode_text(self, text_features):
        """编码文本查询"""
        text_embed = self.text_projection(text_features.mean(dim=0, keepdim=True))
        return text_embed
    
    def compute_similarity(self, text_embed, multimodal_embed):
        """计算相似度"""
        # 余弦相似度
        text_norm = nn.functional.normalize(text_embed, p=2, dim=-1)
        mm_norm = nn.functional.normalize(multimodal_embed, p=2, dim=-1)
        similarity = torch.mm(text_norm, mm_norm.t())
        return similarity


def create_ca_config_for_task(task_type: str) -> Dict[str, Any]:
    """
    为不同任务创建CA配置
    """
    base_config = {
        'video_input_dim': 1024,
        'audio_input_dim': 1024,
        'text_input_dim': 768,
        'num_query_tokens': 32,
        'output_dim': 512
    }
    
    if task_type == "classification":
        # 分类任务：较少的query tokens，关注全局特征
        base_config.update({
            'num_query_tokens': 16,
            'output_dim': 256
        })
    elif task_type == "qa":
        # 问答任务：更多的query tokens，需要细粒度信息
        base_config.update({
            'num_query_tokens': 64,
            'output_dim': 768
        })
    elif task_type == "retrieval":
        # 检索任务：中等query tokens，平衡效率和效果
        base_config.update({
            'num_query_tokens': 32,
            'output_dim': 512
        })
    elif task_type == "generation":
        # 生成任务：大量query tokens，需要丰富信息
        base_config.update({
            'num_query_tokens': 128,
            'output_dim': 1024
        })
    
    return base_config


def load_and_adapt_ca_weights(ca_module: StandaloneCA, 
                             original_checkpoint_path: str,
                             adaptation_strategy: str = "freeze_perceiver"):
    """
    加载原始权重并进行任务适配
    
    Args:
        ca_module: CA模块实例
        original_checkpoint_path: 原始模型权重路径
        adaptation_strategy: 适配策略
    """
    
    # 加载预训练权重
    ca_module.load_pretrained_weights(original_checkpoint_path, strict=False)
    
    if adaptation_strategy == "freeze_perceiver":
        # 冻结Perceiver，只微调Q-Former
        for param in ca_module.perceiver.parameters():
            param.requires_grad = False
            
    elif adaptation_strategy == "freeze_qformer":
        # 冻结Q-Former，只微调Perceiver
        for param in ca_module.video_qformer.parameters():
            param.requires_grad = False
        for param in ca_module.audio_qformer.parameters():
            param.requires_grad = False
            
    elif adaptation_strategy == "freeze_all":
        # 冻结整个CA模块，只作为特征提取器
        for param in ca_module.parameters():
            param.requires_grad = False
            
    elif adaptation_strategy == "finetune_all":
        # 全部微调
        pass
    
    print(f"Applied adaptation strategy: {adaptation_strategy}")


# 使用示例
if __name__ == "__main__":
    # 示例1: 多模态分类
    print("=== 多模态分类示例 ===")
    classifier = MultiModalClassifier(num_classes=10)
    
    # 模拟数据
    text_feat = torch.randn(15, 768)  # 15个词的BERT特征
    video_feat = torch.randn(60, 1024)  # 60帧视频特征
    audio_feat = torch.randn(60, 1024)  # 60帧音频特征
    
    logits = classifier(text_feat, video_feat, audio_feat)
    print(f"分类输出形状: {logits.shape}")
    
    # 示例2: 检索任务
    print("\n=== 检索任务示例 ===")
    retrieval_model = VideoAudioRetrieval()
    
    # 编码多模态内容
    mm_embed = retrieval_model.encode_multimodal(text_feat, video_feat, audio_feat)
    text_embed = retrieval_model.encode_text(text_feat)
    
    # 计算相似度
    similarity = retrieval_model.compute_similarity(text_embed, mm_embed)
    print(f"相似度矩阵形状: {similarity.shape}")
    
    # 示例3: 任务特定配置
    print("\n=== 任务配置示例 ===")
    qa_config = create_ca_config_for_task("qa")
    print(f"问答任务配置: {qa_config}")
    
    classification_config = create_ca_config_for_task("classification")
    print(f"分类任务配置: {classification_config}")
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAdjuster(nn.Module):
    def __init__(self):
        super(FeatureAdjuster, self).__init__()
        self.relu_activation = nn.ReLU(inplace=False)

    def forward(self, feature_a, feature_b):
        cosine_similarity = F.cosine_similarity(feature_b, feature_a, dim=2)
        cosine_similarity = cosine_similarity.unsqueeze(2)

        feature_a = feature_a+feature_b*cosine_similarity

        feature_b = feature_b+feature_a*cosine_similarity

        feature_a = self.relu_activation(feature_a)
        feature_b = self.relu_activation(feature_b)
        return feature_a,feature_b


class AudioPatchWeightedFusion(nn.Module):
    """方案1: 简单加权融合"""
    def __init__(self, d_model=512):
        super().__init__()
        # 学习融合权重
        self.audio_weight = nn.Parameter(torch.tensor(0.7))  # audio的权重
        self.patch_weight = nn.Parameter(torch.tensor(0.3))  # audio_patch的权重
        
    def forward(self, audio, audio_patch):
        # audio: [B, T, D], audio_patch: [B, T, D]
        # 归一化权重
        total_weight = self.audio_weight + self.patch_weight
        w_audio = self.audio_weight / total_weight
        w_patch = self.patch_weight / total_weight
        
        return w_audio * audio + w_patch * audio_patch


class AudioPatchGatedFusion(nn.Module):
    """方案2: 基于门控机制的Audio和Audio_patch融合（推荐）"""
    def __init__(self, d_model=512):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Sigmoid()  # 输出两个门控值，分别控制audio和audio_patch
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, audio, audio_patch):
        # audio: [B, T, D], audio_patch: [B, T, D]
        # 拼接特征用于门控计算
        concat_feat = torch.cat([audio, audio_patch], dim=-1)  # [B, T, 2*D]
        
        # 计算门控权重
        gates = self.gate_network(concat_feat)  # [B, T, 2]
        gate_audio = gates[:, :, 0:1]  # [B, T, 1]
        gate_patch = gates[:, :, 1:2]  # [B, T, 1]
        
        # 门控融合
        fused = gate_audio * audio + gate_patch * audio_patch
        return self.norm(fused)


class AudioPatchResidualFusion(nn.Module):
    """方案3: 残差连接式融合"""
    def __init__(self, d_model=512):
        super().__init__()
        self.patch_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, audio, audio_patch):
        # 将audio_patch作为残差信息加到audio上
        transformed_patch = self.patch_transform(audio_patch)
        fused = audio + transformed_patch
        return self.norm(fused)


class AudioPatchAttentionFusion(nn.Module):
    """方案4: 基于注意力的融合"""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, audio, audio_patch):
        # audio: [B, T, D], audio_patch: [B, T, D]
        # 转换维度用于注意力计算
        audio_t = audio.permute(1, 0, 2)  # [T, B, D]
        patch_t = audio_patch.permute(1, 0, 2)  # [T, B, D]
        
        # audio作为query，audio_patch作为key和value
        attn_out = self.cross_attn(audio_t, patch_t, patch_t)[0]  # [T, B, D]
        
        # 残差连接并转换回原维度
        fused = (audio_t + self.dropout(attn_out)).permute(1, 0, 2)  # [B, T, D]
        return self.norm(fused)


class AudioPatchConcatFusion(nn.Module):
    """方案5: 直接拼接后投影融合"""
    def __init__(self, d_model=512):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, audio, audio_patch):
        # audio: [B, T, D], audio_patch: [B, T, D]
        # 直接拼接
        concat_feat = torch.cat([audio, audio_patch], dim=-1)  # [B, T, 2*D]
        # 投影回原维度
        fused = self.fusion_proj(concat_feat)  # [B, T, D]
        return self.norm(fused)


class AudioPatchCosineFusion(nn.Module):
    """方案6: 基于余弦相似度的融合（类似你的FeatureAdjuster）"""
    def __init__(self, d_model=512):
        super().__init__()
        self.relu_activation = nn.ReLU(inplace=False)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, audio, audio_patch):
        # audio: [B, T, D], audio_patch: [B, T, D]
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(audio_patch, audio, dim=2)  # [B, T]
        cosine_similarity = cosine_similarity.unsqueeze(2)  # [B, T, 1]
        
        # 基于相似度进行特征调整
        enhanced_audio = audio + audio_patch * cosine_similarity
        enhanced_patch = audio_patch + audio * cosine_similarity
        
        # 融合两个增强特征
        fused = (enhanced_audio + enhanced_patch) / 2
        fused = self.relu_activation(fused)
        return self.norm(fused)


def create_fusion_module(fusion_type='gated', d_model=512, nhead=8):
    """
    工厂函数：根据融合类型创建对应的融合模块
    这样只会创建需要的模块，避免内存浪费和梯度问题
    """
    if fusion_type == 'weighted':
        return AudioPatchWeightedFusion(d_model)
    elif fusion_type == 'gated':
        return AudioPatchGatedFusion(d_model)
    elif fusion_type == 'residual':
        return AudioPatchResidualFusion(d_model)
    elif fusion_type == 'attention':
        return AudioPatchAttentionFusion(d_model, nhead)
    elif fusion_type == 'concat':
        return AudioPatchConcatFusion(d_model)
    elif fusion_type == 'cosine':
        return AudioPatchCosineFusion(d_model)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")


class MultiModalFusionHub(nn.Module):
    """多模态融合中心，只初始化指定的融合方式"""
    def __init__(self, d_model=512, nhead=8, fusion_type='gated'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # 只初始化指定的融合模块
        self.fusion_module = create_fusion_module(fusion_type, d_model, nhead)
        
    def forward(self, audio, audio_patch):
        """
        Args:
            audio: [B, T, D] 音频特征
            audio_patch: [B, T, D] 音频相关的patch特征
        """
        return self.fusion_module(audio, audio_patch)
    
    def change_fusion_type(self, new_fusion_type, d_model=512, nhead=8):
        """动态改变融合方式（主要用于实验）"""
        self.fusion_type = new_fusion_type
        self.fusion_module = create_fusion_module(new_fusion_type, d_model, nhead)


class MultiModalFusionHubFlexible(nn.Module):
    """灵活的多模态融合中心，支持运行时切换（实验用）"""
    def __init__(self, d_model=512, nhead=8, fusion_types=['gated']):
        super().__init__()
        self.fusion_types = fusion_types
        
        # 只初始化指定的融合模块
        self.fusion_modules = nn.ModuleDict()
        for fusion_type in fusion_types:
            self.fusion_modules[fusion_type] = create_fusion_module(fusion_type, d_model, nhead)
        
    def forward(self, audio, audio_patch, fusion_type='gated'):
        """
        Args:
            audio: [B, T, D] 音频特征
            audio_patch: [B, T, D] 音频相关的patch特征
            fusion_type: 融合方式
        """
        if fusion_type not in self.fusion_modules:
            raise ValueError(f"Fusion type '{fusion_type}' not available. Available types: {list(self.fusion_modules.keys())}")
        
        return self.fusion_modules[fusion_type](audio, audio_patch)


class AudioPatchMambaFusion(nn.Module):
    """专为Mamba架构设计的Audio-Patch融合"""
    def __init__(self, d_model=512):
        super().__init__()
        # 使用简单的线性投影，类似Mamba的设计
        self.audio_proj = nn.Linear(d_model, d_model)
        self.patch_proj = nn.Linear(d_model, d_model)
        
        # Mamba风格的门控（简化版）
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 层归一化，与Mamba保持一致
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, audio, audio_patch):
        # 线性投影
        audio_proj = self.audio_proj(audio)
        patch_proj = self.patch_proj(audio_patch)
        
        # 计算门控权重
        concat_feat = torch.cat([audio, audio_patch], dim=-1)
        gate_weight = self.gate(concat_feat)
        
        # 门控融合 + 残差连接（Mamba风格）
        fused = audio + gate_weight * (audio_proj + patch_proj - audio)
        
        return self.norm(fused)


class VideoPatchFusion(nn.Module):
    """视频和视频patch的融合（与音频融合类似）"""
    def __init__(self, d_model=512, fusion_type='gated'):
        super().__init__()
        self.fusion_module = create_fusion_module(fusion_type, d_model)
        
    def forward(self, video, video_patch):
        return self.fusion_module(video, video_patch)


if __name__ == "__main__":
    # Example usage
    batch_size, time_steps, d_model = 10, 60, 512
    
    # 创建示例数据
    audio = torch.randn(batch_size, time_steps, d_model)  # 音频特征
    audio_patch = torch.randn(batch_size, time_steps, d_model)  # 音频patch特征
    video = torch.randn(batch_size, time_steps, d_model)  # 视频特征
    video_patch = torch.randn(batch_size, time_steps, d_model)  # 视频patch特征
    
    print("Testing different fusion methods:")
    print(f"Input shapes - Audio: {audio.shape}, Audio_patch: {audio_patch.shape}")
    
    # 测试各种融合方法
    fusion_methods = ['weighted', 'gated', 'residual', 'attention', 'concat', 'cosine']
    
    # 方法1: 使用单一融合模块（推荐）
    print("Method 1: Single fusion module")
    for method in fusion_methods:
        try:
            fusion_hub = MultiModalFusionHub(d_model=d_model, fusion_type=method)
            fused_audio = fusion_hub(audio, audio_patch)
            print(f"{method.capitalize()} fusion - Output shape: {fused_audio.shape}")
        except Exception as e:
            print(f"{method.capitalize()} fusion - Error: {e}")
    
    # 方法2: 使用灵活融合中心（实验用）
    print("\nMethod 2: Flexible fusion hub")
    flexible_hub = MultiModalFusionHubFlexible(d_model=d_model, fusion_types=['gated', 'attention', 'weighted'])
    for method in ['gated', 'attention', 'weighted']:
        try:
            fused_audio = flexible_hub(audio, audio_patch, fusion_type=method)
            print(f"{method.capitalize()} fusion - Output shape: {fused_audio.shape}")
        except Exception as e:
            print(f"{method.capitalize()} fusion - Error: {e}")
    
    # 测试原始的FeatureAdjuster
    print("\nTesting original FeatureAdjuster:")
    adjuster = FeatureAdjuster()
    adjusted_a, adjusted_b = adjuster(audio, audio_patch)
    print(f"FeatureAdjuster - Output shapes: {adjusted_a.shape}, {adjusted_b.shape}")
    
    # 测试视频融合
    print("\nTesting video fusion:")
    video_fusion = VideoPatchFusion(d_model=d_model, fusion_type='gated')
    fused_video = video_fusion(video, video_patch)
    print(f"Video fusion - Output shape: {fused_video.shape}")
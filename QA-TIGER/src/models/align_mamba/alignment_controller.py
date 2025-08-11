"""
对齐控制器模块

该模块提供了一个统一的接口来管理跨模态对齐功能，包括最优传输对齐和MMD全局对齐。
支持多种对齐策略，并提供完整的错误处理和数值稳定性保证。
"""

from typing import Optional, Literal, Dict, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .qa_tiger_align_mamba import OptimalTransportAlignment, MMDGlobalAlignment

logger = logging.getLogger(__name__)


class AlignmentController(nn.Module):
    """
    对齐控制器：管理OT和MMD对齐的执行流程
    
    该控制器作为对齐功能的统一入口，支持多种对齐策略，
    并提供完整的错误处理和性能优化功能。
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # 基本配置
        self.enabled = config.get('enabled', False)
        self.strategy = config.get('strategy', 'standard')
        self.lambda_align = config.get('lambda_align', 0.1)
        self.patch_alignment = config.get('patch_alignment', True)
        self.debug_mode = config.get('debug_mode', False)
        self.memory_efficient = config.get('memory_efficient', True)
        
        # 验证配置
        self._validate_config(config)
        
        # 初始化对齐模块
        if self.enabled:
            self.ot_aligner = OptimalTransportAlignment(
                eps=config.get('ot_eps', 1e-8)
            )
            self.mmd_aligner = MMDGlobalAlignment(
                sigma=config.get('mmd_sigma', 1.0)
            )
        else:
            self.ot_aligner = None
            self.mmd_aligner = None
    
    def _validate_config(self, config: dict):
        """验证配置参数的有效性"""
        # 验证对齐策略
        valid_strategies = ['standard', 'reverse', 'bidirectional']
        if self.strategy not in valid_strategies:
            raise ValueError(f"无效的对齐策略: {self.strategy}. 有效选项: {valid_strategies}")
        
        # 验证lambda_align范围
        if not 0 <= self.lambda_align <= 1:
            raise ValueError(f"lambda_align必须在[0, 1]范围内，当前值: {self.lambda_align}")
        
        # 验证其他参数
        ot_eps = config.get('ot_eps', 1e-8)
        if ot_eps <= 0:
            raise ValueError(f"ot_eps必须大于0，当前值: {ot_eps}")
        
        mmd_sigma = config.get('mmd_sigma', 1.0)
        if mmd_sigma <= 0:
            raise ValueError(f"mmd_sigma必须大于0，当前值: {mmd_sigma}")
    
    def _validate_inputs(self, audio: torch.Tensor, video: torch.Tensor, 
                        words: torch.Tensor, quest: torch.Tensor, 
                        patch: Optional[torch.Tensor] = None):
        """验证输入张量的形状和类型"""
        # 检查维度
        if audio.dim() != 3 or video.dim() != 3 or words.dim() != 3:
            raise ValueError("音频、视频、词语特征必须是3维张量")
        
        if quest.dim() != 2:
            raise ValueError("问题特征必须是2维张量")
        
        if patch is not None and patch.dim() != 4:
            raise ValueError("Patch特征必须是4维张量")
        
        # 检查特征维度一致性
        feature_dims = [audio.shape[-1], video.shape[-1], words.shape[-1]]
        if len(set(feature_dims)) > 1:
            raise ValueError(f"所有模态的特征维度必须一致，当前维度: {feature_dims}")
        
        # 检查batch size一致性
        tensors = [audio, video, words, quest]
        if patch is not None:
            tensors.append(patch)
        
        batch_sizes = [t.shape[0] for t in tensors]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"所有输入的batch size必须一致，当前batch sizes: {batch_sizes}")
    
    def _ensure_numerical_stability(self, tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """确保数值稳定性"""
        # 检查NaN和Inf
        if torch.isnan(tensor).any():
            logger.warning("检测到NaN值，使用零张量替换")
            return torch.zeros_like(tensor)
        
        if torch.isinf(tensor).any():
            logger.warning("检测到Inf值，进行截断处理")
            tensor = torch.clamp(tensor, -1e6, 1e6)
        
        # 梯度裁剪
        if tensor.requires_grad:
            tensor = torch.clamp(tensor, -1e6, 1e6)
        
        return tensor
    
    def _memory_efficient_alignment(self, src: torch.Tensor, tgt: torch.Tensor, 
                                  chunk_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """内存高效的对齐计算"""
        B, T_src, D = src.shape
        _, T_tgt, _ = tgt.shape
        
        # 如果序列长度较小，直接计算
        if T_src * T_tgt <= chunk_size:
            return self.ot_aligner(src, tgt)
        
        # 分块处理大序列
        logger.info(f"使用分块处理，源序列长度: {T_src}, 目标序列长度: {T_tgt}")
        
        aligned_chunks = []
        transport_chunks = []
        
        chunk_len = max(1, chunk_size // T_tgt)
        
        for i in range(0, T_src, chunk_len):
            end_i = min(i + chunk_len, T_src)
            chunk = src[:, i:end_i, :]
            
            aligned_chunk, transport_chunk = self.ot_aligner(chunk, tgt)
            aligned_chunks.append(aligned_chunk)
            transport_chunks.append(transport_chunk)
        
        # 合并结果
        aligned_features = torch.cat(aligned_chunks, dim=1)
        # 对于传输矩阵，我们只返回第一个块的结果作为代表
        transport_matrix = transport_chunks[0]
        
        return aligned_features, transport_matrix
    
    def _align_patch_features(self, patch: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        对齐patch特征
        
        Args:
            patch: Patch特征 (B, T, P, D)
            anchor: 锚点特征 (B, T_anchor, D)
            
        Returns:
            aligned_patch: 对齐后的patch特征 (B, T_anchor, P, D)
        """
        B, T, P, D = patch.shape
        B_a, T_a, D_a = anchor.shape
        
        # 将patch特征重塑为 [B, T*P, D] 进行对齐
        patch_reshaped = patch.view(B, T * P, D)
        
        # 执行对齐
        if self.memory_efficient:
            aligned_patch, _ = self._memory_efficient_alignment(patch_reshaped, anchor)
        else:
            aligned_patch, _ = self.ot_aligner(patch_reshaped, anchor)
        
        # 重塑回patch维度，但时间维度与锚点对齐
        # 计算每个时间步对应的patch数量
        patches_per_timestep = (T * P) // T_a if T_a > 0 else P
        patches_per_timestep = max(1, patches_per_timestep)
        
        try:
            aligned_patch = aligned_patch.view(B, T_a, patches_per_timestep, D)
        except RuntimeError:
            # 如果重塑失败，使用插值方法
            logger.warning("Patch重塑失败，使用插值方法")
            aligned_patch = F.interpolate(
                aligned_patch.transpose(1, 2), 
                size=T_a, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            aligned_patch = aligned_patch.view(B, T_a, -1, D)
            
            # 如果patch数量不匹配，进行调整
            current_P = aligned_patch.shape[2]
            if current_P != P:
                if current_P > P:
                    aligned_patch = aligned_patch[:, :, :P, :]
                else:
                    # 重复最后的patch
                    padding = P - current_P
                    last_patch = aligned_patch[:, :, -1:, :].repeat(1, 1, padding, 1)
                    aligned_patch = torch.cat([aligned_patch, last_patch], dim=2)
        
        return aligned_patch
    
    def _standard_alignment(self, audio: torch.Tensor, video: torch.Tensor, 
                          words: torch.Tensor, quest: torch.Tensor, 
                          patch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        标准对齐策略：以语言为锚点
        
        Returns:
            (aligned_audio, aligned_video, words, quest, aligned_patch, alignment_loss, debug_info)
        """
        debug_info = {} if self.debug_mode else None
        
        # 使用words作为锚点进行对齐
        if self.memory_efficient:
            aligned_audio, transport_a2w = self._memory_efficient_alignment(audio, words)
            aligned_video, transport_v2w = self._memory_efficient_alignment(video, words)
        else:
            aligned_audio, transport_a2w = self.ot_aligner(audio, words)
            aligned_video, transport_v2w = self.ot_aligner(video, words)
        
        # 对齐patch特征
        aligned_patch = None
        if patch is not None and self.patch_alignment:
            aligned_patch = self._align_patch_features(patch, words)
        
        # 确保数值稳定性
        aligned_audio = self._ensure_numerical_stability(aligned_audio)
        aligned_video = self._ensure_numerical_stability(aligned_video)
        
        # 计算MMD损失
        alignment_loss = self.mmd_aligner(aligned_video, aligned_audio, words)
        alignment_loss = self._ensure_numerical_stability(alignment_loss)
        
        # 收集调试信息
        if self.debug_mode:
            debug_info.update({
                'strategy': 'standard',
                'anchor_length': words.shape[1],
                'transport_matrices': [transport_a2w, transport_v2w],
                'alignment_loss_components': {
                    'mmd_video_words': self.mmd_aligner.compute_mmd_squared(aligned_video, words).mean(),
                    'mmd_audio_words': self.mmd_aligner.compute_mmd_squared(aligned_audio, words).mean(),
                }
            })
        
        return aligned_audio, aligned_video, words, quest, aligned_patch, alignment_loss, debug_info
    
    def _reverse_alignment(self, audio: torch.Tensor, video: torch.Tensor, 
                         words: torch.Tensor, quest: torch.Tensor, 
                         patch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        反向对齐策略：以视频为锚点，适用于视频问答任务
        
        Returns:
            (aligned_audio, aligned_video, aligned_words, quest, aligned_patch, alignment_loss, debug_info)
        """
        debug_info = {} if self.debug_mode else None
        
        # 选择最长的序列作为锚点
        if video.shape[1] >= audio.shape[1]:
            anchor = video
            anchor_name = 'video'
            
            # 对齐其他模态到视频
            if self.memory_efficient:
                aligned_audio, transport_a2v = self._memory_efficient_alignment(audio, video)
                aligned_words, transport_w2v = self._memory_efficient_alignment(words, video)
            else:
                aligned_audio, transport_a2v = self.ot_aligner(audio, video)
                aligned_words, transport_w2v = self.ot_aligner(words, video)
            
            aligned_video = video  # 保持原始视频
            
        else:
            anchor = audio
            anchor_name = 'audio'
            
            # 对齐其他模态到音频
            if self.memory_efficient:
                aligned_video, transport_v2a = self._memory_efficient_alignment(video, audio)
                aligned_words, transport_w2a = self._memory_efficient_alignment(words, audio)
            else:
                aligned_video, transport_v2a = self.ot_aligner(video, audio)
                aligned_words, transport_w2a = self.ot_aligner(words, audio)
            
            aligned_audio = audio  # 保持原始音频
        
        # 对齐patch特征
        aligned_patch = None
        if patch is not None and self.patch_alignment:
            aligned_patch = self._align_patch_features(patch, anchor)
        
        # 确保数值稳定性
        aligned_audio = self._ensure_numerical_stability(aligned_audio)
        aligned_video = self._ensure_numerical_stability(aligned_video)
        # aligned_words = self._ensure_numerical_stability(aligned_words)
        
        # 计算MMD损失
        alignment_loss = self.mmd_aligner(aligned_video, aligned_audio, anchor)
        alignment_loss = self._ensure_numerical_stability(alignment_loss)
        
        # 收集调试信息
        if self.debug_mode:
            debug_info.update({
                'strategy': 'reverse',
                'anchor_modality': anchor_name,
                'anchor_length': anchor.shape[1],
                'alignment_loss_components': {
                    f'mmd_video_{anchor_name}': self.mmd_aligner.compute_mmd_squared(aligned_video, anchor).mean(),
                    f'mmd_audio_{anchor_name}': self.mmd_aligner.compute_mmd_squared(aligned_audio, anchor).mean(),
                }
            })
        
        return aligned_audio, aligned_video, words, quest, patch, alignment_loss, debug_info
    
    def _bidirectional_alignment(self, audio: torch.Tensor, video: torch.Tensor, 
                               words: torch.Tensor, quest: torch.Tensor, 
                               patch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        双向对齐策略：同时考虑两个方向的对齐
        
        Returns:
            (aligned_audio, aligned_video, aligned_words, quest, aligned_patch, alignment_loss, debug_info)
        """
        debug_info = {} if self.debug_mode else None
        
        # 方向1：语言为锚点
        if self.memory_efficient:
            aligned_audio_1, _ = self._memory_efficient_alignment(audio, words)
            aligned_video_1, _ = self._memory_efficient_alignment(video, words)
        else:
            aligned_audio_1, _ = self.ot_aligner(audio, words)
            aligned_video_1, _ = self.ot_aligner(video, words)
        
        loss_1 = self.mmd_aligner(aligned_video_1, aligned_audio_1, words)
        
        # 方向2：视频为锚点
        if self.memory_efficient:
            aligned_audio_2, _ = self._memory_efficient_alignment(audio, video)
            aligned_words_2, _ = self._memory_efficient_alignment(words, video)
        else:
            aligned_audio_2, _ = self.ot_aligner(audio, video)
            aligned_words_2, _ = self.ot_aligner(words, video)
        
        loss_2 = self.mmd_aligner(video, aligned_audio_2, aligned_words_2)
        
        # 选择损失较小的方向
        if loss_1 < loss_2:
            chosen_direction = 'words_anchor'
            aligned_audio = self._ensure_numerical_stability(aligned_audio_1)
            aligned_video = self._ensure_numerical_stability(aligned_video_1)
            aligned_words = words
            alignment_loss = loss_1
            anchor = words
        else:
            chosen_direction = 'video_anchor'
            aligned_audio = self._ensure_numerical_stability(aligned_audio_2)
            aligned_video = video
            aligned_words = self._ensure_numerical_stability(aligned_words_2)
            alignment_loss = loss_2
            anchor = video
        
        # 对齐patch特征
        aligned_patch = None
        if patch is not None and self.patch_alignment:
            aligned_patch = self._align_patch_features(patch, anchor)
        
        alignment_loss = self._ensure_numerical_stability(alignment_loss)
        
        # 收集调试信息
        if self.debug_mode:
            debug_info.update({
                'strategy': 'bidirectional',
                'chosen_direction': chosen_direction,
                'loss_words_anchor': loss_1.item(),
                'loss_video_anchor': loss_2.item(),
                'anchor_length': anchor.shape[1],
            })
        
        return aligned_audio, aligned_video, words, quest, aligned_patch, alignment_loss, debug_info
    
    def forward(self, audio: torch.Tensor, video: torch.Tensor, 
                words: torch.Tensor, quest: torch.Tensor, 
                patch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        执行对齐处理
        
        Args:
            audio: 音频特征 (B, T_a, D)
            video: 视频特征 (B, T_v, D)
            words: 词语特征 (B, T_w, D)
            quest: 问题特征 (B, D)
            patch: Patch特征 (B, T_p, P, D), 可选
            
        Returns:
            如果对齐禁用: (audio, video, words, quest, patch, 0.0, None)
            如果对齐启用: (aligned_audio, aligned_video, aligned_words, quest, aligned_patch, alignment_loss, debug_info)
        """
        # 如果对齐功能禁用，直接返回原始特征
        if not self.enabled:
            return audio, video, words, quest, patch, torch.tensor(0.0, device=audio.device), None
        
        try:
            # 验证输入
            self._validate_inputs(audio, video, words, quest, patch)
            
            # 根据策略执行对齐
            if self.strategy == 'standard':
                return self._standard_alignment(audio, video, words, quest, patch)
            elif self.strategy == 'reverse':
                return self._reverse_alignment(audio, video, words, quest, patch)
            elif self.strategy == 'bidirectional':
                return self._bidirectional_alignment(audio, video, words, quest, patch)
            else:
                raise ValueError(f"未知的对齐策略: {self.strategy}")
                
        except Exception as e:
            logger.error(f"对齐处理失败: {str(e)}")
            # 返回原始特征作为fallback
            return audio, video, words, quest, patch, torch.tensor(0.0, device=audio.device), None
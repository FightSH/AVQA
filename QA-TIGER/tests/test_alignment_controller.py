"""
AlignmentController的单元测试
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.align_mamba import AlignmentController


class TestAlignmentController(unittest.TestCase):
    
    def setUp(self):
        """测试前的设置"""
        self.batch_size = 2
        self.d_model = 512
        self.T_audio = 60
        self.T_video = 60
        self.T_words = 77
        self.P_patch = 14
        
        # 创建测试数据
        self.audio = torch.randn(self.batch_size, self.T_audio, self.d_model)
        self.video = torch.randn(self.batch_size, self.T_video, self.d_model)
        self.words = torch.randn(self.batch_size, self.T_words, self.d_model)
        self.quest = torch.randn(self.batch_size, self.d_model)
        self.patch = torch.randn(self.batch_size, self.T_video, self.P_patch, self.d_model)
    
    def test_alignment_disabled(self):
        """测试对齐功能禁用时的行为"""
        config = {'enabled': False}
        controller = AlignmentController(config)
        
        result = controller(self.audio, self.video, self.words, self.quest, self.patch)
        
        # 应该返回原始特征
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        
        self.assertTrue(torch.equal(aligned_audio, self.audio))
        self.assertTrue(torch.equal(aligned_video, self.video))
        self.assertTrue(torch.equal(aligned_words, self.words))
        self.assertTrue(torch.equal(quest, self.quest))
        self.assertTrue(torch.equal(aligned_patch, self.patch))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNone(debug_info)
    
    def test_standard_alignment(self):
        """测试标准对齐策略"""
        config = {
            'enabled': True,
            'strategy': 'standard',
            'lambda_align': 0.1,
            'ot_eps': 1e-8,
            'mmd_sigma': 1.0,
            'patch_alignment': True,
            'debug_mode': False,
            'memory_efficient': False
        }
        controller = AlignmentController(config)
        
        result = controller(self.audio, self.video, self.words, self.quest, self.patch)
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        
        # 检查输出形状
        self.assertEqual(aligned_audio.shape, (self.batch_size, self.T_words, self.d_model))
        self.assertEqual(aligned_video.shape, (self.batch_size, self.T_words, self.d_model))
        self.assertEqual(aligned_words.shape, (self.batch_size, self.T_words, self.d_model))
        self.assertEqual(quest.shape, (self.batch_size, self.d_model))
        
        # 检查对齐损失
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)
        
        # words应该保持不变（作为锚点）
        self.assertTrue(torch.equal(aligned_words, self.words))
    
    def test_reverse_alignment(self):
        """测试反向对齐策略"""
        config = {
            'enabled': True,
            'strategy': 'reverse',
            'lambda_align': 0.1,
            'ot_eps': 1e-8,
            'mmd_sigma': 1.0,
            'patch_alignment': True,
            'debug_mode': False,
            'memory_efficient': False
        }
        controller = AlignmentController(config)
        
        result = controller(self.audio, self.video, self.words, self.quest, self.patch)
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        
        # 在这个测试中，video和audio长度相同，所以video会被选为锚点
        expected_length = max(self.T_video, self.T_audio)
        
        self.assertEqual(aligned_audio.shape[1], expected_length)
        self.assertEqual(aligned_video.shape[1], expected_length)
        self.assertEqual(aligned_words.shape[1], expected_length)
        
        # 检查对齐损失
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_bidirectional_alignment(self):
        """测试双向对齐策略"""
        config = {
            'enabled': True,
            'strategy': 'bidirectional',
            'lambda_align': 0.1,
            'ot_eps': 1e-8,
            'mmd_sigma': 1.0,
            'patch_alignment': True,
            'debug_mode': True,  # 启用调试模式
            'memory_efficient': False
        }
        controller = AlignmentController(config)
        
        result = controller(self.audio, self.video, self.words, self.quest, self.patch)
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        
        # 检查调试信息
        self.assertIsNotNone(debug_info)
        self.assertEqual(debug_info['strategy'], 'bidirectional')
        self.assertIn('chosen_direction', debug_info)
        self.assertIn('loss_words_anchor', debug_info)
        self.assertIn('loss_video_anchor', debug_info)
        
        # 检查对齐损失
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效策略
        with self.assertRaises(ValueError):
            config = {'enabled': True, 'strategy': 'invalid_strategy'}
            AlignmentController(config)
        
        # 测试无效lambda_align
        with self.assertRaises(ValueError):
            config = {'enabled': True, 'lambda_align': 1.5}
            AlignmentController(config)
        
        # 测试无效ot_eps
        with self.assertRaises(ValueError):
            config = {'enabled': True, 'ot_eps': -1e-8}
            AlignmentController(config)
        
        # 测试无效mmd_sigma
        with self.assertRaises(ValueError):
            config = {'enabled': True, 'mmd_sigma': -1.0}
            AlignmentController(config)
    
    def test_input_validation(self):
        """测试输入验证"""
        config = {'enabled': True, 'strategy': 'standard'}
        controller = AlignmentController(config)
        
        # 测试维度不匹配
        wrong_audio = torch.randn(self.batch_size, self.T_audio, self.d_model + 1)
        
        result = controller(wrong_audio, self.video, self.words, self.quest, self.patch)
        # 应该返回原始特征（fallback行为）
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        self.assertEqual(loss.item(), 0.0)
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        config = {'enabled': True, 'strategy': 'standard'}
        controller = AlignmentController(config)
        
        # 创建包含极值的测试数据
        extreme_audio = torch.randn(self.batch_size, self.T_audio, self.d_model) * 1e6
        
        result = controller(extreme_audio, self.video, self.words, self.quest, self.patch)
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        
        # 检查结果是否包含NaN或Inf
        self.assertFalse(torch.isnan(aligned_audio).any())
        self.assertFalse(torch.isinf(aligned_audio).any())
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
    
    def test_memory_efficient_mode(self):
        """测试内存高效模式"""
        config = {
            'enabled': True,
            'strategy': 'standard',
            'memory_efficient': True
        }
        controller = AlignmentController(config)
        
        # 创建较大的序列来触发分块处理
        large_audio = torch.randn(self.batch_size, 200, self.d_model)
        large_video = torch.randn(self.batch_size, 200, self.d_model)
        
        result = controller(large_audio, large_video, self.words, self.quest, self.patch)
        aligned_audio, aligned_video, aligned_words, quest, aligned_patch, loss, debug_info = result
        
        # 检查输出形状
        self.assertEqual(aligned_audio.shape, (self.batch_size, self.T_words, self.d_model))
        self.assertEqual(aligned_video.shape, (self.batch_size, self.T_words, self.d_model))
        
        # 检查对齐损失
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
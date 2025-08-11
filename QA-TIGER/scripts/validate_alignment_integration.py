#!/usr/bin/env python3
"""
QA-TIGER 对齐集成验证脚本

该脚本用于验证对齐功能是否正确集成到QA-TIGER模型中。
包括配置验证、模型初始化、前向传播测试等。
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.net import QA_TIGER
    from src.models.align_mamba import (
        AlignmentController, 
        validate_alignment_config,
        get_default_alignment_config
    )
    from configs.qa_tiger.vitl14_alignment_standard import config as standard_config
    from configs.qa_tiger.vitl14_alignment_reverse import config as reverse_config
    from configs.qa_tiger.vitl14_alignment_bidirectional import config as bidirectional_config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)


class AlignmentIntegrationValidator:
    """对齐集成验证器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        
    def create_test_data(self, batch_size=2):
        """创建测试数据"""
        # 模拟QA-TIGER的输入数据格式
        test_data = {
            'quest': torch.randint(0, 1000, (batch_size, 20)),  # 问题token IDs
            'audio': torch.randn(batch_size, 60, 1024),         # 音频特征
            'video': torch.randn(batch_size, 60, 768),          # 视频特征
            'patch': torch.randn(batch_size, 60, 14, 1024),     # Patch特征
        }
        
        # 移动到设备
        for key, value in test_data.items():
            test_data[key] = value.to(self.device)
            
        return test_data
    
    def test_config_validation(self):
        """测试配置验证"""
        print("=" * 50)
        print("测试1: 配置验证")
        print("=" * 50)
        
        try:
            # 测试标准配置
            validate_alignment_config(standard_config)
            print("✓ 标准配置验证通过")
            
            # 测试反向配置
            validate_alignment_config(reverse_config)
            print("✓ 反向配置验证通过")
            
            # 测试双向配置
            validate_alignment_config(bidirectional_config)
            print("✓ 双向配置验证通过")
            
            # 测试无效配置
            invalid_config = {
                'hyper_params': {
                    'model': {
                        'use_alignment': True,
                        'alignment_config': {
                            'enabled': True,
                            'strategy': 'invalid_strategy'  # 无效策略
                        }
                    }
                }
            }
            
            try:
                validate_alignment_config(invalid_config)
                print("✗ 无效配置验证失败 - 应该抛出异常")
                return False
            except ValueError:
                print("✓ 无效配置正确被拒绝")
            
            self.test_results['config_validation'] = True
            return True
            
        except Exception as e:
            print(f"✗ 配置验证失败: {e}")
            self.test_results['config_validation'] = False
            return False
    
    def test_model_initialization(self):
        """测试模型初始化"""
        print("\n" + "=" * 50)
        print("测试2: 模型初始化")
        print("=" * 50)
        
        try:
            # 测试禁用对齐的模型
            model_config_disabled = {
                'd_model': 512,
                'video_dim': 768,
                'patch_dim': 1024,
                'audio_dim': 1024,
                'encoder_type': 'openai/clip-vit-large-patch14',
                'use_alignment': False,
            }
            
            model_disabled = QA_TIGER(**model_config_disabled)
            model_disabled.to(self.device)
            print("✓ 禁用对齐的模型初始化成功")
            
            # 测试启用对齐的模型 - 标准策略
            model_config_standard = {
                'd_model': 512,
                'video_dim': 768,
                'patch_dim': 1024,
                'audio_dim': 1024,
                'encoder_type': 'openai/clip-vit-large-patch14',
                'use_alignment': True,
                'alignment_config': {
                    'enabled': True,
                    'strategy': 'standard',
                    'lambda_align': 0.1,
                    'debug_mode': True,
                }
            }
            
            model_standard = QA_TIGER(**model_config_standard)
            model_standard.to(self.device)
            print("✓ 标准对齐模型初始化成功")
            
            # 测试启用对齐的模型 - 反向策略
            model_config_reverse = model_config_standard.copy()
            model_config_reverse['alignment_config']['strategy'] = 'reverse'
            
            model_reverse = QA_TIGER(**model_config_reverse)
            model_reverse.to(self.device)
            print("✓ 反向对齐模型初始化成功")
            
            # 测试启用对齐的模型 - 双向策略
            model_config_bidirectional = model_config_standard.copy()
            model_config_bidirectional['alignment_config']['strategy'] = 'bidirectional'
            
            model_bidirectional = QA_TIGER(**model_config_bidirectional)
            model_bidirectional.to(self.device)
            print("✓ 双向对齐模型初始化成功")
            
            # 保存模型用于后续测试
            self.models = {
                'disabled': model_disabled,
                'standard': model_standard,
                'reverse': model_reverse,
                'bidirectional': model_bidirectional
            }
            
            self.test_results['model_initialization'] = True
            return True
            
        except Exception as e:
            print(f"✗ 模型初始化失败: {e}")
            self.test_results['model_initialization'] = False
            return False
    
    def test_forward_pass(self):
        """测试前向传播"""
        print("\n" + "=" * 50)
        print("测试3: 前向传播")
        print("=" * 50)
        
        try:
            test_data = self.create_test_data(batch_size=2)
            
            for model_name, model in self.models.items():
                print(f"\n测试 {model_name} 模型:")
                
                model.eval()
                with torch.no_grad():
                    outputs = model(test_data)
                
                # 检查输出格式
                required_keys = ['out', 'fusion_logits', 'alignment_loss']
                for key in required_keys:
                    if key not in outputs:
                        print(f"✗ 缺少输出键: {key}")
                        return False
                
                # 检查输出形状
                batch_size = test_data['quest'].shape[0]
                if outputs['out'].shape != (batch_size, 42):
                    print(f"✗ 输出形状错误: {outputs['out'].shape}, 期望: ({batch_size}, 42)")
                    return False
                
                # 检查对齐损失
                alignment_loss = outputs['alignment_loss']
                if model_name == 'disabled':
                    if alignment_loss.item() != 0.0:
                        print(f"✗ 禁用对齐时损失应为0，实际: {alignment_loss.item()}")
                        return False
                    print("✓ 禁用对齐时损失正确为0")
                else:
                    if alignment_loss.item() < 0:
                        print(f"✗ 对齐损失不应为负数: {alignment_loss.item()}")
                        return False
                    print(f"✓ 对齐损失正常: {alignment_loss.item():.6f}")
                
                # 检查调试信息
                if model_name != 'disabled':
                    debug_info = outputs.get('alignment_debug_info')
                    if debug_info is not None:
                        print(f"✓ 调试信息可用: 策略={debug_info.get('strategy', 'unknown')}")
                    else:
                        print("- 调试信息未启用")
                
                print(f"✓ {model_name} 模型前向传播成功")
            
            self.test_results['forward_pass'] = True
            return True
            
        except Exception as e:
            print(f"✗ 前向传播测试失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['forward_pass'] = False
            return False
    
    def test_backward_pass(self):
        """测试反向传播"""
        print("\n" + "=" * 50)
        print("测试4: 反向传播")
        print("=" * 50)
        
        try:
            test_data = self.create_test_data(batch_size=2)
            
            for model_name, model in self.models.items():
                print(f"\n测试 {model_name} 模型反向传播:")
                
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                
                # 前向传播
                outputs = model(test_data)
                
                # 计算总损失
                task_loss = torch.nn.functional.cross_entropy(
                    outputs['out'], 
                    torch.randint(0, 42, (test_data['quest'].shape[0],)).to(self.device)
                )
                
                if model_name != 'disabled':
                    total_loss = task_loss + 0.1 * outputs['alignment_loss']
                    print(f"  任务损失: {task_loss.item():.6f}")
                    print(f"  对齐损失: {outputs['alignment_loss'].item():.6f}")
                    print(f"  总损失: {total_loss.item():.6f}")
                else:
                    total_loss = task_loss
                    print(f"  任务损失: {task_loss.item():.6f}")
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                
                # 检查梯度
                has_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_grad = True
                        break
                
                if not has_grad:
                    print("✗ 没有计算出梯度")
                    return False
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"✗ 参数 {name} 的梯度包含NaN")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    return False
                
                optimizer.step()
                print(f"✓ {model_name} 模型反向传播成功")
            
            self.test_results['backward_pass'] = True
            return True
            
        except Exception as e:
            print(f"✗ 反向传播测试失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['backward_pass'] = False
            return False
    
    def test_memory_usage(self):
        """测试内存使用"""
        print("\n" + "=" * 50)
        print("测试5: 内存使用")
        print("=" * 50)
        
        if not torch.cuda.is_available():
            print("- 跳过内存测试（CUDA不可用）")
            self.test_results['memory_usage'] = True
            return True
        
        try:
            test_data = self.create_test_data(batch_size=4)  # 更大的batch size
            
            baseline_memory = torch.cuda.memory_allocated()
            
            for model_name, model in self.models.items():
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
                
                model.eval()
                with torch.no_grad():
                    outputs = model(test_data)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_used = (peak_memory - start_memory) / 1024**2  # MB
                
                print(f"{model_name} 模型内存使用: {memory_used:.1f}MB")
                
                # 检查内存使用是否合理（不超过2GB）
                if memory_used > 2048:
                    print(f"✗ {model_name} 模型内存使用过高: {memory_used:.1f}MB")
                    return False
            
            print("✓ 所有模型内存使用正常")
            self.test_results['memory_usage'] = True
            return True
            
        except Exception as e:
            print(f"✗ 内存测试失败: {e}")
            self.test_results['memory_usage'] = False
            return False
    
    def test_different_input_sizes(self):
        """测试不同输入尺寸"""
        print("\n" + "=" * 50)
        print("测试6: 不同输入尺寸")
        print("=" * 50)
        
        try:
            # 测试不同的序列长度
            test_cases = [
                {'audio_len': 30, 'video_len': 30, 'quest_len': 10},
                {'audio_len': 60, 'video_len': 60, 'quest_len': 20},
                {'audio_len': 45, 'video_len': 60, 'quest_len': 15},  # 不同长度
            ]
            
            for i, case in enumerate(test_cases):
                print(f"\n测试用例 {i+1}: audio={case['audio_len']}, video={case['video_len']}, quest={case['quest_len']}")
                
                test_data = {
                    'quest': torch.randint(0, 1000, (2, case['quest_len'])).to(self.device),
                    'audio': torch.randn(2, case['audio_len'], 1024).to(self.device),
                    'video': torch.randn(2, case['video_len'], 768).to(self.device),
                    'patch': torch.randn(2, case['video_len'], 14, 1024).to(self.device),
                }
                
                # 只测试启用对齐的模型
                for model_name in ['standard', 'reverse', 'bidirectional']:
                    model = self.models[model_name]
                    model.eval()
                    
                    with torch.no_grad():
                        outputs = model(test_data)
                    
                    # 检查输出形状
                    if outputs['out'].shape != (2, 42):
                        print(f"✗ {model_name} 输出形状错误: {outputs['out'].shape}")
                        return False
                    
                    # 检查对齐损失
                    if outputs['alignment_loss'].item() < 0:
                        print(f"✗ {model_name} 对齐损失为负: {outputs['alignment_loss'].item()}")
                        return False
                
                print(f"✓ 测试用例 {i+1} 通过")
            
            self.test_results['different_input_sizes'] = True
            return True
            
        except Exception as e:
            print(f"✗ 不同输入尺寸测试失败: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['different_input_sizes'] = False
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始QA-TIGER对齐集成验证...")
        print(f"设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        tests = [
            self.test_config_validation,
            self.test_model_initialization,
            self.test_forward_pass,
            self.test_backward_pass,
            self.test_memory_usage,
            self.test_different_input_sizes,
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            if test():
                passed_tests += 1
            else:
                print(f"\n测试失败，停止后续测试")
                break
        
        # 打印总结
        print("\n" + "=" * 60)
        print("验证结果总结")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
        
        print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！对齐功能集成成功！")
            return True
        else:
            print("❌ 部分测试失败，请检查错误信息并修复问题")
            return False


def main():
    """主函数"""
    validator = AlignmentIntegrationValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("集成验证完成 - 对齐功能可以正常使用")
        print("=" * 60)
        print("\n下一步:")
        print("1. 使用提供的配置文件开始训练")
        print("2. 监控对齐损失和任务性能")
        print("3. 根据需要调整超参数")
        print("\n示例命令:")
        print("python train.py --config configs/qa_tiger/vitl14_alignment_standard.py")
        return 0
    else:
        print("\n" + "=" * 60)
        print("集成验证失败 - 请修复问题后重新运行")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
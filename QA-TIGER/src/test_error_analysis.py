#!/usr/bin/env python3
"""
验证错误分析功能的测试脚本
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.error_analyzer import ErrorAnalyzer, get_answer_from_idx
    print("✓ ErrorAnalyzer 导入成功")
except ImportError as e:
    print(f"✗ ErrorAnalyzer 导入失败: {e}")
    sys.exit(1)

try:
    from src.dataset import AVQA_dataset
    print("✓ AVQA_dataset 导入成功")
except ImportError as e:
    print(f"✗ AVQA_dataset 导入失败: {e}")

# 模拟数据集类
class MockDataset(Dataset):
    def __init__(self):
        self.answer_to_ix = {
            'yes': 0, 'no': 1, 'guitar': 2, 'piano': 3, 'drum': 4,
            'singing': 5, 'dog': 6, 'cat': 7, 'bird': 8, 'car': 9
        }
        self.ix_to_answer = {v: k for k, v in self.answer_to_ix.items()}
        
    def __len__(self):
        return 100
        
    def get_sample_info(self, idx):
        return {
            'sample_id': f'sample_{idx}',
            'video_name': f'video_{idx % 10}.mp4',
            'question': f'What is in this video? (sample {idx})',
            'correct_answer': self.ix_to_answer[idx % len(self.ix_to_answer)],
            'question_type': ['Audio', 'Existential'][idx % 2],
            'modality': ['Audio', 'Visual', 'Audio-Visual'][idx % 3]
        }

def test_error_analyzer():
    """测试错误分析器功能"""
    print("\n开始测试错误分析器功能...")
    
    # 创建错误分析器
    analyzer = ErrorAnalyzer(output_dir='test_errors')
    print("✓ 错误分析器创建成功")
    
    # 创建模拟数据集
    dataset = MockDataset()
    
    # 模拟一些错误
    for i in range(5):
        sample_info = dataset.get_sample_info(i)
        predicted_answer = dataset.ix_to_answer[(i + 1) % len(dataset.ix_to_answer)]
        predicted_logits = torch.randn(len(dataset.answer_to_ix))
        confidence = torch.softmax(predicted_logits, dim=0).max().item()
        
        analyzer.record_error(
            sample_info=sample_info,
            predicted_answer=predicted_answer,
            predicted_logits=predicted_logits,
            confidence=confidence,
            epoch=1,
            batch_idx=i
        )
    
    print(f"✓ 记录了 {len(analyzer.errors)} 个错误样本")
    
    # 保存错误记录
    analyzer.save_errors('test_epoch_1_errors.json')
    print("✓ 错误记录保存成功")
    
    # 获取统计信息
    stats = analyzer.get_error_statistics()
    print("✓ 错误统计计算成功")
    print(f"  - 总错误数: {stats['total_errors']}")
    print(f"  - 问题类型分布: {stats['question_type_distribution']}")
    print(f"  - 模态分布: {stats['modality_distribution']}")
    
    # 清空错误记录
    analyzer.clear_errors()
    print("✓ 错误记录清空成功")
    
    return True

def test_get_answer_function():
    """测试答案索引转换函数"""
    print("\n测试答案索引转换功能...")
    
    answer_to_ix = {'yes': 0, 'no': 1, 'guitar': 2, 'piano': 3}
    
    # 测试正常情况
    answer = get_answer_from_idx(0, answer_to_ix)
    assert answer == 'yes', f"期望 'yes'，得到 '{answer}'"
    print("✓ 正常索引转换测试通过")
    
    # 测试未知索引
    answer = get_answer_from_idx(999, answer_to_ix)
    assert answer == 'unknown', f"期望 'unknown'，得到 '{answer}'"
    print("✓ 未知索引转换测试通过")
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("错误分析功能验证测试")
    print("=" * 60)
    
    try:
        # 测试各个组件
        test_get_answer_function()
        test_error_analyzer()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！错误分析功能正常工作。")
        print("=" * 60)
        
        # 提供使用说明
        print("\n使用说明:")
        print("1. 在训练/测试脚本中导入: from src.error_analyzer import ErrorAnalyzer")
        print("2. 创建分析器: analyzer = ErrorAnalyzer(output_dir='error_logs')")
        print("3. 在训练循环中传递 error_analyzer 参数")
        print("4. 使用 scripts/analyze_errors.py 分析保存的错误文件")
        print("\n详细文档请参阅 ERROR_ANALYSIS_README.md")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

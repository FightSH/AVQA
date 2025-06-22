#!/usr/bin/env python3
"""
快速使用示例：如何使用错误分析功能
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / 'src'))

from error_analyzer import ErrorAnalyzer, get_answer_from_idx, analyze_errors_by_modality

def example_usage():
    """演示如何使用错误分析器"""
    
    # 1. 创建错误分析器
    analyzer = ErrorAnalyzer(save_dir="./example_output", mode="example")
    
    # 2. 模拟记录一些错误
    sample_errors = [
        {
            'sample_info': {
                'index': 0,
                'question_id': '001',
                'video_id': 'video_001',
                'question_content': 'What instrument is being played?',
                'answer': 'piano',
                'question_type': ['Audio', 'Counting'],
                'template_values': ''
            },
            'predicted_answer': 'guitar',
            'predicted_logits': [0.1, 0.8, 0.05, 0.05],
            'confidence': 0.8,
            'epoch': 1,
            'batch_idx': 0
        },
        {
            'sample_info': {
                'index': 1,
                'question_id': '002',
                'video_id': 'video_002',
                'question_content': 'How many people are visible?',
                'answer': '2',
                'question_type': ['Visual', 'Counting'],
                'template_values': ''
            },
            'predicted_answer': '3',
            'predicted_logits': [0.05, 0.1, 0.75, 0.1],
            'confidence': 0.75,
            'epoch': 1,
            'batch_idx': 1
        }
    ]
    
    # 记录错误
    for error in sample_errors:
        analyzer.record_error(
            sample_info=error['sample_info'],
            predicted_answer=error['predicted_answer'],
            predicted_logits=error['predicted_logits'],
            confidence=error['confidence'],
            epoch=error['epoch'],
            batch_idx=error['batch_idx']
        )
    
    # 3. 保存错误记录
    analyzer.save_errors("example_errors.json")
    
    # 4. 生成统计摘要
    summary = analyzer.save_summary("example_summary.json")
    print("错误统计摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 5. 按模态分析错误
    modality_analysis = analyze_errors_by_modality(analyzer.errors)
    print("\\n按模态错误分析:")
    for modality, data in modality_analysis.items():
        print(f"{modality}: {data['count']} 个错误")

def analyze_existing_errors(error_file: str):
    """分析已存在的错误文件"""
    
    if not Path(error_file).exists():
        print(f"错误文件不存在: {error_file}")
        return
    
    # 加载错误数据
    with open(error_file, 'r', encoding='utf-8') as f:
        errors = json.load(f)
    
    print(f"加载了 {len(errors)} 个错误记录")
    
    # 基本统计
    question_types = {}
    confidences = []
    
    for error in errors:
        # 问题类型统计
        qtype = error['sample_info']['question_type']
        if isinstance(qtype, list):
            qtype_str = f"{qtype[0]}/{qtype[1]}" if len(qtype) >= 2 else str(qtype)
        else:
            qtype_str = str(qtype)
        
        question_types[qtype_str] = question_types.get(qtype_str, 0) + 1
        confidences.append(error['confidence'])
    
    # 输出统计信息
    print("\\n=== 错误分析结果 ===")
    print(f"总错误数: {len(errors)}")
    print(f"平均置信度: {sum(confidences)/len(confidences):.3f}")
    print(f"最高置信度: {max(confidences):.3f}")
    print(f"最低置信度: {min(confidences):.3f}")
    
    print("\\n按问题类型错误分布:")
    for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(errors) * 100
        print(f"  {qtype}: {count} ({percentage:.1f}%)")
    
    # 找出高置信度错误
    high_conf_errors = [e for e in errors if e['confidence'] > 0.9]
    print(f"\\n高置信度错误 (>0.9): {len(high_conf_errors)} 个")
    
    # 找出最常见的错误模式
    error_patterns = {}
    for error in errors:
        true_ans = error['sample_info']['answer']
        pred_ans = error['predicted_answer']
        pattern = f"{true_ans} -> {pred_ans}"
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
    
    print("\\n最常见错误模式:")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {pattern}: {count} 次")

if __name__ == "__main__":
    print("=== 错误分析功能使用示例 ===\\n")
    
    if len(sys.argv) > 1:
        # 如果提供了错误文件路径，分析该文件
        error_file = sys.argv[1]
        print(f"分析错误文件: {error_file}")
        analyze_existing_errors(error_file)
    else:
        # 否则运行示例
        print("运行使用示例...")
        example_usage()
        
        print("\\n=== 示例完成 ===")
        print("生成的文件:")
        print("- example_output/example_errors.json")
        print("- example_output/example_summary.json")
        print("\\n要分析实际的错误文件，请运行:")
        print("python example_usage.py path/to/your/error_file.json")

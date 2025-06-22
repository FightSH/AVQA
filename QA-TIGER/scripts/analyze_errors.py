#!/usr/bin/env python3
"""
错误分析脚本 - 用于分析保存的错误记录文件
使用方法: python analyze_errors.py --error_file path/to/error_file.json --output_dir path/to/output
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_error_data(error_file: str) -> List[Dict]:
    """加载错误数据文件"""
    with open(error_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_error_patterns(errors: List[Dict]) -> Dict[str, Any]:
    """分析错误模式"""
    analysis = {}
    
    # 按问题类型分析
    question_types = []
    for error in errors:
        qtype = error['sample_info']['question_type']
        if isinstance(qtype, list):
            qtype_str = f"{qtype[0]}/{qtype[1]}" if len(qtype) >= 2 else str(qtype)
        else:
            qtype_str = str(qtype)
        question_types.append(qtype_str)
    
    type_counts = Counter(question_types)
    analysis['error_by_type'] = dict(type_counts)
    
    # 置信度分析
    confidences = [error['confidence'] for error in errors]
    analysis['confidence_stats'] = {
        'mean': sum(confidences) / len(confidences) if confidences else 0,
        'min': min(confidences) if confidences else 0,
        'max': max(confidences) if confidences else 0,
        'std': pd.Series(confidences).std() if confidences else 0
    }
    
    # 最常见的错误答案组合
    error_pairs = []
    for error in errors:
        true_ans = error['sample_info']['answer']
        pred_ans = error['predicted_answer']
        error_pairs.append(f"{true_ans} -> {pred_ans}")
    
    common_errors = Counter(error_pairs).most_common(20)
    analysis['common_error_pairs'] = common_errors
    
    # 按视频ID分析
    video_errors = Counter([error['sample_info']['video_id'] for error in errors])
    analysis['errors_by_video'] = dict(video_errors.most_common(20))
    
    return analysis

def create_visualizations(errors: List[Dict], output_dir: str):
    """创建可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 错误分布按问题类型
    question_types = []
    for error in errors:
        qtype = error['sample_info']['question_type']
        if isinstance(qtype, list):
            qtype_str = f"{qtype[0]}/{qtype[1]}" if len(qtype) >= 2 else str(qtype)
        else:
            qtype_str = str(qtype)
        question_types.append(qtype_str)
    
    plt.figure(figsize=(12, 6))
    type_counts = Counter(question_types)
    plt.bar(range(len(type_counts)), list(type_counts.values()))
    plt.xticks(range(len(type_counts)), list(type_counts.keys()), rotation=45)
    plt.title('错误分布按问题类型')
    plt.xlabel('问题类型')
    plt.ylabel('错误数量')
    plt.tight_layout()
    plt.savefig(output_path / 'errors_by_question_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 置信度分布
    confidences = [error['confidence'] for error in errors]
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    plt.title('错误预测的置信度分布')
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.axvline(sum(confidences)/len(confidences), color='red', linestyle='--', 
                label=f'平均值: {sum(confidences)/len(confidences):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 错误数量按视频
    video_errors = Counter([error['sample_info']['video_id'] for error in errors])
    top_videos = dict(video_errors.most_common(20))
    
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(top_videos)), list(top_videos.values()))
    plt.xticks(range(len(top_videos)), list(top_videos.keys()), rotation=90)
    plt.title('错误最多的20个视频')
    plt.xlabel('视频ID')
    plt.ylabel('错误数量')
    plt.tight_layout()
    plt.savefig(output_path / 'errors_by_video.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {output_path}")

def generate_report(errors: List[Dict], analysis: Dict[str, Any], output_file: str):
    """生成分析报告"""
    report = []
    report.append("# 错误分析报告\\n")
    
    report.append(f"## 总体统计")
    report.append(f"- 总错误数: {len(errors)}")
    report.append(f"- 平均置信度: {analysis['confidence_stats']['mean']:.3f}")
    report.append(f"- 置信度标准差: {analysis['confidence_stats']['std']:.3f}")
    report.append("")
    
    report.append("## 按问题类型错误分布")
    for qtype, count in analysis['error_by_type'].items():
        percentage = count / len(errors) * 100
        report.append(f"- {qtype}: {count} ({percentage:.1f}%)")
    report.append("")
    
    report.append("## 最常见错误模式 (真实答案 -> 预测答案)")
    for error_pair, count in analysis['common_error_pairs'][:10]:
        report.append(f"- {error_pair}: {count} 次")
    report.append("")
    
    report.append("## 错误最多的视频")
    for video_id, count in list(analysis['errors_by_video'].items())[:10]:
        report.append(f"- {video_id}: {count} 个错误")
    report.append("")
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(report))
    
    print(f"分析报告已保存到: {output_file}")

def export_error_details(errors: List[Dict], output_file: str):
    """导出详细错误信息到CSV"""
    data = []
    for error in errors:
        row = {
            'question_id': error['sample_info'].get('question_id', ''),
            'video_id': error['sample_info']['video_id'],
            'question': error['sample_info']['question_content'],
            'true_answer': error['sample_info']['answer'],
            'predicted_answer': error['predicted_answer'],
            'confidence': error['confidence'],
            'question_type': str(error['sample_info']['question_type']),
            'epoch': error.get('epoch', ''),
            'batch_idx': error.get('batch_idx', ''),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"详细错误信息已导出到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='分析错误记录文件')
    parser.add_argument('--error_file', required=True, help='错误记录JSON文件路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--create_plots', action='store_true', help='是否生成可视化图表')
    
    args = parser.parse_args()
    
    # 加载错误数据
    print(f"加载错误数据: {args.error_file}")
    errors = load_error_data(args.error_file)
    print(f"共加载 {len(errors)} 个错误记录")
    
    # 分析错误模式
    print("分析错误模式...")
    analysis = analyze_error_patterns(errors)
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成报告
    report_file = output_path / 'error_analysis_report.md'
    generate_report(errors, analysis, str(report_file))
    
    # 导出详细信息
    csv_file = output_path / 'error_details.csv'
    export_error_details(errors, str(csv_file))
    
    # 保存分析结果
    analysis_file = output_path / 'error_analysis.json'
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存到: {analysis_file}")
    
    # 生成可视化图表
    if args.create_plots:
        print("生成可视化图表...")
        create_visualizations(errors, str(output_path))
    
    print("\\n分析完成!")

if __name__ == '__main__':
    main()

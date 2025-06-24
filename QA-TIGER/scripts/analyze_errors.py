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
from collections import Counter, defaultdict

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
        qtype = error['question_type']
        if isinstance(qtype, list):
            qtype_str = f"{qtype[0]}/{qtype[1]}" if len(qtype) >= 2 else str(qtype)
        else:
            qtype_str = str(qtype)
        question_types.append(qtype_str)
    
    type_counts = Counter(question_types)
    analysis['error_by_type'] = dict(type_counts)
    
    # 最常见的错误答案组合
    error_pairs = []
    for error in errors:
        true_ans = error['true_answer']
        pred_ans = error['predicted_answer']
        error_pairs.append(f"{true_ans} -> {pred_ans}")

    common_errors = Counter(error_pairs).most_common(20)
    analysis['common_error_pairs'] = common_errors

    # 按视频ID分析
    video_errors = Counter([error['video_name'] for error in errors])
    analysis['errors_by_video'] = dict(video_errors.most_common(20))

    # 按视频和问题的详细错误分析
    video_question_errors = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'true_answers': set(), 'predicted_answers': set()}))
    for error in errors:
        video_name = error['video_name']
        question = error['question_content']
        true_ans = error['true_answer']
        pred_ans = error['predicted_answer']
        
        # 处理问题类型
        qtype = error['question_type']
        if isinstance(qtype, list):
            qtype_str = f"{qtype[0]}/{qtype[1]}" if len(qtype) >= 2 else str(qtype)
        else:
            qtype_str = str(qtype)
        
        video_question_errors[video_name][question]['count'] += 1
        video_question_errors[video_name][question]['true_answers'].add(true_ans)
        video_question_errors[video_name][question]['predicted_answers'].add(pred_ans)
        video_question_errors[video_name][question]['question_type'] = qtype_str
    
    # 转换为普通字典并处理集合为列表
    analysis['errors_by_video_details'] = {}
    for video, questions in video_question_errors.items():
        analysis['errors_by_video_details'][video] = {}
        for question, details in questions.items():
            analysis['errors_by_video_details'][video][question] = {
                'count': details['count'],
                'question_type': details['question_type'],
                'true_answers': list(details['true_answers']),
                'predicted_answers': list(details['predicted_answers'])
            }

    return analysis

def create_visualizations(errors: List[Dict], output_dir: str):
    """创建可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 错误分布按问题类型
    question_types = []
    for error in errors:
        qtype = error['question_type']
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

    # 3. 错误数量按视频
    video_errors = Counter([error['video_name'] for error in errors])
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

    report.append("## 按视频和问题的错误详情")
    for video, questions in analysis['errors_by_video_details'].items():
        report.append(f"### {video}")
        for question, details in questions.items():
            report.append(f"- **问题**: {question}")
            report.append(f"  - 错误次数: {details['count']}")
            report.append(f"  - 正确答案: {', '.join(details['true_answers'])}")
            report.append(f"  - 错误答案: {', '.join(details['predicted_answers'])}")
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
            'question_id': error.get('question_id', ''),
            'video_name': error['video_name'],
            'question': error['question_content'],
            'true_answer': error['true_answer'],
            'predicted_answer': error['predicted_answer'],
            'question_type': str(error['question_type']),
            'predicted_answer_id': error.get('predicted_answer_id', ''),
            'true_answer_id': error.get('true_answer_id', ''),
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
    parser.add_argument('--create_plots', action='store_true', help='是否生成可视化图表')
    
    args = parser.parse_args()
    
    # 加载错误数据
    print(f"加载错误数据: {args.error_file}")
    errors = load_error_data(args.error_file)
    print(f"共加载 {len(errors)} 个错误记录")
    
    # 分析错误模式
    print("分析错误模式...")
    analysis = analyze_error_patterns(errors)
    
    # 获取错误文件所在目录并创建analyze子目录作为输出目录
    error_file_path = Path(args.error_file)
    output_dir = error_file_path.parent / "analyze"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    print("\n分析完成! 结果已保存到错误文件所在目录")

if __name__ == '__main__':
    main()

# python analyze_errors.py --error_file /mnt/sda/shenhao/code/AVQA/QA-TIGER/qa-tiger_clip_vitl14@336px/2025-06-24-17-16-40_seed713/test_errors.json

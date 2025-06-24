"""
错误分析工具 - 用于记录和分析训练/测试过程中的错误预测
"""
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch


class ErrorAnalyzer:
    """错误分析器类，用于记录和保存错误的问题"""
    
    def __init__(self, save_dir: str, mode: str = 'train'):
        """
        初始化错误分析器
        Args:
            save_dir: 保存目录
            mode: 模式 ('train', 'val', 'test')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.errors = []
        self.answer_vocab = self._load_answer_vocab()
        
    def _load_answer_vocab(self) -> Dict[int, str]:
        """从答案词汇表文件加载答案映射"""
        vocab_path = Path(__file__).parent.parent / "data" / "annots" / "music_avqa" / "answer2idx.json"
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ans2ix = data.get('ans2ix', {})
                # 创建反向映射：从索引到答案文本
                return {v: k for k, v in ans2ix.items()}
        except FileNotFoundError:
            print(f"Warning: Answer vocabulary file not found at {vocab_path}")
            return {}
        except Exception as e:
            print(f"Warning: Error loading answer vocabulary: {e}")
            return {}
        
    def _convert_answer_id_to_text(self, answer_id: int) -> str:
        """将答案ID转换为文本"""
        return self.answer_vocab.get(answer_id, f"unknown_answer_{answer_id}")
        
    def record_error(self, 
                    sample_info: Dict,
                    predicted_answer: int,
                    epoch: int = None,
                    batch_idx: int = None):
        """
        记录一个错误的预测
        Args:
            sample_info: 样本信息字典，包含question_type, answer, question_content等
            predicted_answer: 预测的答案ID
            epoch: 训练轮次（可选）
            batch_idx: 批次索引（可选）
        """
        true_answer_id = sample_info.get('answer')
        
        error_record = {
            'question_content': sample_info.get('question_content', sample_info.get('question', 'Unknown question')),
            'question_type': sample_info.get('question_type'),
            'predicted_answer': self._convert_answer_id_to_text(predicted_answer),
            'true_answer': self._convert_answer_id_to_text(true_answer_id),
            'video_name': sample_info.get('name'),
            'predicted_answer_id': predicted_answer,
            'true_answer_id': true_answer_id,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'mode': self.mode
        }
        self.errors.append(error_record)
        
    def save_errors(self, filename: str = None):
        """
        保存错误记录到文件
        Args:
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            filename = f"{self.mode}_errors.json"
            
        filepath = self.save_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.errors, f, ensure_ascii=False, indent=2)
            
        print(f"已保存 {len(self.errors)} 个错误记录到: {filepath}")
        
    def get_error_summary(self) -> Dict[str, Any]:
        """
        获取错误统计摘要
        Returns:
            错误统计信息字典
        """
        if not self.errors:
            return {"total_errors": 0}
            
        # 按问题类型统计
        type_stats = {}
        for error in self.errors:
            qtype = error['question_type']
            if isinstance(qtype, str):
                type_key = qtype
            else:
                # qtype可能是list格式，如['Audio', 'Counting']
                type_key = f"{qtype[0]}/{qtype[1]}" if len(qtype) >= 2 else str(qtype)
                
            if type_key not in type_stats:
                type_stats[type_key] = 0
            type_stats[type_key] += 1
            
        # 按预测答案统计
        predicted_answer_stats = {}
        true_answer_stats = {}
        for error in self.errors:
            pred_ans = error['predicted_answer']
            true_ans = error['true_answer']
            
            predicted_answer_stats[pred_ans] = predicted_answer_stats.get(pred_ans, 0) + 1
            true_answer_stats[true_ans] = true_answer_stats.get(true_ans, 0) + 1
        
        summary = {
            "total_errors": len(self.errors),
            "mode": self.mode,
            "error_by_type": type_stats,
            "most_common_wrong_predictions": dict(sorted(predicted_answer_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "most_common_missed_answers": dict(sorted(true_answer_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        return summary
        
    def save_summary(self, filename: str = None):
        """
        保存错误统计摘要
        Args:
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            filename = f"{self.mode}_error_summary.json"
            
        summary = self.get_error_summary()
        filepath = self.save_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        print(f"已保存错误统计摘要到: {filepath}")
        return summary
        
    def clear_errors(self):
        """清空错误记录"""
        self.errors = []


def create_answer_vocab_from_dataset(dataset) -> Dict[int, str]:
    """
    从数据集创建答案词汇表 (已弃用，现在直接从文件加载)
    Args:
        dataset: AVQA数据集对象
    Returns:
        答案索引到文本的映射字典
    """
    print("Warning: create_answer_vocab_from_dataset is deprecated. Answer vocab is now loaded automatically.")
    return {}


def get_answer_from_idx(answer_idx: int, answer_to_ix: Dict[str, int]) -> str:
    """
    根据索引获取答案文本
    Args:
        answer_idx: 答案索引
        answer_to_ix: 答案到索引的映射字典
    Returns:
        答案文本
    """
    # 创建反向映射
    ix_to_answer = {v: k for k, v in answer_to_ix.items()}
    return ix_to_answer.get(answer_idx, f"unknown_{answer_idx}")


def analyze_errors_by_modality(errors: List[Dict]) -> Dict[str, Any]:
    """
    按模态分析错误
    Args:
        errors: 错误记录列表
    Returns:
        按模态分组的错误分析
    """
    modality_errors = {
        'Audio': [],
        'Visual': [],
        'Audio-Visual': []
    }
    
    for error in errors:
        qtype = error['question_type']
        if isinstance(qtype, list) and len(qtype) >= 1:
            modality = qtype[0]
            if modality in modality_errors:
                modality_errors[modality].append(error)
                
    return {
        modality: {
            'count': len(error_list),
            'errors': error_list
        } for modality, error_list in modality_errors.items()
    }

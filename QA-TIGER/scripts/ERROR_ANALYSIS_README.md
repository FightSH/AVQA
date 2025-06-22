# 错误分析功能使用说明

本项目已集成了错误分析功能，可以记录训练和测试过程中模型回答错误的问题，并提供详细的分析工具。

## 功能概述

1. **错误记录**: 自动记录训练、验证和测试过程中的错误预测
2. **详细信息**: 保存问题内容、真实答案、预测答案、置信度等信息
3. **统计分析**: 按问题类型、模态、视频等维度分析错误模式
4. **可视化**: 生成图表展示错误分布和模式

## 修改的文件

- `src/dataset.py`: 添加了 `get_sample_info()` 方法获取样本详细信息
- `src/error_analyzer.py`: 新增错误分析器类
- `src/trainutils.py`: 修改 `train()`, `evaluate()`, `test()` 函数集成错误记录
- `src/train.py`: 修改主训练脚本集成错误分析器
- `src/test.py`: 修改测试脚本集成错误分析器
- `scripts/analyze_errors.py`: 新增错误分析脚本

## 输出文件说明

训练/测试完成后，会在输出目录中生成以下文件：

### 训练过程
- `train_errors.json`: 训练错误记录（每5个epoch记录一次）
- `val_errors_epoch_{N}.json`: 第N个epoch的验证错误记录
- `val_error_summary_epoch_{N}.json`: 第N个epoch的验证错误统计

### 测试过程
- `test_errors.json`: 测试错误记录
- `test_error_summary.json`: 测试错误统计
- `test_errors_{i}.json`: 多个测试集的错误记录（如果有多个测试集）

### 错误记录格式

每个错误记录包含以下信息：
```json
{
  "sample_info": {
    "index": 123,
    "question_id": "12345",
    "video_id": "video_001",
    "question_content": "What instrument is being played?",
    "answer": "piano",
    "question_type": ["Audio", "Counting"],
    "template_values": "..."
  },
  "predicted_answer": "guitar",
  "true_answer": "piano",
  "confidence": 0.85,
  "predicted_logits": [...],
  "epoch": 10,
  "batch_idx": 15,
  "mode": "val"
}
```

## 使用方法

### 1. 训练时自动记录错误

直接运行训练脚本，错误记录功能会自动启用：

```bash
python src/train.py --config configs/qa_tiger/vitl14.py
```

### 2. 测试时记录错误

运行测试脚本：

```bash
python src/test.py --config configs/qa_tiger/vitl14.py --weight path/to/best.pt
```

### 3. 分析错误记录

使用分析脚本对保存的错误记录进行深入分析：

```bash
python scripts/analyze_errors.py \\
    --error_file path/to/test_errors.json \\
    --output_dir path/to/analysis_output \\
    --create_plots
```

分析脚本会生成：
- `error_analysis_report.md`: 可读的分析报告
- `error_details.csv`: 详细错误信息表格
- `error_analysis.json`: 结构化分析结果
- 可视化图表（如果使用 `--create_plots`）：
  - `errors_by_question_type.png`: 按问题类型的错误分布
  - `confidence_distribution.png`: 错误预测的置信度分布
  - `errors_by_video.png`: 错误最多的视频

## 配置选项

可以通过修改代码来调整错误记录的行为：

### 控制训练时错误记录频率
在 `src/train.py` 中修改：
```python
# 每5个epoch记录一次训练错误
error_analyzer=train_error_analyzer if epoch % 5 == 0 else None

# 改为每个epoch都记录
error_analyzer=train_error_analyzer

# 或者完全不记录训练错误
error_analyzer=None
```

### 控制记录的错误数量
在 `src/error_analyzer.py` 中可以添加限制：
```python
def record_error(self, ...):
    # 只记录前1000个错误
    if len(self.errors) >= 1000:
        return
    # ... 原有代码
```

## 分析示例

### 按问题类型分析错误
```python
from src.error_analyzer import ErrorAnalyzer, analyze_errors_by_modality
import json

# 加载错误数据
with open('test_errors.json', 'r') as f:
    errors = json.load(f)

# 按模态分析
modality_analysis = analyze_errors_by_modality(errors)
for modality, data in modality_analysis.items():
    print(f"{modality}: {data['count']} errors")
```

### 查找特定类型的错误
```python
# 查找音频相关的错误
audio_errors = [
    error for error in errors 
    if error['sample_info']['question_type'][0] == 'Audio'
]

# 查找高置信度但错误的预测
high_conf_errors = [
    error for error in errors 
    if error['confidence'] > 0.9
]
```

## 注意事项

1. **存储空间**: 错误记录会占用一定的存储空间，特别是在大数据集上
2. **性能影响**: 记录错误会轻微影响训练/测试速度
3. **分布式训练**: 在分布式训练中，只有rank 0进程会保存错误记录
4. **内存使用**: 如果错误数量很大，建议定期清空或限制记录数量

## 扩展功能

可以进一步扩展错误分析功能：

1. **实时分析**: 在训练过程中实时分析错误模式
2. **错误可视化**: 创建更丰富的可视化图表
3. **错误聚类**: 使用聚类算法发现错误模式
4. **难例挖掘**: 识别模型持续预测错误的样本
5. **偏差分析**: 分析模型在特定类型问题上的偏差

通过这些错误记录和分析功能，你可以深入了解模型的弱点，指导后续的改进工作。

# AlignMamba Implementation

这是基于论文method.md的AlignMamba框架的完整实现。该框架通过整合局部和全局跨模态对齐机制，来增强Mamba模型在多模态融合任务中的表现。

## 🚀 核心特性

- **严格遵循论文规范**: 每个组件都严格按照method.md中的数学公式实现
- **完整的对齐机制**: 
  - 基于最优传输(OT)的局部跨模态对齐
  - 基于最大均值差异(MMD)的全局跨模态对齐
- **灵活的任务支持**: 支持分类、回归和特征提取任务
- **高质量工程实践**: 完整的类型注解、错误处理和文档
- **自动回退机制**: 当mamba-ssm不可用时自动使用Transformer

## 📁 文件结构

```
src/models/align_mamba/
├── align_mamba_implementation.py  # 主要实现文件
├── example_usage.py              # 使用示例和演示
├── test_align_mamba.py          # 单元测试
├── README.md                    # 本文件
├── method.md                    # 论文方法描述（中文）
└── align.md                     # 论文方法描述（英文）
```

## 🔧 安装依赖

```bash
# 基础依赖
pip install torch torchvision numpy matplotlib

# 可选：Mamba支持（推荐）
pip install mamba-ssm

# 测试依赖
pip install pytest
```

## 📖 快速开始

### 1. 基本使用

```python
import torch
from align_mamba_implementation import create_align_mamba_classifier

# 创建分类模型
model = create_align_mamba_classifier(
    dim_audio=128,      # 音频特征维度
    dim_video=256,      # 视频特征维度  
    dim_language=768,   # 语言特征维度
    num_classes=10,     # 分类类别数
    d_model=256,        # 模型隐藏维度
    n_layers=4,         # Mamba层数
    lambda_align=0.1    # 对齐损失权重
)

# 准备数据
batch_size = 4
audio = torch.randn(batch_size, 50, 128)    # (B, T_a, D_a)
video = torch.randn(batch_size, 30, 256)    # (B, T_v, D_v)
language = torch.randn(batch_size, 20, 768) # (B, T_l, D_l)
labels = torch.randint(0, 10, (batch_size,))

# 前向传播
outputs = model(audio, video, language, labels=labels)
print(f"Logits: {outputs['logits'].shape}")
print(f"Loss: {outputs['loss'].item():.4f}")
```

### 2. 不同任务类型

```python
from align_mamba_implementation import (
    create_align_mamba_classifier,
    create_align_mamba_regressor,
    create_align_mamba_feature_extractor
)

# 分类任务
classifier = create_align_mamba_classifier(
    dim_audio=128, dim_video=256, dim_language=768,
    num_classes=5, d_model=256
)

# 回归任务
regressor = create_align_mamba_regressor(
    dim_audio=128, dim_video=256, dim_language=768,
    output_dim=1, d_model=256
)

# 特征提取
feature_extractor = create_align_mamba_feature_extractor(
    dim_audio=128, dim_video=256, dim_language=768,
    d_model=256
)
```

### 3. 获取对齐信息

```python
# 返回对齐矩阵和中间特征
outputs = model(
    audio, video, language,
    return_alignments=True,
    return_features=True
)

# 传输矩阵
transport_matrices = outputs["transport_matrices"]
Ma2l = transport_matrices["audio_to_lang"]  # 音频到语言的对齐
Mv2l = transport_matrices["video_to_lang"]  # 视频到语言的对齐

# 对齐后的特征
aligned_features = outputs["aligned_features"]
aligned_audio = aligned_features["audio"]
aligned_video = aligned_features["video"]

# 融合后的特征
fused_features = outputs["features"]  # (B, 3*T_l, d_model)
```

## 🧪 运行测试

```bash
# 运行所有测试
python test_align_mamba.py

# 或使用pytest
pytest test_align_mamba.py -v
```

## 📊 运行示例

```bash
# 运行完整的演示示例
python example_usage.py
```

这将运行：
- 多模态分类任务演示
- 多模态回归任务演示  
- 特征提取演示
- 对齐矩阵可视化
- 训练历史绘制

## 🏗️ 架构详解

### 1. 核心组件

#### OptimalTransportAlignment
实现论文3.2节的松弛版最优传输算法：

```python
# 公式(3): 余弦距离成本矩阵
C_{v2l}(i,j) = 1 - cos(X_v^i, X_l^j)

# 公式(5): 松弛版OT解
M_{v2l}(i,j) = 1/T_v if j = argmin_j' C_{v2l}(i,j'), else 0

# 公式(6): 特征对齐
X̃_v = M_{v2l}^T @ X_v
```

#### MMDGlobalAlignment
实现论文3.3节的MMD全局对齐：

```python
# 公式(9): 高斯核
k(x,y) = exp(-||x-y||²/(2σ²))

# 公式(8): MMD²计算
MMD²(X,Y) = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]

# 公式(10): 全局对齐损失
L_align = MMD²(X̃_v, X_l) + MMD²(X̃_a, X_l)
```

#### MambaBackbone
支持自动回退的Mamba骨干网络：
- 优先使用`mamba-ssm`实现
- 不可用时回退到`TransformerEncoder`

### 2. 数据流

```
原始输入 → 单模态投影 → 局部对齐(OT) → 全局对齐(MMD) → 时间交错 → Mamba融合 → 任务输出
   ↓           ↓            ↓            ↓           ↓          ↓         ↓
 (B,T,D)   (B,T,d_model)  (B,T_l,d_model)  损失计算   (B,3*T_l,d_model)  池化    最终预测
```

### 3. 损失函数

```python
# 公式(12): 总损失
L = L_task + λ * L_align

其中:
- L_task: 任务特定损失（分类用交叉熵，回归用MSE）
- L_align: MMD全局对齐损失
- λ: 平衡超参数
```

## ⚙️ 配置参数

### 模型参数
- `dim_audio/video/language`: 各模态输入维度
- `d_model`: 统一的模型隐藏维度
- `n_layers`: Mamba/Transformer层数
- `dropout`: Dropout比例

### 对齐参数
- `sigma`: MMD高斯核带宽
- `lambda_align`: 对齐损失权重
- `ot_eps`: OT计算的数值稳定性参数

### 任务参数
- `task_type`: "classification", "regression", "feature_extraction"
- `num_classes`: 分类类别数或回归输出维度
- `pooling`: 序列池化方式 ("mean", "max", "last")

## 🎯 性能优化建议

### 1. 内存优化
```python
# 对于长序列，考虑使用梯度检查点
model = create_align_mamba_classifier(...)
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
```

### 2. 训练技巧
```python
# 使用梯度裁剪防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### 3. 超参数调优
- `lambda_align`: 通常在0.01-0.5之间，根据任务调整
- `sigma`: MMD核带宽，影响对齐的敏感度
- `d_model`: 更大的模型容量，但计算成本更高

## 🐛 常见问题

### Q1: 出现"mamba-ssm not found"错误
**A**: 这是正常的，模型会自动回退到Transformer实现。如需使用Mamba，请安装：
```bash
pip install mamba-ssm
```

### Q2: 训练时损失不收敛
**A**: 尝试调整以下参数：
- 降低学习率
- 调整`lambda_align`权重
- 检查数据预处理和归一化

### Q3: 内存不足
**A**: 
- 减小`batch_size`
- 减小`d_model`或`n_layers`
- 使用梯度累积
- 启用混合精度训练

### Q4: 不同模态序列长度差异很大
**A**: 这是正常的，OT对齐会自动处理不同长度的序列。最终都会对齐到语言模态的长度。

## 📚 引用

如果使用此实现，请引用原论文：

```bibtex
@article{alignmamba2024,
  title={AlignMamba: Integrating Local and Global Cross-modal Alignment for Multimodal Fusion},
  author={...},
  journal={...},
  year={2024}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个实现！

## 📄 许可证

本项目遵循MIT许可证。
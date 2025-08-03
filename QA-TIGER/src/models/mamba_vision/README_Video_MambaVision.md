# Video MambaVision: 多模态视频理解模型

基于 MambaVision 架构改造的多模态视频理解模型，专门用于处理图像序列和音频序列的联合建模。

## 🚀 主要特性

- **多模态融合**: 同时处理视觉和听觉信息
- **高效序列建模**: 利用 Mamba 的线性复杂度处理长视频序列
- **灵活架构**: 支持因果和非因果模式，适应不同应用场景
- **可扩展设计**: 提供多种模型规模，从轻量级到大型模型

## 📋 模型架构

### 核心组件

1. **多模态预处理器 (MultiModalPreprocessor)**
   - 将图像和音频特征投影到统一空间
   - 添加模态特定的嵌入
   - 特征归一化和融合

2. **时序位置编码 (TemporalPositionalEncoding)**
   - 为序列添加时间位置信息
   - 支持任意长度的序列

3. **混合架构**
   - 前期阶段: 1D 卷积处理局部时序模式
   - 后期阶段: Mamba + Attention 处理全局依赖

4. **视频 Mamba 混合器 (VideoMambaVisionMixer)**
   - 基于状态空间模型的高效序列建模
   - 线性复杂度，适合长序列处理

### 架构流程

```
输入: 图像特征序列 + 音频特征序列
  ↓
多模态预处理 (投影 + 融合)
  ↓
时序位置编码
  ↓
阶段1-2: 时序卷积块 (局部模式)
  ↓
阶段3-4: Mamba + Attention (全局依赖)
  ↓
全局池化 + 分类头
  ↓
输出: 分类结果
```

## 🛠️ 安装要求

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install timm einops

# Mamba 依赖
pip install mamba-ssm

# 可选：用于可视化和分析
pip install matplotlib seaborn tensorboard
```

## 🎯 问题向量融合 (新功能)

针对视频问答、视频检索等需要问题引导的任务，我们提供了问题感知的视频理解模型。

### 四种融合策略

1. **早期拼接 (Early Concatenation)** - 简单直接，计算高效
2. **交叉注意力 (Cross Attention)** - 动态关注，平衡性能
3. **问题引导 (Question Guided)** - 模态选择，适合特定场景  
4. **自适应融合 (Adaptive Fusion)** - 完全自适应，最高潜力

### 快速使用

```python
from models.question_aware_video_mamba import question_aware_video_mamba_base

# 创建问题感知模型
model = question_aware_video_mamba_base(
    img_dim=2048,           # 图像特征维度
    audio_dim=128,          # 音频特征维度
    question_dim=512,       # 问题向量维度
    num_classes=1000,
    fusion_strategy='cross_attention'  # 推荐策略
)

# 输入数据
img_seq = torch.randn(4, 30, 2048)      # 图像序列
audio_seq = torch.randn(4, 30, 128)     # 音频序列
question_vec = torch.randn(4, 512)      # 问题向量

# 前向传播
output = model(img_seq, audio_seq, question_vec)  # (4, 1000)
```

### 策略选择指南

```python
# 视频问答任务 - 推荐交叉注意力
model = question_aware_video_mamba_base(
    fusion_strategy='cross_attention',
    img_dim=2048, audio_dim=128, question_dim=512
)

# 实时应用 - 推荐早期拼接
model = question_aware_video_mamba_small(
    fusion_strategy='early_concat',
    img_dim=1024, audio_dim=64, question_dim=256
)

# 复杂任务 - 推荐自适应融合
model = question_aware_video_mamba_base(
    fusion_strategy='adaptive_fusion',
    img_dim=2048, audio_dim=128, question_dim=768
)
```

## 📖 使用方法

### 基础使用

```python
import torch
from models.video_mamba_vision import video_mamba_vision_base

# 创建模型
model = video_mamba_vision_base(
    img_dim=2048,      # 图像特征维度 (如 ResNet-50 输出)
    audio_dim=128,     # 音频特征维度
    num_classes=400,   # 分类数量
    causal=False       # 非因果模式
)

# 准备数据
batch_size, seq_len = 4, 30
img_features = torch.randn(batch_size, seq_len, 2048)  # 图像特征序列
audio_features = torch.randn(batch_size, seq_len, 128)  # 音频特征序列

# 前向传播
output = model(img_features, audio_features)
print(f"输出形状: {output.shape}")  # (4, 400)
```

### 使用配置文件

```python
from configs.video_config import get_config
from models.video_mamba_vision import VideoMambaVision

# 获取 Kinetics-400 配置
config = get_config('kinetics400')

# 创建模型
model = VideoMambaVision(
    img_dim=config['data']['img_dim'],
    audio_dim=config['data']['audio_dim'],
    num_classes=config['model']['num_classes'],
    **config['model']
)
```

### 训练示例

```python
from examples.video_understanding_example import VideoTrainer, VideoDataset
from torch.utils.data import DataLoader

# 创建数据集
train_dataset = VideoDataset(num_samples=1000, seq_len=30)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 创建训练器
trainer = VideoTrainer(model, device='cuda', learning_rate=1e-4)

# 开始训练
trainer.train(train_loader, val_loader, epochs=10)
```

## 🎯 应用场景

### 1. 动作识别
```python
# Kinetics-400 动作识别
model = video_mamba_vision_base(
    img_dim=2048,
    audio_dim=128,
    num_classes=400,
    causal=False  # 可以看到未来帧
)
```

### 2. 实时视频理解
```python
# 实时应用，因果模型
model = video_mamba_vision_small(
    img_dim=1024,
    audio_dim=64,
    num_classes=101,
    causal=True,  # 只能看到过去和当前帧
    hidden_dim=128  # 更小的模型以提高速度
)
```

### 3. 情感分析
```python
# 多模态情感分析
model = video_mamba_vision_base(
    img_dim=512,   # 面部特征
    audio_dim=256, # 语音特征
    num_classes=7, # 7种基本情感
)
```

### 4. 长视频理解
```python
# 处理长视频序列
model = video_mamba_vision_base(
    img_dim=1024,
    audio_dim=128,
    num_classes=1000,
    hidden_dim=384,
    depths=[3, 3, 12, 3]  # 更深的网络
)

# 长序列输入
long_img_seq = torch.randn(2, 100, 1024)    # 100帧
long_audio_seq = torch.randn(2, 100, 128)
output = model(long_img_seq, long_audio_seq)
```

## ⚙️ 模型变体

| 模型 | 隐藏维度 | 深度 | 参数量 | 适用场景 |
|------|----------|------|--------|----------|
| Tiny | 128 | [1,2,4,2] | ~5M | 移动端/实时应用 |
| Small | 192 | [2,2,6,2] | ~15M | 一般视频理解 |
| Base | 256 | [2,2,8,2] | ~35M | 高精度应用 |
| Large | 384 | [3,3,12,3] | ~80M | 复杂长视频 |

## 🔧 高级配置

### 因果 vs 非因果模式

```python
# 非因果模式 - 适用于离线视频分析
model = VideoMambaVision(causal=False)  # 可以看到未来帧

# 因果模式 - 适用于实时应用
model = VideoMambaVision(causal=True)   # 只能看到过去帧
```

### 自定义特征维度平衡

```python
# 当图像和音频特征维度差异很大时
model = VideoMambaVision(
    img_dim=2048,    # 大的图像特征
    audio_dim=64,    # 小的音频特征
    hidden_dim=256,  # 统一投影维度
)
```

### 梯度累积训练

```python
# 处理大模型或长序列时的内存优化
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📊 性能对比

### 计算复杂度

| 序列长度 | Transformer | Video MambaVision | 加速比 |
|----------|-------------|-------------------|--------|
| 30 帧    | O(900)      | O(30)            | 30x    |
| 60 帧    | O(3600)     | O(60)            | 60x    |
| 100 帧   | O(10000)    | O(100)           | 100x   |

### 内存使用

```python
# 测试不同序列长度的内存使用
seq_lengths = [16, 32, 64, 128]
for seq_len in seq_lengths:
    model = video_mamba_vision_base()
    img_seq = torch.randn(1, seq_len, 2048)
    audio_seq = torch.randn(1, seq_len, 128)
    
    # 测量内存使用
    torch.cuda.reset_peak_memory_stats()
    output = model(img_seq, audio_seq)
    memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"序列长度 {seq_len}: {memory_used:.1f} MB")
```

## 🐛 常见问题

### Q1: 如何处理不同长度的视频序列？
```python
# 方法1: 填充到固定长度
def pad_sequence(seq, target_len):
    if seq.shape[1] < target_len:
        pad_len = target_len - seq.shape[1]
        padding = torch.zeros(seq.shape[0], pad_len, seq.shape[2])
        seq = torch.cat([seq, padding], dim=1)
    return seq[:, :target_len]

# 方法2: 使用掩码
def create_mask(seq_lens, max_len):
    mask = torch.arange(max_len)[None, :] < seq_lens[:, None]
    return mask
```

### Q2: 如何处理缺失的音频或图像特征？
```python
# 使用零填充或学习的默认特征
class RobustVideoMambaVision(VideoMambaVision):
    def forward(self, img_seq, audio_seq=None):
        if audio_seq is None:
            # 使用学习的默认音频特征
            audio_seq = self.default_audio_features.expand(
                img_seq.shape[0], img_seq.shape[1], -1
            )
        return super().forward(img_seq, audio_seq)
```

### Q3: 如何微调预训练模型？
```python
# 加载预训练权重
model = video_mamba_vision_base(num_classes=1000)  # 原始类别数
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint, strict=False)

# 替换分类头
model.head = nn.Linear(model.head.in_features, new_num_classes)

# 冻结特征提取器，只训练分类头
for param in model.parameters():
    param.requires_grad = False
model.head.weight.requires_grad = True
model.head.bias.requires_grad = True
```

## 📚 引用

如果您在研究中使用了这个模型，请引用：

```bibtex
@article{video_mambavision2024,
  title={Video MambaVision: Efficient Multi-modal Video Understanding with State Space Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
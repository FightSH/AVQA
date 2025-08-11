# QA-TIGER 对齐功能故障排除指南

## 概述

本指南提供了QA-TIGER跨模态对齐功能的详细故障排除方法，包括常见问题的诊断和解决方案。

## 诊断工具

### 1. 启用调试模式

```python
alignment_config = dict(
    debug_mode=True,  # 启用详细调试信息
    # 其他配置...
)

# 检查调试输出
outputs = model(inputs)
debug_info = outputs['alignment_debug_info']
if debug_info:
    print(f"对齐策略: {debug_info['strategy']}")
    print(f"锚点长度: {debug_info['anchor_length']}")
    if 'alignment_loss_components' in debug_info:
        for key, value in debug_info['alignment_loss_components'].items():
            print(f"{key}: {value}")
```

### 2. 使用对齐监控器

```python
from src.models.align_mamba.alignment_monitor import alignment_monitor, log_alignment_step

# 在训练循环中记录对齐信息
log_alignment_step(debug_info, alignment_loss, step=current_step)

# 生成分析报告
report = alignment_monitor.generate_alignment_report()
print(report)

# 绘制趋势图
alignment_monitor.plot_alignment_trends()
```

### 3. 性能分析

```python
from src.models.align_mamba.alignment_monitor import performance_profiler

# 分析性能瓶颈
summary = performance_profiler.get_performance_summary()
performance_profiler.print_performance_report()
```

## 常见问题及解决方案

### 1. 配置相关问题

#### 问题：ImportError: cannot import name 'AlignmentController'

**症状**：
```
ImportError: cannot import name 'AlignmentController' from 'src.models.align_mamba'
```

**原因**：
- 模块路径错误
- 文件未正确创建

**解决方案**：
```python
# 检查文件是否存在
import os
print(os.path.exists('src/models/align_mamba/alignment_controller.py'))

# 检查__init__.py文件
print(os.path.exists('src/models/align_mamba/__init__.py'))

# 重新安装或检查Python路径
import sys
sys.path.append('.')
```

#### 问题：ValueError: 无效的对齐策略

**症状**：
```
ValueError: 无效的对齐策略: standrd. 有效选项: ['standard', 'reverse', 'bidirectional']
```

**原因**：
- 配置文件中strategy参数拼写错误

**解决方案**：
```python
# 正确的配置
alignment_config = dict(
    strategy='standard',  # 注意拼写
    # 不是 'standrd' 或其他拼写错误
)

# 验证配置
from src.models.align_mamba import validate_alignment_config
validate_alignment_config(config)
```

#### 问题：ValueError: lambda_align必须在[0, 1]范围内

**症状**：
```
ValueError: lambda_align必须在[0, 1]范围内，当前值: 1.5
```

**解决方案**：
```python
# 调整lambda_align到合理范围
alignment_config = dict(
    lambda_align=0.1,  # 推荐值：0.05-0.2
)
```

### 2. 训练相关问题

#### 问题：对齐损失不收敛或异常高

**症状**：
- 对齐损失持续上升
- 对齐损失值异常大（>10）
- 对齐损失为NaN

**诊断步骤**：
```python
# 1. 检查输入数据
print(f"Audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")
print(f"Video shape: {video.shape}, range: [{video.min():.3f}, {video.max():.3f}]")
print(f"Words shape: {words.shape}, range: [{words.min():.3f}, {words.max():.3f}]")

# 2. 检查特征维度一致性
assert audio.shape[-1] == video.shape[-1] == words.shape[-1], "特征维度不一致"

# 3. 检查数值稳定性
print(f"Audio contains NaN: {torch.isnan(audio).any()}")
print(f"Video contains NaN: {torch.isnan(video).any()}")
print(f"Words contains NaN: {torch.isnan(words).any()}")
```

**解决方案**：
```python
# 方案1: 调整超参数
alignment_config = dict(
    lambda_align=0.05,    # 降低对齐损失权重
    ot_eps=1e-6,          # 增加数值稳定性
    mmd_sigma=0.5,        # 减小MMD核参数
)

# 方案2: 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 方案3: 使用更稳定的优化器设置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,              # 降低学习率
    weight_decay=1e-3,    # 增加正则化
    eps=1e-8
)
```

#### 问题：训练速度显著下降

**症状**：
- 启用对齐后训练速度下降50%以上
- GPU利用率低

**诊断步骤**：
```python
# 测量各组件耗时
import time

start_time = time.time()
outputs = model(inputs)
total_time = time.time() - start_time

print(f"总前向传播时间: {total_time:.3f}s")

# 检查内存使用
if torch.cuda.is_available():
    print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
```

**解决方案**：
```python
# 方案1: 启用内存优化
alignment_config = dict(
    memory_efficient=True,
    debug_mode=False,     # 关闭调试模式
)

# 方案2: 使用更高效的策略
alignment_config = dict(
    strategy='standard',  # 标准策略比双向策略更快
)

# 方案3: 减小batch size
data_config = dict(
    batch_size=16,        # 从32减少到16
    eval_batch_size=16,
)

# 方案4: 使用梯度累积
accumulation_steps = 2
effective_batch_size = batch_size * accumulation_steps
```

### 3. 内存相关问题

#### 问题：CUDA out of memory

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**：
```python
# 方案1: 启用内存优化模式
alignment_config = dict(
    memory_efficient=True,
    patch_alignment=False,  # 如果不需要可以关闭
)

# 方案2: 减小batch size
batch_size = 8  # 大幅减小

# 方案3: 使用梯度检查点
model.gradient_checkpointing = True

# 方案4: 清理GPU缓存
torch.cuda.empty_cache()

# 方案5: 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 问题：内存泄漏

**症状**：
- 内存使用持续增长
- 长时间训练后内存耗尽

**诊断步骤**：
```python
import gc
import torch

# 监控内存使用
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
    print(f"Python对象数量: {len(gc.get_objects())}")

# 在训练循环中定期调用
if step % 100 == 0:
    print_memory_usage()
```

**解决方案**：
```python
# 方案1: 及时释放中间变量
def training_step(model, inputs):
    outputs = model(inputs)
    loss = outputs['out']
    alignment_loss = outputs['alignment_loss']
    
    # 不要保存不必要的中间结果
    del outputs['alignment_debug_info']  # 如果不需要
    
    return loss, alignment_loss

# 方案2: 定期清理缓存
if step % 100 == 0:
    torch.cuda.empty_cache()
    gc.collect()

# 方案3: 使用上下文管理器
with torch.no_grad():
    # 推理代码
    pass
```

### 4. 数值稳定性问题

#### 问题：出现NaN或Inf值

**症状**：
```
RuntimeWarning: invalid value encountered in divide
RuntimeWarning: overflow encountered in exp
```

**诊断步骤**：
```python
# 检查输入数据范围
def check_tensor_health(tensor, name):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"  Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
    print(f"  NaN count: {torch.isnan(tensor).sum()}")
    print(f"  Inf count: {torch.isinf(tensor).sum()}")

check_tensor_health(audio, "Audio")
check_tensor_health(video, "Video")
check_tensor_health(words, "Words")
```

**解决方案**：
```python
# 方案1: 增加数值稳定性参数
alignment_config = dict(
    ot_eps=1e-6,          # 增加epsilon值
    mmd_sigma=1.0,        # 避免过小的sigma
)

# 方案2: 添加输入归一化
def normalize_features(features):
    # L2归一化
    features = F.normalize(features, p=2, dim=-1)
    # 或者标准化
    # features = (features - features.mean(dim=-1, keepdim=True)) / (features.std(dim=-1, keepdim=True) + 1e-8)
    return features

audio = normalize_features(audio)
video = normalize_features(video)
words = normalize_features(words)

# 方案3: 使用更稳定的激活函数
# 在投影层中使用GELU而不是ReLU
projection = nn.Sequential(
    nn.Linear(input_dim, output_dim),
    nn.GELU(),  # 更稳定
    nn.LayerNorm(output_dim)
)
```

### 5. 性能优化问题

#### 问题：推理速度慢

**解决方案**：
```python
# 方案1: 推理时关闭不必要的功能
model.eval()
alignment_config = dict(
    debug_mode=False,
    memory_efficient=True,
)

# 方案2: 使用torch.jit编译
model = torch.jit.script(model)

# 方案3: 批量推理
def batch_inference(model, data_loader):
    results = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            results.append(outputs['out'])
    return torch.cat(results, dim=0)
```

## 调试检查清单

### 启动前检查

- [ ] 配置文件语法正确
- [ ] 所有必需的模块已安装
- [ ] GPU内存充足
- [ ] 数据路径正确

### 训练过程检查

- [ ] 对齐损失在合理范围内（0.01-1.0）
- [ ] 任务损失正常下降
- [ ] 内存使用稳定
- [ ] 没有NaN或Inf值

### 性能检查

- [ ] 训练速度可接受（相比基线<50%下降）
- [ ] GPU利用率正常（>80%）
- [ ] 内存使用效率高

## 日志分析

### 正常日志示例

```
INFO - 对齐配置验证通过
INFO - AlignmentController初始化成功
INFO - 策略: standard, 锚点长度: 77
INFO - 对齐损失: 0.156, MMD_video_words: 0.089, MMD_audio_words: 0.067
INFO - 训练步骤 100/1000, 损失: 2.345, 对齐损失: 0.156
```

### 异常日志示例

```
WARNING - 检测到NaN值，使用零张量替换
ERROR - 对齐处理失败，继续使用原始特征: 输入张量具有不兼容的形状
WARNING - 对齐损失异常高: 15.67，请检查配置
```

## 联系支持

如果以上解决方案都无法解决问题，请提供以下信息：

1. 完整的错误堆栈跟踪
2. 配置文件内容
3. 输入数据的形状和范围
4. 硬件环境信息（GPU型号、内存大小）
5. PyTorch版本信息

```python
# 收集环境信息
import torch
import sys

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

通过系统化的诊断和解决方案，大多数对齐相关问题都可以得到有效解决。
# VideoMamba聚合器使用指南

## 概述

VideoMamba聚合器是对QA-TIGER中传统TempMoE聚合器的改进，使用VideoMambaVision的强大时序建模能力来替代混合专家网络，提供更好的多模态时序特征聚合效果。

## 核心优势

### 1. 更强的时序建模能力
- **Mamba状态空间模型**: 相比MLP专家网络，更适合长序列建模
- **层次化处理**: 多stage设计，从局部到全局逐步抽象
- **注意力机制**: 结合Mamba和Attention的优势

### 2. 简化的架构设计
- **避免专家选择复杂性**: 不需要路由网络和高斯权重计算
- **端到端优化**: 整个聚合过程可以端到端训练
- **统一的技术栈**: 与特征增强部分使用相同的VideoMamba技术

### 3. 灵活的问题融合方式
- **concat**: 拼接问题特征到时序特征
- **cross_attn**: 使用交叉注意力融合问题信息
- **add**: 直接将问题特征加到每个时间步

## 使用方法

### 1. 基本配置

```python
# 在配置文件中启用VideoMamba聚合器
model=dict(
    d_model=512,
    # ... 其他参数
    use_mamba_aggregator=True,  # 启用VideoMamba聚合器
    mamba_aggregator_config=dict(
        mamba_hidden_dim=256,
        depths=[1, 1, 2, 1],  # 聚合任务使用较小配置
        num_heads=[4, 8, 16, 32],
        question_fusion='concat',  # 问题融合方式
        dropout=0.1,
    ),
)
```

### 2. 不同聚合器类型

#### 单输出聚合器 (用于音频聚合)
```python
aggregator = create_mamba_aggregator(
    aggregator_type="single",
    d_model=512,
    mamba_config={
        'question_fusion': 'concat',
        'mamba_hidden_dim': 256,
    }
)

# 使用
a_global = aggregator(question, audio=audio_features)
```

#### 双输出聚合器 (用于视频+patch聚合)
```python
aggregator = create_mamba_aggregator(
    aggregator_type="dual",
    d_model=512,
    mamba_config={
        'question_fusion': 'cross_attn',
        'mamba_hidden_dim': 256,
    }
)

# 使用
ap_global, vp_global = aggregator(question, video_features, patch_features)
```

#### 统一聚合器 (处理所有模态)
```python
aggregator = create_mamba_aggregator(
    aggregator_type="unified",
    d_model=512,
    mamba_config={
        'question_fusion': 'add',
        'mamba_hidden_dim': 256,
    }
)

# 使用
a_global, v_global, p_global = aggregator(question, audio, video, patch)
```

### 3. 问题融合方式选择

#### concat (推荐)
- **优点**: 保留完整的问题信息，融合效果好
- **缺点**: 增加输入维度，计算量稍大
- **适用**: 大多数场景

#### cross_attn
- **优点**: 动态融合，注意力机制灵活
- **缺点**: 计算复杂度高
- **适用**: 问题与时序特征关联性强的场景

#### add
- **优点**: 计算简单，参数少
- **缺点**: 可能丢失问题信息的细节
- **适用**: 资源受限或问题特征相对简单的场景

## 配置参数详解

### 核心参数
- `mamba_hidden_dim`: VideoMamba内部隐藏维度，建议为d_model的一半
- `depths`: 各stage的层数，聚合任务建议使用较小值如[1,1,2,1]
- `num_heads`: 各stage的注意力头数
- `question_fusion`: 问题融合方式，'concat'/'cross_attn'/'add'
- `dropout`: Dropout率

### 性能调优参数
- `drop_path_rate`: DropPath率，防止过拟合
- `layer_scale`: Layer Scale参数，稳定训练
- `causal`: 是否使用因果注意力，聚合任务通常设为False

## 性能对比

### 参数量对比
- **传统TempMoE**: ~2M参数 (10个专家 × 512维MLP)
- **VideoMamba聚合器**: ~1.5M参数 (更紧凑的设计)

### 计算效率
- **TempMoE**: 需要动态专家选择和高斯权重计算
- **VideoMamba**: 固定计算图，更适合并行化

### 建模能力
- **TempMoE**: 基于MLP的局部建模
- **VideoMamba**: 状态空间模型的全局时序建模

## 训练建议

### 1. 渐进式训练
```python
# 第一阶段：只启用音频聚合器
use_mamba_aggregator=True
mamba_aggregator_config={'question_fusion': 'concat'}

# 第二阶段：启用所有聚合器
# 在第一阶段基础上继续训练
```

### 2. 学习率设置
```python
# VideoMamba聚合器可能需要较小的学习率
optim=dict(
    lr=1e-4,  # 比传统TempMoE稍小
    weight_decay=1e-2,
)
```

### 3. 正则化
```python
mamba_aggregator_config=dict(
    dropout=0.1,
    drop_path_rate=0.05,  # 防止过拟合
)
```

## 故障排除

### 1. 内存不足
- 减少`mamba_hidden_dim`
- 使用更小的`depths`配置
- 选择'add'融合方式

### 2. 训练不稳定
- 增加`dropout`
- 使用`layer_scale`
- 降低学习率

### 3. 性能不如预期
- 尝试不同的`question_fusion`方式
- 调整`depths`配置
- 检查数据预处理

## 示例配置文件

完整的配置示例请参考：
- `configs/qa_tiger/vitl14_with_mamba_aggregator.py`

## 测试脚本

运行测试脚本验证集成：
```bash
python src/models/test_mamba_aggregator.py
```

## 注意事项

1. **兼容性**: VideoMamba聚合器与传统TempMoE完全兼容，可以通过配置开关
2. **内存使用**: VideoMamba聚合器可能使用更多显存，注意batch_size调整
3. **训练时间**: 初期训练可能较慢，但收敛后推理速度更快
4. **超参数**: 建议从默认配置开始，根据具体任务调优

## 未来改进方向

1. **自适应融合**: 根据问题类型动态选择融合方式
2. **多尺度聚合**: 在不同时间尺度上进行聚合
3. **知识蒸馏**: 从传统TempMoE向VideoMamba聚合器迁移知识
4. **压缩优化**: 进一步减少参数量和计算量
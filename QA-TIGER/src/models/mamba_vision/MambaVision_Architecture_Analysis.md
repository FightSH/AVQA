# MambaVision 模型架构分析

## 概述
MambaVision 是一个结合了卷积神经网络（CNN）和 Mamba 状态空间模型（SSM）的混合视觉架构。该模型通过在不同阶段使用不同的计算模块来平衡效率和性能。

## 核心组件

### 1. 补丁嵌入 (PatchEmbed)
```python
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),  # 第一次下采样
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),       # 第二次下采样
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )
```
- **功能**: 将输入图像转换为特征表示
- **操作**: 两次 3x3 卷积，每次步长为 2，总共将分辨率降低 4 倍
- **输出**: 从 224x224 → 56x56 的特征图

### 2. 卷积块 (ConvBlock)
```python
class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
```
- **结构**: 残差连接的双卷积块
- **激活函数**: GELU (tanh 近似)
- **正则化**: BatchNorm + DropPath + 可选的 LayerScale

### 3. Mamba 混合器 (MambaVisionMixer)
这是模型的核心创新组件，实现了状态空间模型：

#### 关键参数
- `d_model`: 模型维度
- `d_state=16`: 状态维度
- `d_conv=4`: 卷积核大小
- `expand=2`: 扩展比例

#### 核心机制
```python
def forward(self, hidden_states):
    xz = self.in_proj(hidden_states)  # 线性投影
    x, z = xz.chunk(2, dim=1)         # 分割为两部分
    
    # 1D 卷积处理
    x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, ...))
    z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, ...))
    
    # 状态空间计算
    y = selective_scan_fn(x, dt, A, B, C, self.D.float(), ...)
    
    # 合并和输出投影
    y = torch.cat([y, z], dim=1)
    out = self.out_proj(y)
```

### 4. 注意力机制 (Attention)
标准的多头自注意力实现：
- 支持 QK 归一化
- 使用 `F.scaled_dot_product_attention` 进行高效计算
- 可配置的注意力和投影 dropout

### 5. 混合块 (Block)
根据 `counter` 和 `transformer_blocks` 参数决定使用注意力还是 Mamba：
```python
if counter in transformer_blocks:
    self.mixer = Attention(...)  # 使用注意力
else:
    self.mixer = MambaVisionMixer(...)  # 使用 Mamba
```

## 整体架构

### 层级结构 (MambaVisionLayer)
每个层级包含：
- **深度**: 该层级的块数量
- **窗口大小**: 用于窗口注意力的窗口尺寸
- **混合器类型**: 卷积块 vs Transformer 块
- **下采样**: 可选的分辨率降低

### 四阶段设计
```python
# 示例配置 (mamba_vision_B)
depths = [3, 3, 10, 5]           # 每阶段的层数
num_heads = [2, 4, 8, 16]        # 每阶段的注意力头数
window_size = [8, 8, 14, 7]      # 每阶段的窗口大小
dim = 128                        # 基础维度
```

1. **阶段 1-2**: 使用卷积块 (`conv=True`)
2. **阶段 3-4**: 使用混合的 Mamba + Attention 块

### 窗口机制
对于 Transformer 块，使用窗口分割来提高效率：
```python
def window_partition(x, window_size):
    # 将特征图分割为不重叠的窗口
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows
```

## 模型变体

| 模型 | 维度 | 深度 | 参数量 | 特点 |
|------|------|------|--------|------|
| mamba_vision_T | 80 | [1,3,8,4] | ~7M | 轻量级 |
| mamba_vision_S | 96 | [3,3,7,5] | ~25M | 小型 |
| mamba_vision_B | 128 | [3,3,10,5] | ~75M | 基础版 |
| mamba_vision_L | 196 | [3,3,10,5] | ~200M | 大型 |
| mamba_vision_L2 | 196 | [3,3,12,5] | ~240M | 更深的大型 |
| mamba_vision_L3 | 256 | [3,3,20,10] | ~740M | 超大型 |

## 关键创新点

### 1. 混合架构
- **早期阶段**: 使用卷积块处理低级特征
- **后期阶段**: 使用 Mamba + Attention 处理高级语义特征

### 2. 状态空间模型集成
- 将 Mamba 的线性复杂度优势引入视觉任务
- 通过选择性扫描实现长距离依赖建模

### 3. 窗口注意力
- 在 Transformer 块中使用窗口机制降低计算复杂度
- 支持不同阶段的不同窗口大小

### 4. 灵活的块分配
- 通过 `transformer_blocks` 参数控制每层使用的混合器类型
- 通常后半部分层使用注意力机制

## 计算复杂度
- **卷积阶段**: O(H×W×C²)
- **Mamba 阶段**: O(H×W×C) - 线性复杂度
- **注意力阶段**: O(W²×C) - 窗口内二次复杂度

这种设计在保持高性能的同时显著降低了计算成本，特别适合处理高分辨率图像。
## 后
两个阶段（阶段3和阶段4）的输入输出详细分析

### 输入要求

#### 1. 输入格式
- **形状**: `(B, C, H, W)` - 标准的4D张量格式
  - `B`: 批次大小
  - `C`: 通道数（特征维度）
  - `H`: 特征图高度
  - `W`: 特征图宽度

#### 2. 分辨率要求
后两个阶段使用 Transformer 块，需要进行窗口分割，因此对输入分辨率有特殊要求：

```python
def forward(self, x):
    _, _, H, W = x.shape
    
    if self.transformer_block:
        # 计算需要的填充
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        # 如果需要，进行填充使分辨率能被窗口大小整除
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
```

**关键要求**:
- 输入的 H 和 W 最好能被对应的 `window_size` 整除
- 如果不能整除，模型会自动进行零填充
- 不同模型变体的窗口大小不同：
  - 阶段3: 通常为 14 (基础模型) 或 16/32 (大模型)
  - 阶段4: 通常为 7 (基础模型) 或 8/16 (大模型)

#### 3. 通道维度要求
- **阶段3**: 输入通道数 = `dim * 4` (经过前两个阶段的2次下采样，每次通道翻倍)
- **阶段4**: 输入通道数 = `dim * 8` (经过三个阶段后)

### 数据流转换过程

#### 窗口分割过程 (window_partition)
```python
def window_partition(x, window_size):
    # 输入: (B, C, H, W)
    B, C, H, W = x.shape
    
    # 重塑为窗口格式
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    
    # 输出: (num_windows*B, window_size*window_size, C)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows
```

**转换说明**:
- 输入: `(B, C, H, W)`
- 输出: `(num_windows*B, window_size², C)`
- `num_windows = (H//window_size) * (W//window_size)`

#### 块内处理
每个 Block 内部的处理流程：

1. **Mamba 块处理**:
   ```python
   # 输入: (num_windows*B, window_size², C)
   # MambaVisionMixer 期望: (B, L, D) 格式
   # 其中 L = window_size², D = C
   ```

2. **Attention 块处理**:
   ```python
   # 输入: (num_windows*B, window_size², C)
   # 标准的自注意力计算
   # 输出: (num_windows*B, window_size², C)
   ```

#### 窗口恢复过程 (window_reverse)
```python
def window_reverse(windows, window_size, H, W):
    # 输入: (num_windows*B, window_size², C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    
    # 重塑回原始格式
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    
    # 输出: (B, C, H, W)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x
```

### 输出特征

#### 1. 输出格式
- **形状**: `(B, C_out, H_out, W_out)`
- **通道变化**: 
  - 如果有下采样: `C_out = 2 * C_in`
  - 如果无下采样: `C_out = C_in`

#### 2. 分辨率变化
- **有下采样的层**: `H_out = H_in // 2`, `W_out = W_in // 2`
- **无下采样的层**: `H_out = H_in`, `W_out = W_in`

#### 3. 具体示例 (以 mamba_vision_B 为例)

假设输入图像为 224×224：

| 阶段 | 输入形状 | 窗口大小 | 输出形状 | 说明 |
|------|----------|----------|----------|------|
| 阶段1 | (B, 128, 56, 56) | - | (B, 256, 28, 28) | 卷积+下采样 |
| 阶段2 | (B, 256, 28, 28) | - | (B, 512, 14, 14) | 卷积+下采样 |
| **阶段3** | **(B, 512, 14, 14)** | **14** | **(B, 1024, 7, 7)** | **Mamba+Attn+下采样** |
| **阶段4** | **(B, 1024, 7, 7)** | **7** | **(B, 1024, 7, 7)** | **Mamba+Attn，无下采样** |

### 关键约束和注意事项

1. **窗口大小匹配**: 输入分辨率必须与窗口大小兼容
2. **内存效率**: 窗口分割减少了注意力计算的复杂度
3. **序列长度**: 每个窗口内的序列长度为 `window_size²`
4. **批次处理**: 窗口分割后批次维度变为 `num_windows * B`

这种设计使得后两个阶段能够高效处理高级语义特征，同时保持合理的计算复杂度。
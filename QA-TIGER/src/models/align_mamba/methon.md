
### 3. 方法 (Method)

本节详细阐述了 AlignMamba 框架的设计。该框架通过整合局部和全局跨模态对齐机制，来增强 Mamba 模型在多模态融合任务中的表现。

#### 3.1 概述 (Overview)

AlignMamba 的整体流程如图2所示，以音频-视频-语言三模态融合为例：

1.  **单模态编码 (Unimodal Encoding):** 首先，来自各个模态的原始信号（如音频波形、视频帧、文本）通过各自模态特定的编码器（Encoders）被转换为 unimodal embedding sequences，记为 `Xa` (音频), `Xv` (视频), 和 `Xl` (语言)。
2.  **双重对齐 (Dual Alignment):** 接下来，框架采用两种互补的对齐机制：
    *   **局部对齐 (Local Alignment):** 一个基于最优传输 (OT) 的模块，用于学习 token 级别的显式对应关系。这里以语言模态 `Xl` 为锚点，将 `Xa` 和 `Xv` 向其对齐，生成对齐后的特征 `X̃a` 和 `X̃v`。
    *   **全局对齐 (Global Alignment):** 一个基于最大均值差异 (MMD) 的损失函数，用于确保不同模态在整体分布上的一致性。
3.  **多模态融合 (Multimodal Fusion):** 最后，对齐后的单模态嵌入 `X̃a`, `X̃v` 和锚点嵌入 `Xl` 被送入 Mamba 骨干网络进行高效的跨模态交互和最终融合。

以下各小节将详细描述每个组件。

---

#### 3.2 基于最优传输的局部跨模态对齐 (OT-based Local Cross-modal Alignment)

最优传输 (Optimal Transport, OT) 提供了一个有原则的框架，通过最小化运输成本来比较和对齐概率分布。在多模态对齐的背景下，OT 自然地适用于建立不同模态 token 之间的对应关系。

给定来自音频、视频和语言模态的特征序列 `Xa ∈ R^(Ta×d)`, `Xv ∈ R^(Tv×d)`, 和 `Xl ∈ R^(Tl×d)`，其中 `Ta`, `Tv`, `Tl` 分别是序列长度，`d` 是特征维度。我们的目标是学习一个传输矩阵 `M` 来捕捉模态间的细粒度对应关系。

以视频到语言的对齐为例，经典的 OT 问题可以表述为：

**最小化总运输成本:**
$$
\min_{\mathbf{M}_{v2l}} \sum_{i=1}^{T_v} \sum_{j=1}^{T_l} \mathbf{M}_{v2l}(i, j) \mathbf{C}_{v2l}(i, j) \quad \cdots \quad (1)
$$

其中 `M_v2l` 是传输矩阵，`C_v2l` 是成本矩阵。

**约束条件:**
$$
\begin{cases}
\sum_{j=1}^{T_l} \mathbf{M}_{v2l}(i, j) = \frac{1}{T_v}, & \forall i \in [1, T_v] \\
\sum_{i=1}^{T_v} \mathbf{M}_{v2l}(i, j) = \frac{1}{T_l}, & \forall j \in [1, T_l] \\
\mathbf{M}_{v2l}(i, j) \ge 0, & \forall i, j
\end{cases} \quad \cdots \quad (2)
$$

成本矩阵 `C_v2l ∈ R^(Tv×Tl)` 定义了匹配视频 token `i` 和语言 token `j` 的成本。我们采用**余弦距离**作为成本函数，因为它能有效捕捉特征向量间的角度关系，并且数值稳定。

**成本矩阵定义:**
$$
\mathbf{C}_{v2l}(i, j) = 1 - \frac{\mathbf{X}_v(i) \cdot \mathbf{X}_l(j)}{\|\mathbf{X}_v(i)\|_2 \|\mathbf{X}_l(j)\|_2} \quad \cdots \quad (3)
$$

然而，求解经典的 OT 问题计算量极大。因此，我们借鉴 [12] 的工作，采用一个**松弛版本 (relaxed version)**，通过移除“传入流量总和”的约束（即公式(2)中的第二个等式）来简化问题。

**松弛后的约束条件:**
$$
\begin{cases}
\sum_{j=1}^{T_l} \mathbf{M}_{v2l}(i, j) = \frac{1}{T_v}, & \forall i \in [1, T_v] \\
\mathbf{M}_{v2l}(i, j) \ge 0, & \forall i, j
\end{cases} \quad \cdots \quad (4)
$$

这种松弛允许一个文本特征匹配多个视频特征，显著降低了计算复杂度的同时，依然能捕捉有意义的跨模态对应关系。该松弛问题的解可以被贪婪地确定。

**松弛版OT的解:**
$$
\mathbf{M}_{v2l}(i, j) =
\begin{cases}
\frac{1}{T_v}, & \text{if } j = \arg\min_{j'} \mathbf{C}_{v2l}(i, j') \\
0, & \text{otherwise}
\end{cases} \quad \cdots \quad (5)
$$

同样地，我们计算音频到语言的传输矩阵 `M_a2l`。最后，通过传输矩阵对源模态特征进行变换，得到对齐后的特征：

**生成对齐特征:**
$$
\begin{cases}
\tilde{\mathbf{X}}_v = \mathbf{M}_{v2l}^T \mathbf{X}_v \in \mathbb{R}^{T_l \times d} \\
\tilde{\mathbf{X}}_a = \mathbf{M}_{a2l}^T \mathbf{X}_a \in \mathbb{R}^{T_l \times d}
\end{cases} \quad \cdots \quad (6)
$$
*(注意：论文原文公式(6)的矩阵乘法形式可能存在歧义，从实现角度看，`M` 的维度应为 `Tv x Tl`，对齐操作为 `M^T * Xv`，或者 `M` 的维度为 `Tl x Tv`，操作为 `M * Xv`。这里根据典型实现 `M^T * Xv` 进行表述，其中 `M` 的维度为 `Tv x Tl`)*

---

#### 3.3 基于MMD的全局跨模态对齐 (MMD-based Global Cross-modal Alignment)

为了确保模态间的分布级一致性，我们采用最大均值差异 (Maximum Mean Discrepancy, MMD) 作为全局对齐的度量。MMD 通过比较两个分布在再生核希尔伯特空间 (RKHS) 中的均值来衡量它们的统计差异。

对于两个特征序列 `X` 和 `Y`，其平方MMD距离定义为：
$$
\text{MMD}^2(\mathbf{X}, \mathbf{Y}) = \left\| \frac{1}{T} \sum_{i=1}^{T} \phi(\mathbf{x}_i) - \frac{1}{T} \sum_{j=1}^{T} \phi(\mathbf{y}_j) \right\|_{\mathcal{H}}^2 \quad \cdots \quad (7)
$$
其中 `φ(·)` 是一个映射到 RKHS `H` 的特征映射。

利用核技巧 (kernel trick)，MMD² 可以被高效计算：
$$
\text{MMD}^2(\mathbf{X}, \mathbf{Y}) = \frac{1}{T^2} \sum_{i=1}^{T} \sum_{i'=1}^{T} k(\mathbf{x}_i, \mathbf{x}_{i'}) + \frac{1}{T^2} \sum_{j=1}^{T} \sum_{j'=1}^{T} k(\mathbf{y}_j, \mathbf{y}_{j'}) - \frac{2}{T^2} \sum_{i=1}^{T} \sum_{j=1}^{T} k(\mathbf{x}_i, \mathbf{y}_j) \quad \cdots \quad (8)
$$
其中 `k(·,·)` 是一个正定核函数。在我们的实现中，我们采用高斯核：

**高斯核函数:**
$$
k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|_2^2}{2\sigma^2}\right) \quad \cdots \quad (9)
$$
其中 `σ` 是核带宽参数。

对于对齐后的音频特征 `X̃a`、视频特征 `X̃v` 和语言特征 `Xl`，全局对齐损失 `L_align` 定义为各对模态之间 MMD 距离的总和：

**全局对齐损失:**
$$
\mathcal{L}_{\text{align}} = \text{MMD}^2(\tilde{\mathbf{X}}_v, \mathbf{X}_l) + \text{MMD}^2(\tilde{\mathbf{X}}_a, \mathbf{X}_l) \quad \cdots \quad (10)
$$

在训练中最小化此损失，可以促使不同模态的特征分布在 RKHS 中对齐。

---

#### 3.4 基于Mamba的融合与优化 (Mamba-based Fusion and Optimization)

**Mamba-based Multimodal Fusion:**
在局部和全局对齐之后，我们利用 Mamba 进行高效的多模态融合。为了使 Mamba 能够有效地处理跨模态交互，我们构建一个统一的多模态特征序列 `X_mm`。该序列通过**时间优先的交错策略 (time-priority interleaving strategy)** 形成，将不同模态在同一时间步的特征组织在一起：

**交错序列:**
$$
\mathbf{X}_{mm} = [\tilde{\mathbf{X}}_v^1, \tilde{\mathbf{X}}_a^1, \mathbf{X}_l^1, \tilde{\mathbf{X}}_v^2, \tilde{\mathbf{X}}_a^2, \mathbf{X}_l^2, \dots, \tilde{\mathbf{X}}_v^{T_l}, \tilde{\mathbf{X}}_a^{T_l}, \mathbf{X}_l^{T_l}] \quad \cdots \quad (11)
$$
其中上标表示时间索引。这种组织方式使得 Mamba 的选择性扫描机制能够同时捕捉到时间步内的**模态间 (inter-modal)**依赖和时间步之间的**模态内 (intra-modal)**依赖。

**Training Objective:**
整个框架通过一个复合损失函数进行端到端的优化，该函数结合了任务特定目标和对齐约束：

**总损失函数:**
$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{align}} \quad \cdots \quad (12)
$$
其中：
*   `L_task` 是下游任务的损失函数（例如，分类任务的交叉熵损失或回归任务的均方误差损失）。
*   `L_align` 是公式 (10) 中定义的 MMD 全局对齐损失。
*   `λ` 是一个超参数，用于平衡两个损失项。

在训练过程中，最小化 `L_task` 驱动模型学习与任务相关的多模态表示，而最小化 `L_align` 则确保了不同模态特征分布的一致性。
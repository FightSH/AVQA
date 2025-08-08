---

### 3. Method

#### 3.1. Overview

Fig. 2 presents the framework of our proposed AlignMamba. Using audio-visual-language trimodal data as a case study, the framework first processes raw signals from each modality through modality-specific encoders to generate corresponding unimodal embedding sequences $X_a$, $X_v$, and $X_l$. The framework then employs two complementary alignment mechanisms: an OT-based local alignment module that captures token-level correspondences, and an MMD-based global alignment loss that ensures distribution-level consistency. These mechanisms yield aligned embedding sequences $\tilde{X}_a$ and $\tilde{X}_v$ (illustrated here by aligning audio and visual modalities to the language modality as the anchor). The aligned unimodal embeddings, which now incorporate cross-modal correspondence information, are subsequently processed by the Mamba backbone for multimodal fusion. The following sections provide a detailed description of each component.

#### 3.2. OT-based Local Cross-modal Alignment

Optimal Transport provides a principled framework for comparing and aligning probability distributions by finding the optimal way to transform one distribution into another while minimizing the transportation cost [32]. In our multimodal alignment context, OT offers a natural way to establish token-level correspondences between different modalities by treating feature sequences as discrete distributions.

Given the unimodal feature sequences $X_a \in \mathbb{R}^{T_a \times d}$, $X_v \in \mathbb{R}^{T_v \times d}$, and $X_l \in \mathbb{R}^{T_l \times d}$ from audio, video, and language modalities respectively, where $T_a$, $T_v$, and $T_l$ denote the sequence lengths of different modalities and $d$ is the feature dimension, we aim to learn the transport matrix $M$ that capture fine-grained correspondences between different modalities. Take video-to-language alignment as an example, the classical optimal transport problem can be formulated as follows:
$$
\min_{\mathbf{M}_{v2l}} \sum_{i=1}^{T_v} \sum_{j=1}^{T_l} \mathbf{M}_{v2l}(i, j) \mathbf{C}_{v2l}(i, j).
\quad\quad(1)
$$
The optimization is constrained by:
$$
\begin{cases}
\sum_{j=1}^{T_l} \mathbf{M}_{v2l}(i, j) = \frac{1}{T_v}, & \forall i \in [1, T_v] \\
\sum_{i=1}^{T_v} \mathbf{M}_{v2l}(i, j) = \frac{1}{T_l}, & \forall j \in [1, T_l] \\
\mathbf{M}_{v2l}(i, j) \ge 0, & \forall i, j
\end{cases}
\quad\quad(2)
$$
where $\mathbf{C}_{v2l} \in \mathbb{R}^{T_v \times T_l}$ is the cost matrix. Given that the cosine distance emphasizes angular relationships between feature vectors while providing numerical stability through its bounded range, we use cosine distance as the cost matrix:
$$
\mathbf{C}_{v2l}(i, j) = 1 - \frac{\mathbf{X}_v^i \cdot \mathbf{X}_l^j}{\|\mathbf{X}_v^i\|_2 \|\mathbf{X}_l^j\|_2}.
\quad\quad(3)
$$
However, solving this OT problem is extremely computationally expensive. Following [12], we adopt a relaxed version by removing the incoming sum constraint:
$$
\begin{cases}
\sum_{j=1}^{T_l} \mathbf{M}_{v2l}(i, j) = \frac{1}{T_v}, & \forall i \in [1, T_v] \\
\mathbf{M}_{v2l}(i, j) \ge 0, & \forall i, j
\end{cases}
\quad\quad(4)
$$
This relaxed formulation allows each textual feature to be matched with multiple video features without constraining the total incoming flow, significantly reducing the computational complexity while maintaining the ability to capture meaningful cross-modal correspondences. The corresponding solution is defined as:
$$
\mathbf{M}_{v2l}(i, j) =
\begin{cases}
\frac{1}{T_v}, & j = \arg\min_{j'} \mathbf{C}_{v2l}(i, j'), \\
0, & j \neq \arg\min_{j'} \mathbf{C}_{v2l}(i, j').
\end{cases}
\quad\quad(5)
$$
Similarly, we compute the transport matrix $\mathbf{M}_{a2l}$ for audio-to-language alignment. Finally, the aligned video and audio features can then be obtained through:
$$
\begin{cases}
\tilde{\mathbf{X}}_v = \mathbf{M}_{v2l}^T \mathbf{X}_v \in \mathbb{R}^{T_l \times d}, \\
\tilde{\mathbf{X}}_a = \mathbf{M}_{a2l}^T \mathbf{X}_a \in \mathbb{R}^{T_l \times d}.
\end{cases}
\quad\quad(6)
$$
This relaxed OT-based alignment process provides an efficient way to capture fine-grained cross-modal correspondences while maintaining computational tractability. The resulting transport matrices provide interpretable alignment information between different modalities. However, while this token-level alignment effectively captures local correspondences, ensuring global distribution-level consistency across modalities requires additional consideration, which we address through our MMD-based global alignment mechanism in the following section.

#### 3.3. MMD-based Global Cross-modal Alignment

To ensure distribution-level consistency across modalities, we employ Maximum Mean Discrepancy as the global alignment metric. MMD measures the statistical discrepancy between different modalities in a high-dimensional Reproducing Kernel Hilbert Space (RKHS) by comparing all orders of their statistics. For two feature sequences $\mathbf{X}$ and $\mathbf{Y}$, the squared MMD distance is defined as:
$$
\text{MMD}^2(\mathbf{X}, \mathbf{Y}) = \left\| \frac{1}{T} \sum_{i=1}^{T} \phi(\mathbf{x}_i) - \frac{1}{T} \sum_{j=1}^{T} \phi(\mathbf{y}_j) \right\|_{\mathcal{H}}^2,
\quad\quad(7)
$$
where $\phi(\cdot)$ is a feature mapping to a RKHS $\mathcal{H}$. Using the kernel trick, this can be computed as:
$$
\text{MMD}^2(\mathbf{X}, \mathbf{Y}) = \frac{1}{T^2} \sum_{i=1}^{T} \sum_{i'=1}^{T} k(\mathbf{x}_i, \mathbf{x}_{i'}) + \frac{1}{T^2} \sum_{j=1}^{T} \sum_{j'=1}^{T} k(\mathbf{y}_j, \mathbf{y}_{j'}) - \frac{2}{T^2} \sum_{i=1}^{T} \sum_{j=1}^{T} k(\mathbf{x}_i, \mathbf{y}_j),
\quad\quad(8)
$$
where $k(\cdot, \cdot)$ is a positive definite kernel function. In our implementation, we adopt the Gaussian kernel:
$$
k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|_2^2}{2\sigma^2}\right),
\quad\quad(9)
$$
where $\sigma$ is the kernel bandwidth parameter.

For the aligned audio features $\tilde{\mathbf{X}}_a$, the aligned video features $\tilde{\mathbf{X}}_v$, and the language features $\mathbf{X}_l$, the global alignment loss is defined as the sum of MMD distances between each pair of modalities:
$$
\mathcal{L}_{\text{align}} = \text{MMD}^2(\tilde{\mathbf{X}}_v, \mathbf{X}_l) + \text{MMD}^2(\tilde{\mathbf{X}}_a, \mathbf{X}_l).
\quad\quad(10)
$$
By minimizing this loss during training, we encourage the feature distributions of different modalities to be aligned in the RKHS. While OT establishes token-level correspondences, MMD ensures the consistency of overall feature distributions, providing complementary alignment signals at different granularities. This dual-alignment strategy facilitates more effective multimodal fusion in subsequent processing stages.

#### 3.4. Mamba-based Fusion and Optimization

**Mamba-based Multimodal Fusion.** Following the local and global alignment processes, we employ Mamba to facilitate efficient multimodal fusion while maintaining its inherent linear computational complexity. Unlike traditional Transformer-based methods that process all tokens simultaneously through self-attention mechanisms, our approach implements a time-priority scanning strategy that preserves Mamba's sequential nature while enabling effective cross-modal interactions. Given the aligned audio features $\tilde{\mathbf{X}}_a$, the aligned video features $\tilde{\mathbf{X}}_v$, and the language features $\mathbf{X}_l$, we construct a unified multimodal feature sequence $\mathbf{X}_{mm}$ by interleaving features from different modalities at each timestep:
$$
\mathbf{X}_{mm} = [\tilde{\mathbf{X}}_v^1, \tilde{\mathbf{X}}_a^1, \mathbf{X}_l^1, \tilde{\mathbf{X}}_v^2, \tilde{\mathbf{X}}_a^2, \mathbf{X}_l^2, \dots, \tilde{\mathbf{X}}_v^{T_l}, \tilde{\mathbf{X}}_a^{T_l}, \mathbf{X}_l^{T_l}],
\quad\quad(11)
$$
where the superscript denotes the temporal index. This temporal-priority organization ensures that features from different modalities at the same timestep are processed sequentially, allowing the selective scan mechanism of Mamba to effectively capture both intra- and inter-modal dependencies. The fused representations are obtained by processing the constructed sequence through multiple Mamba layers.

**Training Objective.** The framework is optimized end-to-end using a composite loss function that combines the task-specific objective with the alignment constraints:
$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{align}},
\quad\quad(12)
$$
where $\mathcal{L}_{\text{task}}$ is determined by the downstream task (e.g., cross-entropy loss for classification or mean squared error for regression), $\mathcal{L}_{\text{align}}$ is the MMD-based alignment loss, and $\lambda$ is a hyperparameter that balances the two objectives. During training, minimizing $\mathcal{L}_{\text{task}}$ drives the model to learn task-relevant multimodal representations, while $\mathcal{L}_{\text{align}}$ ensures consistent feature distributions across modalities.
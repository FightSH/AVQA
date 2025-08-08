from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TransformerBackbone(nn.Module):
    """Mamba 的回退实现，使用 TransformerEncoder。"""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return self.encoder(x)


class MambaBackbone(nn.Module):
    """
    Mamba 主干：优先尝试 mamba-ssm，若不可用则回退到 TransformerEncoder。
    输入输出：(B, T, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm.torch_modules.mamba2 import Mamba2

            layers = []
            for _ in range(n_layers):
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        Mamba2(d_model=d_model, d_state=16, d_conv=4, expand=2),
                        nn.Dropout(dropout),
                    )
                )
            self.backbone = nn.Sequential(*layers)
        except Exception:
            self.backbone = _TransformerBackbone(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def cosine_cost_matrix(src: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    余弦距离成本矩阵：C(i,j)=1-cos(src_i,tgt_j)。
    src: (B,Ts,d), tgt: (B,Tt,d) -> (B,Ts,Tt)
    """
    src_n = F.normalize(src, dim=-1, eps=eps)
    tgt_n = F.normalize(tgt, dim=-1, eps=eps)
    cosine = torch.matmul(src_n, tgt_n.transpose(1, 2))  # (B, Ts, Tt)
    return 1.0 - cosine.clamp(-1.0, 1.0)


def relaxed_ot_matrix(cost: torch.Tensor) -> torch.Tensor:
    """
    松弛 OT：每个源 token 选 argmin 目标，置 1/Ts。cost:(B,Ts,Tt) -> M:(B,Ts,Tt)
    """
    B, Ts, Tt = cost.shape
    with torch.no_grad():
        idx = cost.argmin(dim=2)  # (B, Ts)
    M = torch.zeros((B, Ts, Tt), device=cost.device, dtype=cost.dtype)
    M.scatter_(2, idx.unsqueeze(-1), 1.0)
    M = M / float(Ts)
    return M


def apply_alignment(src: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    X_tilde = M^T @ X_src。src:(B,Ts,d), M:(B,Ts,Tt) -> (B,Tt,d)
    """
    return torch.bmm(M.transpose(1, 2), src)


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """高斯核：exp(-||x-y||^2/(2*sigma^2))，返回 (B,Tx,Ty)。"""
    x2 = (x * x).sum(dim=-1, keepdim=True)  # (B, Tx, 1)
    y2 = (y * y).sum(dim=-1).unsqueeze(1)   # (B, 1, Ty)
    xy = torch.matmul(x, y.transpose(1, 2))  # (B, Tx, Ty)
    dist2 = (x2 + y2 - 2.0 * xy).clamp_min(0.0)
    return torch.exp(-dist2 / (2.0 * (sigma ** 2)))


def mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """有偏 MMD^2 估计，返回每个 batch 的标量 (B,)。"""
    B, T, _ = x.shape
    k_xx = rbf_kernel(x, x, sigma)
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)
    scale = 1.0 / float(T * T)
    return scale * (k_xx.sum(dim=(1, 2)) + k_yy.sum(dim=(1, 2)) - 2.0 * k_xy.sum(dim=(1, 2)))


class AlignMamba(nn.Module):
    """
    - OT 局部对齐（语言为锚）
    - MMD 全局对齐
    - 时间优先交错 + Mamba/Transformer 融合
    - L = L_task + lambda * L_align
    """

    def __init__(
        self,
        dim_a: int,
        dim_v: int,
        dim_l: int,
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        sigma: float = 1.0,
        lambda_align: float = 0.1,
        task_type: Literal["classification", "regression", "none"] = "none",
        num_classes: Optional[int] = None,
        pooling: Literal["mean", "max"] = "mean",
    ) -> None:
        super().__init__()
        # 模态投影到统一维度
        self.proj_a = nn.Linear(dim_a, d_model) if dim_a != d_model else nn.Identity()
        self.proj_v = nn.Linear(dim_v, d_model) if dim_v != d_model else nn.Identity()
        self.proj_l = nn.Linear(dim_l, d_model) if dim_l != d_model else nn.Identity()
        # 主干
        self.backbone = MambaBackbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
        )
        # 任务头
        self.task_type = task_type
        self.pooling = pooling
        if task_type == "classification":
            assert num_classes is not None and num_classes > 1, "分类任务需提供 num_classes > 1"
            self.head = nn.Linear(d_model, num_classes)
            self.ce_loss = nn.CrossEntropyLoss()
        elif task_type == "regression":
            out_dim = 1 if (num_classes is None) else int(num_classes)
            self.head = nn.Linear(d_model, out_dim)
            self.mse_loss = nn.MSELoss()
        else:
            self.head = nn.Identity()
        # 对齐超参
        self.sigma = float(sigma)
        self.lambda_align = float(lambda_align)

    @staticmethod
    def _pool(seq: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if mode == "max":
            return seq.max(dim=1).values
        return seq.mean(dim=1)

    def local_align(self, Xa: torch.Tensor, Xv: torch.Tensor, Xl: torch.Tensor):
        """松弛 OT 对齐到语言时间轴：返回 Xa~, Xv~ (B,Tl,d)。"""
        Ca2l = cosine_cost_matrix(Xa, Xl)
        Ma2l = relaxed_ot_matrix(Ca2l)
        Xa_tilde = apply_alignment(Xa, Ma2l)
        Cv2l = cosine_cost_matrix(Xv, Xl)
        Mv2l = relaxed_ot_matrix(Cv2l)
        Xv_tilde = apply_alignment(Xv, Mv2l)
        return Xa_tilde, Xv_tilde

    def global_align_loss(self, Xa_tilde: torch.Tensor, Xv_tilde: torch.Tensor, Xl: torch.Tensor) -> torch.Tensor:
        mmd_vl = mmd2(Xv_tilde, Xl, sigma=self.sigma)
        mmd_al = mmd2(Xa_tilde, Xl, sigma=self.sigma)
        return (mmd_vl + mmd_al).mean()

    def build_interleaved(self, Xa_tilde: torch.Tensor, Xv_tilde: torch.Tensor, Xl: torch.Tensor) -> torch.Tensor:
        """时间优先交错：[Xv^t~, Xa^t~, Xl^t], t=1..Tl -> (B,3*Tl,d)"""
        B, T, d = Xl.shape
        stacked = torch.stack([Xv_tilde, Xa_tilde, Xl], dim=2)  # (B,T,3,d)
        return stacked.reshape(B, T * 3, d)

    def forward(
        self,
        Xa: torch.Tensor,
        Xv: torch.Tensor,
        Xl: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # 投影
        Xa = self.proj_a(Xa)
        Xv = self.proj_v(Xv)
        Xl = self.proj_l(Xl)
        # 局部对齐（OT）
        Xa_tilde, Xv_tilde = self.local_align(Xa, Xv, Xl)
        # 全局对齐（MMD）
        loss_align = self.global_align_loss(Xa_tilde, Xv_tilde, Xl)
        # 融合（交错 + 主干）
        X_mm = self.build_interleaved(Xa_tilde, Xv_tilde, Xl)
        X_mm = self.backbone(X_mm)

        out: Dict[str, torch.Tensor] = {}
        if return_features:
            out["features"] = X_mm

        # 任务头
        loss_task = None
        if not isinstance(self.head, nn.Identity):
            pooled = self._pool(X_mm, mode=self.pooling)
            logits = self.head(pooled)
            out["logits"] = logits
            if self.task_type == "classification" and labels is not None:
                loss_task = self.ce_loss(logits, labels)
            elif self.task_type == "regression" and labels is not None:
                if labels.dim() == 1:
                    labels = labels.unsqueeze(-1)
                loss_task = self.mse_loss(logits, labels)

        total_loss = self.lambda_align * loss_align if loss_task is None else (loss_task + self.lambda_align * loss_align)
        out["loss_align"] = loss_align
        if loss_task is not None:
            out["loss_task"] = loss_task
        out["loss"] = total_loss
        return out


__all__ = ["AlignMamba", "MambaBackbone"]
#!/usr/bin/env python
"""
Shared fleet / set-attention architecture
=========================================
Single source of truth for ISAB, SetBlock, FleetPolicy, and baseline set models.

Soft Reynolds proxy
-------------------
SetBlock cross-attends elements to the set mean centroid in R^h. This is NOT the
hard rank-1 P_inv of docs/Paper.md §2. See Paper §7 ("Code vs theorem").
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# Core building blocks
# ============================================================

class ISAB(nn.Module):
    """Induced Set Attention Block (Lee et al., 2019)."""

    def __init__(self, h: int, m: int = 8, n_heads: int = 4):
        super().__init__()
        self.m = m
        self.I = nn.Parameter(torch.randn(m, h) * 0.02)
        self.attn1 = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (n, h)
        I = self.I.unsqueeze(0)  # (1, m, h)
        out1, _ = self.attn1(I, X.unsqueeze(0), X.unsqueeze(0))
        out1 = self.norm1(out1.squeeze(0))
        out2, _ = self.attn2(X.unsqueeze(0), out1.unsqueeze(0), out1.unsqueeze(0))
        return self.norm2(out2.squeeze(0))


class SetBlock(nn.Module):
    """ISAB + FFN + optional soft Reynolds (centroid cross-attention)."""

    def __init__(
        self,
        h: int,
        m: int = 8,
        use_reynolds: bool = True,
        use_isab: bool = True,
        n_heads: int = 4,
    ):
        super().__init__()
        self.use_reynolds = use_reynolds
        self.use_isab = use_isab
        if use_isab:
            self.isab = ISAB(h, m, n_heads=n_heads)
            self.norm1 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
        self.norm2 = nn.LayerNorm(h)
        if use_reynolds:
            self.xattn = nn.MultiheadAttention(h, num_heads=n_heads, batch_first=True)
            self.norm3 = nn.LayerNorm(h)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if self.use_isab:
            H = self.norm1(H + self.isab(H))
        H = self.norm2(H + self.ffn(H))
        if self.use_reynolds:
            c = H.mean(dim=0, keepdim=True)  # soft Reynolds proxy
            H, _ = self.xattn(H, c, c)
            H = self.norm3(H)
        return H


class FleetPolicy(nn.Module):
    """
    Permutation-invariant set → cost-matrix predictor.

    Parameters
    ----------
    score_mode :
        ``"shared"``  — project H and T through a shared MLP to ``score_dim``
                        (breadth / baseline / complexity default).
        ``"element"`` — optional MLP on H only, then ``H @ T.T`` (ablation).
        ``"dot"``     — plain ``H @ T.T`` after cross-attn (no score MLP).
    """

    def __init__(
        self,
        d_in: int,
        h: int = 128,
        L: int = 4,
        m: int = 8,
        d_task: int = 2,
        use_reynolds: bool = True,
        use_isab: bool = True,
        use_cross_attn: bool = True,
        use_score_head: bool = True,
        score_mode: str = "shared",
        score_dim: int = 32,
        n_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        # Accept legacy aliases from older call sites
        if "d" in kwargs and kwargs["d"] is not None:
            d_in = kwargs["d"]
        # n_tasks was informational only in older constructors

        if score_mode not in ("shared", "element", "dot"):
            raise ValueError(f"Unknown score_mode={score_mode!r}")

        self.use_cross_attn = use_cross_attn
        self.use_score_head = use_score_head
        self.score_mode = score_mode

        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([
            SetBlock(h, m, use_reynolds=use_reynolds, use_isab=use_isab, n_heads=n_heads)
            for _ in range(L)
        ])
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(h, num_heads=n_heads, batch_first=True)
            self.final_norm = nn.LayerNorm(h)

        if use_score_head and score_mode == "shared":
            self.score_proj = nn.Sequential(
                nn.Linear(h, 64), nn.GELU(), nn.Linear(64, score_dim)
            )
        elif use_score_head and score_mode == "element":
            self.score_head = nn.Sequential(
                nn.Linear(h, 64), nn.GELU(), nn.Linear(64, h)
            )

    def forward(self, X: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        H = self.enc(X)
        for blk in self.blocks:
            H = blk(H)
        T = self.task_enc(tasks)

        if self.use_cross_attn:
            H, _ = self.cross_attn(H, T, T)
            H = self.final_norm(H)

        if self.use_score_head and self.score_mode == "shared":
            return self.score_proj(H) @ self.score_proj(T).T
        if self.use_score_head and self.score_mode == "element":
            H = self.score_head(H)
            return H @ T.T
        return H @ T.T


# ============================================================
# Baseline set models (for baseline_comparison.py)
# ============================================================

class DeepSets(nn.Module):
    """Elementwise MLP → sum pool → MLP → broadcast."""

    def __init__(self, d_in: int, h: int = 128, d_task: int = 2, d_score: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))
        self.task_score = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))

    def forward(self, X: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        H = self.phi(X)
        c = self.rho(H.sum(dim=0, keepdim=True))
        H_s = self.score_proj(H + c)
        T_s = self.task_score(self.task_enc(tasks))
        return H_s @ T_s.T


class PointNet(nn.Module):
    """Elementwise MLP → max pool → MLP → broadcast."""

    def __init__(self, d_in: int, h: int = 128, d_task: int = 2, d_score: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))
        self.task_score = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))

    def forward(self, X: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        H = self.phi(X)
        c = self.rho(H.max(dim=0, keepdim=True).values)
        H_s = self.score_proj(H + c)
        T_s = self.task_score(self.task_enc(tasks))
        return H_s @ T_s.T


class PAB(nn.Module):
    """Permutation Attention Block (Set Transformer)."""

    def __init__(self, h: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(h)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(X.unsqueeze(0), X.unsqueeze(0), X.unsqueeze(0))
        return self.norm(out.squeeze(0))


class SetTransformer(nn.Module):
    """ISAB × L + PAB × K (no Reynolds centroid)."""

    def __init__(
        self,
        d_in: int,
        h: int = 128,
        L: int = 4,
        m: int = 8,
        K: int = 2,
        d_task: int = 2,
        d_score: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.isabs = nn.ModuleList([ISAB(h, m, n_heads=n_heads) for _ in range(L)])
        self.pabs = nn.ModuleList([PAB(h, n_heads=n_heads) for _ in range(K)])
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.cross_attn = nn.MultiheadAttention(h, num_heads=n_heads, batch_first=True)
        self.final_norm = nn.LayerNorm(h)
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))
        self.task_score = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))

    def forward(self, X: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        H = self.enc(X)
        for isab in self.isabs:
            H = H + isab(H)
        for pab in self.pabs:
            H = H + pab(H)
        T = self.task_enc(tasks)
        H, _ = self.cross_attn(H, T, T)
        H = self.final_norm(H)
        return self.score_proj(H) @ self.task_score(T).T

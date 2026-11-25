#!/usr/bin/env python
"""
Ablation Study: Constrained-Operation Catalogue
===============================================
Systematically removes each Active component to prove the Reynolds
projection is the sole cause of depth-irrelevance.

Variants:
  1. Full (baseline)       — ISAB + Reynolds + Cross-Attn + Score Head
  2. No Reynolds           — remove centroid cross-attention from SetBlock
  3. No ISAB               — replace ISAB with identity
  4. No Cross-Attn         — remove robot→task cross-attention
  5. No Score Head         — raw dot-product readout
  6. L=1 (no depth)        — single SetBlock

Metrics per variant:
  - Training loss (per epoch)
  - Test cost ratio (model_cost / optimal_cost) on n=10, 20, 50
  - S_n-invariance max |Δ|

Outputs:
  - ablation_metrics.csv
  - ablation_loss_curves.png
  - ablation_cost_ratio.png
  - ablation_invariance.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ============================================================
# ARCHITECTURE VARIANTS
# ============================================================

class ISAB(nn.Module):
    def __init__(self, h, m=8, n_heads=4):
        super().__init__()
        self.m = m
        self.I = nn.Parameter(torch.randn(m, h) * 0.02)
        self.attn1 = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)

    def forward(self, X):
        I = self.I.unsqueeze(0)
        out1, _ = self.attn1(I, X.unsqueeze(0), X.unsqueeze(0))
        out1 = self.norm1(out1.squeeze(0))
        out2, _ = self.attn2(X.unsqueeze(0), out1.unsqueeze(0), out1.unsqueeze(0))
        return self.norm2(out2.squeeze(0))


class SetBlock(nn.Module):
    """SetBlock with optional Reynolds centroid."""
    def __init__(self, h, m=8, use_reynolds=True):
        super().__init__()
        self.use_reynolds = use_reynolds
        self.isab = ISAB(h, m)
        self.ffn = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Linear(4*h, h))
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        if use_reynolds:
            self.xattn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
            self.norm3 = nn.LayerNorm(h)

    def forward(self, H):
        H = self.norm1(H + self.isab(H))
        H = self.norm2(H + self.ffn(H))
        if self.use_reynolds:
            c = H.mean(dim=0, keepdim=True)
            H, _ = self.xattn(H, c, c)
            H = self.norm3(H)
        return H


class FleetPolicy(nn.Module):
    """Parameterized fleet policy for ablation."""
    def __init__(self, d, h=128, L=4, m=8, n_tasks=10,
                 use_reynolds=True, use_isab=True,
                 use_cross_attn=True, use_score_head=True):
        super().__init__()
        self.use_isab = use_isab
        self.use_cross_attn = use_cross_attn
        self.use_score_head = use_score_head

        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([SetBlock(h, m, use_reynolds=use_reynolds) for _ in range(L)])

        self.task_enc = nn.Sequential(nn.Linear(2, h), nn.GELU(), nn.Linear(h, h))

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
            self.final_norm = nn.LayerNorm(h)

        if use_score_head:
            self.score_head = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, h))

    def forward(self, X, tasks):
        H = self.enc(X)
        for blk in self.blocks:
            H = blk(H)
        T = self.task_enc(tasks)

        if self.use_cross_attn:
            H, _ = self.cross_attn(H, T, T)
            H = self.final_norm(H)

        if self.use_score_head:
            H = self.score_head(H)

        scores = H @ T.T
        return scores


# ============================================================
# PROBLEM GENERATION
# ============================================================

def generate_instance(n, m, d=4, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    robots = torch.randn(n, d)
    robots[:, 2] = torch.sigmoid(robots[:, 2])
    robots[:, 3] = torch.sigmoid(robots[:, 3])
    tasks = torch.randn(m, 2) * 2.0
    pos_r = robots[:, :2]
    dist = torch.cdist(pos_r, tasks)
    battery_factor = (1.5 - 0.5 * robots[:, 2:3])
    cost_matrix = dist * battery_factor
    row, col = linear_sum_assignment(cost_matrix.numpy())
    optimal_cost = cost_matrix[row, col].sum().item()
    return robots, tasks, cost_matrix, optimal_cost


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_variant(model, epochs=80, lr=1e-3, n_train_range=(5, 15), n_tasks=10, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 8
        for _ in range(n_batches):
            n = np.random.randint(n_train_range[0], n_train_range[1] + 1)
            robots, tasks, cost_matrix, _ = generate_instance(n, n_tasks, d=4)
            robots, tasks, cost_matrix = robots.to(device), tasks.to(device), cost_matrix.to(device)
            scores = model(robots, tasks)
            loss = F.mse_loss(scores, cost_matrix)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() / n_batches
        scheduler.step()
        loss_history.append(epoch_loss)

    return loss_history


def evaluate_variant(model, n_tasks=10, device='cpu'):
    """Evaluate cost ratio and invariance."""
    model.eval()
    results = {}

    # Cost ratio at multiple n
    for n_test in [10, 20, 50]:
        ratios = []
        for seed in range(20):
            robots, tasks, cost_matrix, optimal = generate_instance(n_test, n_tasks, d=4, seed=seed)
            robots, tasks = robots.to(device), tasks.to(device)
            with torch.no_grad():
                scores = model(robots, tasks)
            r, c = linear_sum_assignment(scores.cpu().numpy())
            model_cost = scores.cpu()[r, c].sum().item()
            ratios.append(model_cost / optimal if optimal > 0 else float('inf'))
        results[f'cost_ratio_n{n_test}'] = float(np.mean(ratios))

    # Invariance
    max_diff = 0
    for seed in range(50):
        n = np.random.randint(5, 16)
        robots, tasks, _, _ = generate_instance(n, n_tasks, d=4, seed=seed)
        robots, tasks = robots.to(device), tasks.to(device)
        with torch.no_grad():
            scores_orig = model(robots, tasks)
            perm = torch.randperm(n)
            scores_perm = model(robots[perm], tasks)
        diff = (scores_perm - scores_orig[perm]).abs().max().item()
        max_diff = max(max_diff, diff)
    results['invariance_max_diff'] = max_diff

    return results


# ============================================================
# MAIN ABLATION
# ============================================================

def run_ablation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    variants = {
        'Full (baseline)': dict(use_reynolds=True, use_isab=True, use_cross_attn=True, use_score_head=True, L=4),
        'No Reynolds':     dict(use_reynolds=False, use_isab=True, use_cross_attn=True, use_score_head=True, L=4),
        'No ISAB':         dict(use_reynolds=True, use_isab=False, use_cross_attn=True, use_score_head=True, L=4),
        'No Cross-Attn':   dict(use_reynolds=True, use_isab=True, use_cross_attn=False, use_score_head=True, L=4),
        'No Score Head':   dict(use_reynolds=True, use_isab=True, use_cross_attn=True, use_score_head=False, L=4),
        'L=1 (no depth)':  dict(use_reynolds=True, use_isab=True, use_cross_attn=True, use_score_head=True, L=1),
    }

    # For No ISAB, we need a special block
    class SetBlockNoISAB(nn.Module):
        def __init__(self, h, m=8, use_reynolds=True):
            super().__init__()
            self.use_reynolds = use_reynolds
            self.ffn = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Linear(4*h, h))
            self.norm1 = nn.LayerNorm(h)
            self.norm2 = nn.LayerNorm(h)
            if use_reynolds:
                self.xattn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
                self.norm3 = nn.LayerNorm(h)

        def forward(self, H):
            H = self.norm1(H + self.ffn(H))
            if self.use_reynolds:
                c = H.mean(dim=0, keepdim=True)
                H, _ = self.xattn(H, c, c)
                H = self.norm3(H)
            return H

    all_metrics = {}
    all_loss_histories = {}

    for name, params in variants.items():
        print(f"\n{'='*70}")
        print(f"  VARIANT: {name}")
        print(f"  Params: {params}")
        print(f"{'='*70}")

        d, h, m, n_tasks = 4, 128, 8, 10
        L = params.pop('L', 4)

        # Build model (special-case No ISAB)
        if params.get('use_isab') == False:
            model = nn.Module()
            model.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
            model.blocks = nn.ModuleList([SetBlockNoISAB(h, m, use_reynolds=params['use_reynolds']) for _ in range(L)])
            model.task_enc = nn.Sequential(nn.Linear(2, h), nn.GELU(), nn.Linear(h, h))
            if params['use_cross_attn']:
                model.cross_attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
                model.final_norm = nn.LayerNorm(h)
            if params['use_score_head']:
                model.score_head = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, h))
            model.use_cross_attn = params['use_cross_attn']
            model.use_score_head = params['use_score_head']

            def forward_fn(X, tasks, _m=model):
                H = _m.enc(X)
                for blk in _m.blocks:
                    H = blk(H)
                T = _m.task_enc(tasks)
                if _m.use_cross_attn:
                    H, _ = _m.cross_attn(H, T, T)
                    H = _m.final_norm(H)
                if _m.use_score_head:
                    H = _m.score_head(H)
                return H @ T.T
            model.forward = forward_fn
        else:
            model = FleetPolicy(d, h, L, m, n_tasks, **params)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Train
        t0 = time.time()
        loss_history = train_variant(model, epochs=80, device=device)
        train_time = time.time() - t0
        print(f"  Training: {train_time:.1f}s | Final loss: {loss_history[-1]:.6f}")

        # Evaluate
        metrics = evaluate_variant(model, device=device)
        metrics['n_params'] = n_params
        metrics['train_time_s'] = train_time
        metrics['final_loss'] = loss_history[-1]

        all_metrics[name] = metrics
        all_loss_histories[name] = loss_history

        print(f"  Cost ratio n=10: {metrics['cost_ratio_n10']:.4f}")
        print(f"  Cost ratio n=20: {metrics['cost_ratio_n20']:.4f}")
        print(f"  Cost ratio n=50: {metrics['cost_ratio_n50']:.4f}")
        print(f"  Invariance max|Δ|: {metrics['invariance_max_diff']:.2e}")

    # ============================================================
    # SAVE METRICS
    # ============================================================
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV
    import csv
    csv_path = os.path.join(out_dir, 'ablation_metrics.csv')
    fieldnames = ['variant', 'n_params', 'final_loss', 'train_time_s',
                  'cost_ratio_n10', 'cost_ratio_n20', 'cost_ratio_n50', 'invariance_max_diff']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in all_metrics.items():
            row = {'variant': name}
            for k in fieldnames[1:]:
                row[k] = m.get(k, '')
            writer.writerow(row)
    print(f"\nMetrics saved: {csv_path}")

    # JSON (full)
    json_path = os.path.join(out_dir, 'ablation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ============================================================
    # PLOTS
    # ============================================================
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#795548']
    for i, (name, hist) in enumerate(all_loss_histories.items()):
        ax.plot(hist, label=name, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Ablation: Training Loss Curves')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_loss_curves.png'))
    plt.close(fig)
    print(f"Plot saved: ablation_loss_curves.png")

    # Plot 2: Cost ratio bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    names = list(all_metrics.keys())
    x = np.arange(len(names))
    for ax, n_test, title in zip(axes, [10, 20, 50], ['n=10', 'n=20', 'n=50']):
        ratios = [all_metrics[nm][f'cost_ratio_n{n_test}'] for nm in names]
        bars = ax.bar(x, ratios, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([nm.replace(' (baseline)', '\n(baseline)').replace(' (no depth)', '\n(no depth)') for nm in names], rotation=30, ha='right', fontsize=8)
        ax.set_title(f'Cost Ratio ({title})')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Optimal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    axes[0].set_ylabel('Model Cost / Optimal Cost')
    fig.suptitle('Ablation: Assignment Cost Ratio by Fleet Size', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_cost_ratio.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: ablation_cost_ratio.png")

    # Plot 3: Invariance error
    fig, ax = plt.subplots(figsize=(8, 4))
    inv_errors = [all_metrics[nm]['invariance_max_diff'] for nm in names]
    bars = ax.bar(x, inv_errors, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([nm.replace(' (baseline)', '\n(baseline)').replace(' (no depth)', '\n(no depth)') for nm in names], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Max |Δ| (S_n-invariance)')
    ax.set_title('Ablation: Invariance Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_invariance.png'))
    plt.close(fig)
    print(f"Plot saved: ablation_invariance.png")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'Params':>10} {'Loss':>10} {'CR n=10':>10} {'CR n=50':>10} {'Inv Δ':>12}")
    print(f"{'-'*70}")
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['n_params']:>10,} {m['final_loss']:>10.6f} "
              f"{m['cost_ratio_n10']:>10.4f} {m['cost_ratio_n50']:>10.4f} "
              f"{m['invariance_max_diff']:>12.2e}")
    print(f"{'='*70}")

    return all_metrics, all_loss_histories


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_ablation()

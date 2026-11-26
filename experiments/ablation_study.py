#!/usr/bin/env python
"""
Ablation Study: Constrained-Operation Catalogue
===============================================
Two complementary axes:

  A. Reynolds × depth factorial (the causal test for depth-irrelevance)
       Reynolds × {L=1, L=4}
       No Reynolds × {L=1, L=4}

  B. Component ablations at L=4 (architecture contribution)
       No ISAB / No Cross-Attn / No Score Head

Cost ratio = true assignment cost under the model's predicted ranking
             divided by the optimal true cost  (NOT predicted-score sum).

Outputs (in results/):
  - ablation_metrics.{csv,json}
  - ablation_loss_curves.png
  - ablation_cost_ratio.png
  - ablation_factorial.png
  - ablation_invariance.png
"""

import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.architecture import FleetPolicy  # noqa: E402


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
        epoch_loss = 0.0
        n_batches = 8
        for _ in range(n_batches):
            n = np.random.randint(n_train_range[0], n_train_range[1] + 1)
            robots, tasks, cost_matrix, _ = generate_instance(n, n_tasks, d=4)
            robots = robots.to(device)
            tasks = tasks.to(device)
            cost_matrix = cost_matrix.to(device)
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
    """Evaluate true cost ratio and invariance.

    Cost ratio uses the *true* cost of the assignment induced by the model's
    scores, divided by the optimal true cost.  (Using the predicted-score sum
    would conflate score scale with assignment quality.)
    """
    model.eval()
    results = {}

    for n_test in [10, 20, 50]:
        ratios = []
        for seed in range(20):
            robots, tasks, cost_matrix, optimal = generate_instance(
                n_test, n_tasks, d=4, seed=seed
            )
            robots = robots.to(device)
            tasks = tasks.to(device)
            with torch.no_grad():
                scores = model(robots, tasks)
            r, c = linear_sum_assignment(scores.cpu().numpy())
            model_cost = cost_matrix[r, c].sum().item()
            ratios.append(model_cost / optimal if optimal > 0 else float('inf'))
        results[f'cost_ratio_n{n_test}'] = float(np.mean(ratios))

    max_diff = 0.0
    for seed in range(50):
        n = 5 + (seed % 11)  # deterministic n in [5, 15]
        robots, tasks, _, _ = generate_instance(n, n_tasks, d=4, seed=seed)
        robots = robots.to(device)
        tasks = tasks.to(device)
        with torch.no_grad():
            scores_orig = model(robots, tasks)
            # Fixed permutation for this seed (reproducible)
            g = torch.Generator().manual_seed(10_000 + seed)
            perm = torch.randperm(n, generator=g)
            scores_perm = model(robots[perm], tasks)
        diff = (scores_perm - scores_orig[perm]).abs().max().item()
        max_diff = max(max_diff, diff)
    results['invariance_max_diff'] = max_diff

    return results


# ============================================================
# VARIANT TABLE
# ============================================================

# Factorial cells first (causal depth test), then component ablations.
VARIANTS = {
    # --- Reynolds × depth factorial ---
    'Reynolds L=4': dict(
        use_reynolds=True, use_isab=True, use_cross_attn=True, use_score_head=True, L=4,
        axis='factorial', reynolds=True, depth=4,
    ),
    'Reynolds L=1': dict(
        use_reynolds=True, use_isab=True, use_cross_attn=True, use_score_head=True, L=1,
        axis='factorial', reynolds=True, depth=1,
    ),
    'No Reynolds L=4': dict(
        use_reynolds=False, use_isab=True, use_cross_attn=True, use_score_head=True, L=4,
        axis='factorial', reynolds=False, depth=4,
    ),
    'No Reynolds L=1': dict(
        use_reynolds=False, use_isab=True, use_cross_attn=True, use_score_head=True, L=1,
        axis='factorial', reynolds=False, depth=1,
    ),
    # --- Component ablations at L=4 (Reynolds on) ---
    'No ISAB': dict(
        use_reynolds=True, use_isab=False, use_cross_attn=True, use_score_head=True, L=4,
        axis='component',
    ),
    'No Cross-Attn': dict(
        use_reynolds=True, use_isab=True, use_cross_attn=False, use_score_head=True, L=4,
        axis='component',
    ),
    'No Score Head': dict(
        use_reynolds=True, use_isab=True, use_cross_attn=True, use_score_head=False, L=4,
        axis='component',
    ),
}

FACTORIAL_KEYS = [
    'Reynolds L=4', 'Reynolds L=1', 'No Reynolds L=4', 'No Reynolds L=1',
]


# ============================================================
# MAIN ABLATION
# ============================================================

def run_ablation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 72)
    print("Cost ratio = true cost(assignment from scores) / optimal true cost")
    print("=" * 72)

    all_metrics = {}
    all_loss_histories = {}
    meta = {}

    for name, raw in VARIANTS.items():
        params = {k: v for k, v in raw.items()
                  if k in ('use_reynolds', 'use_isab', 'use_cross_attn', 'use_score_head', 'L')}
        meta[name] = {k: v for k, v in raw.items() if k not in params}

        print(f"\n{'=' * 72}")
        print(f"  VARIANT: {name}")
        print(f"  Params: {params}")
        print(f"{'=' * 72}")

        d, h, m, n_tasks = 4, 128, 8, 10
        L = params.pop('L', 4)
        model = FleetPolicy(
            d, h=h, L=L, m=m, d_task=2, score_mode="element", **params
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        loss_history = train_variant(model, epochs=80, device=device)
        train_time = time.time() - t0
        print(f"  Training: {train_time:.1f}s | Final loss: {loss_history[-1]:.6f}")

        metrics = evaluate_variant(model, device=device)
        metrics['n_params'] = n_params
        metrics['train_time_s'] = train_time
        metrics['final_loss'] = loss_history[-1]
        metrics['axis'] = meta[name].get('axis', '')
        metrics['reynolds'] = meta[name].get('reynolds', params.get('use_reynolds'))
        metrics['depth'] = meta[name].get('depth', L)

        all_metrics[name] = metrics
        all_loss_histories[name] = loss_history

        print(f"  Cost ratio n=10: {metrics['cost_ratio_n10']:.4f}")
        print(f"  Cost ratio n=20: {metrics['cost_ratio_n20']:.4f}")
        print(f"  Cost ratio n=50: {metrics['cost_ratio_n50']:.4f}")
        print(f"  Invariance max|Δ|: {metrics['invariance_max_diff']:.2e}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(out_dir, 'ablation_metrics.csv')
    fieldnames = [
        'variant', 'axis', 'reynolds', 'depth', 'n_params', 'final_loss', 'train_time_s',
        'cost_ratio_n10', 'cost_ratio_n20', 'cost_ratio_n50', 'invariance_max_diff',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in all_metrics.items():
            row = {'variant': name}
            for k in fieldnames[1:]:
                row[k] = m.get(k, '')
            writer.writerow(row)
    print(f"\nMetrics saved: {csv_path}")

    # --- JSON ---
    json_path = os.path.join(out_dir, 'ablation_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"JSON saved: {json_path}")

    # --- Factorial deltas (primary claim check) ---
    factorial = _factorial_summary(all_metrics)
    fac_path = os.path.join(out_dir, 'ablation_factorial.json')
    with open(fac_path, 'w') as f:
        json.dump(factorial, f, indent=2)
    print(f"Factorial summary: {fac_path}")

    _plot_all(all_metrics, all_loss_histories, factorial, out_dir)
    _print_summary(all_metrics, factorial)

    return all_metrics, all_loss_histories, factorial


def _factorial_summary(all_metrics):
    """ΔCR = CR(L=4) − CR(L=1); depth-irrelevant ⇒ |ΔCR| ≈ 0."""
    def cr(name, n):
        return all_metrics[name][f'cost_ratio_n{n}']

    summary = {'by_n': {}, 'interpretation': {}}
    for n in (10, 20, 50):
        d_rey = cr('Reynolds L=4', n) - cr('Reynolds L=1', n)
        d_norey = cr('No Reynolds L=4', n) - cr('No Reynolds L=1', n)
        summary['by_n'][str(n)] = {
            'reynolds_L4': cr('Reynolds L=4', n),
            'reynolds_L1': cr('Reynolds L=1', n),
            'reynolds_delta': d_rey,
            'no_reynolds_L4': cr('No Reynolds L=4', n),
            'no_reynolds_L1': cr('No Reynolds L=1', n),
            'no_reynolds_delta': d_norey,
        }
    # Lead with n=50 (headline).
    # With Reynolds we expect |Δ|≈0 (depth irrelevant).
    # Without Reynolds we only claim that depth is free to move CR
    # (|Δ| large) — the sign depends on optimization, not the theorem.
    d50 = summary['by_n']['50']
    summary['interpretation'] = {
        'reynolds_depth_irrelevant': abs(d50['reynolds_delta']) < 0.05,
        'no_reynolds_depth_matters': abs(d50['no_reynolds_delta']) >= 0.05,
        'headline_n50': d50,
    }
    return summary


def _short_label(name):
    return (name
            .replace('Reynolds ', 'Rey ')
            .replace('No Reynolds ', 'NoRey ')
            .replace('No Cross-Attn', 'No XAttn')
            .replace('No Score Head', 'No Score'))


def _plot_all(all_metrics, all_loss_histories, factorial, out_dir):
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
    colors = ['#2196F3', '#1565C0', '#F44336', '#C62828', '#4CAF50', '#FF9800', '#9C27B0']
    names = list(all_metrics.keys())

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, hist) in enumerate(all_loss_histories.items()):
        ax.plot(hist, label=_short_label(name), color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Ablation: Training Loss Curves')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_loss_curves.png'))
    plt.close(fig)
    print("Plot saved: ablation_loss_curves.png")

    # Plot 2: Cost ratio — all variants
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    x = np.arange(len(names))
    for ax, n_test in zip(axes, [10, 20, 50]):
        ratios = [all_metrics[nm][f'cost_ratio_n{n_test}'] for nm in names]
        ax.bar(x, ratios, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(nm) for nm in names], rotation=35, ha='right', fontsize=7)
        ax.set_title(f'Cost Ratio (n={n_test})')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Optimal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    axes[0].set_ylabel('True model cost / optimal cost')
    fig.suptitle('Ablation: Assignment Cost Ratio (corrected metric)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_cost_ratio.png'), bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: ablation_cost_ratio.png")

    # Plot 3: Factorial 2×2 grouped bars
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    width = 0.35
    for ax, n_test in zip(axes, [10, 20, 50]):
        cell = factorial['by_n'][str(n_test)]
        rey = [cell['reynolds_L1'], cell['reynolds_L4']]
        norey = [cell['no_reynolds_L1'], cell['no_reynolds_L4']]
        xpos = np.array([0, 1])
        ax.bar(xpos - width / 2, rey, width, label='Reynolds', color='#2196F3', edgecolor='black', linewidth=0.5)
        ax.bar(xpos + width / 2, norey, width, label='No Reynolds', color='#F44336', edgecolor='black', linewidth=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(['L=1', 'L=4'])
        ax.set_title(f'n={n_test}')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        # Annotate deltas
        ax.annotate(f"Δ={cell['reynolds_delta']:+.3f}",
                    xy=(0.5, max(rey) + 0.05), ha='center', fontsize=8, color='#1565C0')
        ax.annotate(f"Δ={cell['no_reynolds_delta']:+.3f}",
                    xy=(0.5, max(norey) + 0.15), ha='center', fontsize=8, color='#C62828')
    axes[0].set_ylabel('True model cost / optimal cost')
    axes[2].legend(loc='upper right', fontsize=9)
    fig.suptitle('Reynolds × Depth Factorial (Δ = CR(L=4) − CR(L=1))', fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_factorial.png'), bbox_inches='tight')
    plt.close(fig)
    print("Plot saved: ablation_factorial.png")

    # Plot 4: Invariance
    fig, ax = plt.subplots(figsize=(9, 4))
    inv_errors = [all_metrics[nm]['invariance_max_diff'] for nm in names]
    ax.bar(x, inv_errors, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(nm) for nm in names], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Max |Δ| (S_n-invariance)')
    ax.set_title('Ablation: Invariance Error')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ablation_invariance.png'))
    plt.close(fig)
    print("Plot saved: ablation_invariance.png")

    # Mirror hero plots to repo root for casual browsing
    try:
        import shutil
        for fname in ('ablation_cost_ratio.png', 'ablation_factorial.png'):
            shutil.copy2(os.path.join(out_dir, fname),
                         os.path.join(out_dir, '..', fname))
    except OSError:
        pass


def _print_summary(all_metrics, factorial):
    print(f"\n{'=' * 72}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Variant':<20} {'Params':>10} {'Loss':>10} {'CR n=10':>10} {'CR n=50':>10} {'Inv Δ':>12}")
    print("-" * 72)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['n_params']:>10,} {m['final_loss']:>10.6f} "
              f"{m['cost_ratio_n10']:>10.4f} {m['cost_ratio_n50']:>10.4f} "
              f"{m['invariance_max_diff']:>12.2e}")

    print(f"\n{'=' * 72}")
    print("REYNOLDS × DEPTH FACTORIAL  (ΔCR = CR(L=4) − CR(L=1))")
    print(f"{'=' * 72}")
    print(f"{'Setting':<16} {'CR L=1':>10} {'CR L=4':>10} {'ΔCR n=50':>12}")
    print("-" * 52)
    h = factorial['by_n']['50']
    print(f"{'Reynolds':<16} {h['reynolds_L1']:>10.4f} {h['reynolds_L4']:>10.4f} "
          f"{h['reynolds_delta']:>+12.4f}")
    print(f"{'No Reynolds':<16} {h['no_reynolds_L1']:>10.4f} {h['no_reynolds_L4']:>10.4f} "
          f"{h['no_reynolds_delta']:>+12.4f}")
    interp = factorial['interpretation']
    print()
    print(f"  With Reynolds:    |ΔCR|={'small' if interp['reynolds_depth_irrelevant'] else 'large'} "
          f"→ depth {'irrelevant' if interp['reynolds_depth_irrelevant'] else 'matters'}")
    sign = 'hurts' if h['no_reynolds_delta'] > 0.05 else (
        'helps' if h['no_reynolds_delta'] < -0.05 else 'neutral')
    print(f"  Without Reynolds: |ΔCR|={'large' if interp['no_reynolds_depth_matters'] else 'small'} "
          f"(depth {sign} at n=50) — depth is free to move quality")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_ablation()

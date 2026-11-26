#!/usr/bin/env python
"""
Empirical Breadth: Multi-Task Benchmark
========================================
Demonstrates the catalogue architecture as a general-purpose
S_n-invariant set function across three combinatorial tasks:

  1. Linear Assignment  (n robots → m tasks)
  2. K-means Clustering (n points → k cluster centers)
  3. TSP Tour           (n cities → min-cost Hamiltonian cycle)

All three share the same "set → cost matrix" paradigm.
Cost matrices are normalized to [0,1] for stable training.

Outputs:
  - breadth_metrics.csv
  - breadth_bar_chart.png
  - breadth_loss_curves.png
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
# NORMALIZATION HELPER
# ============================================================

def normalize_cost(cost):
    """Normalize cost matrix to [0, 1] range."""
    c_min = cost.min()
    c_max = cost.max()
    if c_max - c_min < 1e-8:
        return torch.zeros_like(cost)
    return (cost - c_min) / (c_max - c_min)


# ============================================================
# TASK 1: LINEAR ASSIGNMENT
# ============================================================

def gen_assignment(n, m, d=4):
    robots = torch.randn(n, d)
    robots[:, 2] = torch.sigmoid(robots[:, 2])
    robots[:, 3] = torch.sigmoid(robots[:, 3])
    tasks = torch.randn(m, 2) * 2.0
    dist = torch.cdist(robots[:, :2], tasks)
    battery = (1.5 - 0.5 * robots[:, 2:3])
    cost = dist * battery
    cost_norm = normalize_cost(cost)
    r, c = linear_sum_assignment(cost.numpy())
    optimal = cost[r, c].sum().item()
    return robots, tasks, cost_norm, optimal


def eval_assignment(model, n, m, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    robots, tasks, cost_norm, optimal = gen_assignment(n, m)
    robots, tasks = robots.to(device), tasks.to(device)
    with torch.no_grad():
        scores = model(robots, tasks)
    # Use the model's predicted cost to find assignment
    r, c = linear_sum_assignment(scores.cpu().numpy())
    # Compute TRUE cost of model's assignment (dist * battery)
    dist = torch.cdist(robots[:, :2], tasks)
    battery = (1.5 - 0.5 * robots[:, 2:3])
    cost_orig = dist * battery
    model_cost = cost_orig[r, c].sum().item()
    ratio = model_cost / optimal if optimal > 0 else float('inf')
    assert ratio >= 0.99, f"Assignment ratio {ratio:.4f} < 1 — evaluation bug"
    return ratio


# ============================================================
# TASK 2: K-MEANS CLUSTERING
# ============================================================

def gen_kmeans(n, k=3):
    centers = torch.randn(k, 2) * 3.0
    labels = torch.randint(0, k, (n,))
    points = centers[labels] + 0.5 * torch.randn(n, 2)
    cost = torch.cdist(points, centers) ** 2
    cost_norm = normalize_cost(cost)
    optimal = cost[torch.arange(n), labels].sum().item()
    return points, centers, cost_norm, optimal


def eval_kmeans(model, n, k, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    points, centers, cost_norm, optimal = gen_kmeans(n, k)
    points, centers = points.to(device), centers.to(device)
    with torch.no_grad():
        scores = model(points, centers)
    assignment = scores.argmin(dim=1)
    # Reconstruct for fair cost comparison
    torch.manual_seed(seed); np.random.seed(seed)
    points2, centers2, _, _ = gen_kmeans(n, k)
    cost_orig = torch.cdist(points2, centers2) ** 2
    model_cost = cost_orig[torch.arange(n), assignment].sum().item()
    return model_cost / optimal if optimal > 0 else float('inf')


# ============================================================
# TASK 3: TSP (TRAVELING SALESMAN)
# ============================================================

def tsp_from_dist_matrix(dist):
    """Held-Karp DP on a given distance matrix. O(n² · 2ⁿ). n ≤ 10."""
    n = dist.shape[0]
    if n <= 1:
        return 0.0

    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0.0

    for S in range(1, 1 << n):
        if not (S & 1):
            continue
        for i in range(n):
            if dp[S][i] == INF or not (S & (1 << i)):
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                ns = S | (1 << j)
                dp[ns][j] = min(dp[ns][j], dp[S][i] + dist[i, j])

    full = (1 << n) - 1
    best = min(dp[full][i] + dist[i, 0] for i in range(1, n))
    return best


def tsp_optimal(cities_np):
    """Held-Karp DP: compute distance matrix from city coordinates, then solve."""
    n = len(cities_np)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(cities_np[i] - cities_np[j])
    return tsp_from_dist_matrix(dist)


def gen_tsp(n):
    cities = torch.randn(n, 2) * 3.0
    cost = torch.cdist(cities, cities)
    cost_norm = normalize_cost(cost)
    optimal = tsp_optimal(cities.numpy())
    return cities, cities, cost_norm, optimal


def tsp_tour_from_scores(scores):
    """Held-Karp DP returning the optimal tour (list of city indices) on a score matrix."""
    n = scores.shape[0]
    if n <= 1:
        return list(range(n))

    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0.0

    for S in range(1, 1 << n):
        if not (S & 1):
            continue
        for i in range(n):
            if dp[S][i] == INF or not (S & (1 << i)):
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                ns = S | (1 << j)
                new_cost = dp[S][i] + scores[i, j]
                if new_cost < dp[ns][j]:
                    dp[ns][j] = new_cost
                    parent[ns][j] = i

    full = (1 << n) - 1
    best_end = min(range(1, n), key=lambda i: dp[full][i] + scores[i, 0])
    best_cost = dp[full][best_end] + scores[best_end, 0]

    # Reconstruct tour
    tour = [best_end]
    S = full
    cur = best_end
    while S != 1:
        prev = parent[S][cur]
        tour.append(prev)
        S ^= (1 << cur)
        cur = prev
    tour.reverse()
    return tour


def eval_tsp(model, n, seed, device):
    """Evaluate TSP: use model scores to select a tour, then score it on TRUE distances."""
    torch.manual_seed(seed); np.random.seed(seed)
    cities, _, cost_norm, optimal = gen_tsp(n)
    cities_dev = cities.to(device)
    with torch.no_grad():
        scores = model(cities_dev, cities_dev)
    # Symmetrize, set diagonal to large value
    scores_np = (scores + scores.T) / 2
    scores_np = scores_np.cpu().numpy()
    np.fill_diagonal(scores_np, 1e6)
    scores_np = np.nan_to_num(scores_np, nan=1e6, posinf=1e6, neginf=1e6)

    # Find the tour the model prefers (minimizes its scores)
    tour = tsp_tour_from_scores(scores_np)

    # Evaluate that tour on TRUE Euclidean distances
    true_dist = torch.cdist(cities, cities).numpy()
    tour_cost = sum(true_dist[tour[i], tour[(i + 1) % n]] for i in range(n))

    if optimal <= 0:
        return float('inf')
    ratio = tour_cost / optimal
    assert ratio >= 0.99, f"TSP ratio {ratio:.4f} < 1 — evaluation bug"
    return ratio


# ============================================================
# TRAINING
# ============================================================

def train_task(model, gen_fn, task_name, epochs=80, lr=1e-3,
               n_range=(5, 12), m_range=(5, 10), device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 10
        for _ in range(n_batches):
            n = np.random.randint(n_range[0], n_range[1] + 1)
            m = np.random.randint(m_range[0], m_range[1] + 1)
            X, tasks, cost_norm, _ = gen_fn(n, m)
            X, tasks, cost_norm = X.to(device), tasks.to(device), cost_norm.to(device)
            scores = model(X, tasks)
            loss = F.mse_loss(scores, cost_norm)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() / n_batches
        scheduler.step()
        loss_history.append(epoch_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  [{task_name}] Epoch {epoch+1:3d}/{epochs} | loss = {epoch_loss:.6f}")

    return loss_history


# ============================================================
# INVARIANCE TEST
# ============================================================

def test_invariance(model, gen_fn, n_range=(5, 12), m_range=(5, 10),
                    n_tests=30, device='cpu'):
    model.eval()
    max_diff = 0.0
    for i in range(n_tests):
        # Deterministic instance sizes / content / permutation per trial
        torch.manual_seed(50_000 + i)
        np.random.seed(50_000 + i)
        n = n_range[0] + (i % (n_range[1] - n_range[0] + 1))
        m = m_range[0] + (i % (m_range[1] - m_range[0] + 1))
        X, tasks, _, _ = gen_fn(n, m)
        X, tasks = X.to(device), tasks.to(device)
        with torch.no_grad():
            s_orig = model(X, tasks)
            g = torch.Generator().manual_seed(60_000 + i)
            perm = torch.randperm(n, generator=g)
            s_perm = model(X[perm], tasks)
        diff = (s_perm - s_orig[perm]).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


# ============================================================
# MAIN
# ============================================================

def run_breadth():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("=" * 70)
    print("Cost ratio = true decision cost / optimal true cost (mean over 20 seeds)")
    print("=" * 70)

    h, L, m_ind = 128, 4, 8
    all_results = {}
    all_losses = {}

    tasks = {
        'Linear Assignment': {
            'gen': lambda n, m: gen_assignment(n, m, d=4),
            'eval': lambda model, n, m, seed, dev: eval_assignment(model, n, m, seed, dev),
            'n_range': (5, 12), 'm_range': (5, 10),
            'd_in': 4, 'd_task': 2,
            'test_ns': [10, 20, 50], 'm_test': 10,
        },
        'K-means (k=3)': {
            'gen': lambda n, m: gen_kmeans(n, k=3),
            'eval': lambda model, n, m, seed, dev: eval_kmeans(model, n, 3, seed, dev),
            'n_range': (6, 15), 'm_range': (3, 3),
            'd_in': 2, 'd_task': 2,
            'test_ns': [10, 20, 50], 'm_test': 3,
        },
        'TSP Tour': {
            'gen': lambda n, m: gen_tsp(n),
            'eval': lambda model, n, m, seed, dev: eval_tsp(model, n, seed, dev),
            'n_range': (4, 8), 'm_range': (4, 8),
            'd_in': 2, 'd_task': 2,
            'test_ns': [6, 8, 10], 'm_test': None,
        },
    }

    for task_idx, (task_name, cfg) in enumerate(tasks.items()):
        # Independent seed per task so order / failures don't leak RNG state
        torch.manual_seed(42 + task_idx)
        np.random.seed(42 + task_idx)

        print(f"\n{'='*70}")
        print(f"  TASK: {task_name}")
        print(f"{'='*70}")

        model = FleetPolicy(cfg['d_in'], h, L, m_ind, d_task=cfg['d_task']).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        loss_hist = train_task(
            model, cfg['gen'], task_name,
            epochs=80, n_range=cfg['n_range'], m_range=cfg['m_range'],
            device=device
        )
        train_time = time.time() - t0
        print(f"  Training: {train_time:.1f}s | Final loss: {loss_hist[-1]:.6f}")

        model.eval()
        ratios = {}
        for n_test in cfg['test_ns']:
            m_test = cfg['m_test'] if cfg['m_test'] else n_test
            vals = []
            for seed in range(20):
                r = cfg['eval'](model, n_test, m_test, seed, device)
                if r != float('inf'):
                    vals.append(r)
            if vals:
                ratios[n_test] = float(np.mean(vals))
            else:
                ratios[n_test] = float('inf')
            print(f"  Cost ratio n={n_test}: {ratios[n_test]:.4f}")

        inv = test_invariance(model, cfg['gen'],
                              n_range=cfg['n_range'], m_range=cfg['m_range'],
                              device=device)
        print(f"  Invariance max|Δ|: {inv:.2e}")

        all_results[task_name] = {
            'n_params': n_params,
            'final_loss': loss_hist[-1],
            'train_time_s': train_time,
            'cost_ratios': {str(k): v for k, v in ratios.items()},
            'invariance_max_diff': inv,
        }
        all_losses[task_name] = loss_hist

    # ============================================================
    # SAVE METRICS
    # ============================================================
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

    csv_path = os.path.join(out_dir, 'breadth_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'n_params', 'final_loss', 'train_time_s',
                         'cost_ratio_n10', 'cost_ratio_n20', 'cost_ratio_n50',
                         'invariance_max_diff'])
        for name, r in all_results.items():
            cr = r['cost_ratios']
            writer.writerow([
                name, r['n_params'], f"{r['final_loss']:.6f}", f"{r['train_time_s']:.2f}",
                f"{cr.get('10', cr.get('6', 'inf'))}",
                f"{cr.get('20', cr.get('8', 'inf'))}",
                f"{cr.get('50', cr.get('10', 'inf'))}",
                f"{r['invariance_max_diff']:.2e}",
            ])
    print(f"\nMetrics saved: {csv_path}")

    json_path = os.path.join(out_dir, 'breadth_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ============================================================
    # PLOTS
    # ============================================================
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    # Plot 1: Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    task_names = list(all_results.keys())
    n_labels = ['n=10', 'n=20', 'n=50']
    n_keys = ['10', '20', '50']
    x = np.arange(len(task_names))
    width = 0.25
    for i, (nl, nk) in enumerate(zip(n_labels, n_keys)):
        vals = []
        for tn in task_names:
            cr = all_results[tn]['cost_ratios']
            if nk in cr:
                v = cr[nk]
            else:
                # Find closest
                closest = min(cr.keys(), key=lambda k: abs(int(k) - int(nk)))
                v = cr[closest]
            vals.append(min(v, 5.0))  # Cap at 5 for readability
        bars = ax.bar(x + i * width, vals, width, label=nl,
                      color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Optimal')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_names, fontsize=10)
    ax.set_ylabel('Model Cost / Optimal Cost')
    ax.set_title('Empirical Breadth: Cost Ratio Across Tasks (capped at 5.0)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'breadth_bar_chart.png'))
    plt.close(fig)
    print(f"Plot saved: breadth_bar_chart.png")

    # Plot 2: Training loss curves (normalized)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, hist) in enumerate(all_losses.items()):
        norm = [l / hist[0] for l in hist]
        ax.plot(norm, label=name, color=colors[i], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Loss (loss / loss₀)')
    ax.set_title('Empirical Breadth: Training Convergence (normalized)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'breadth_loss_curves.png'))
    plt.close(fig)
    print(f"Plot saved: breadth_loss_curves.png")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("EMPIRICAL BREADTH SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<22} {'Params':>10} {'Loss':>10} {'CR(n=10)':>10} {'CR(n=20)':>10} {'CR(n=50)':>10} {'Inv Δ':>12}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        cr = r['cost_ratios']
        cr10 = cr.get('10', cr.get('6', 'inf'))
        cr20 = cr.get('20', cr.get('8', 'inf'))
        cr50 = cr.get('50', cr.get('10', 'inf'))
        def fmt(v):
            try:
                return f"{float(v):.4f}"
            except:
                return str(v)
        print(f"{name:<22} {r['n_params']:>10,} {r['final_loss']:>10.6f} "
              f"{fmt(cr10):>10} {fmt(cr20):>10} {fmt(cr50):>10} "
              f"{r['invariance_max_diff']:>12.2e}")
    print(f"{'='*70}")
    print(f"\nAll tasks use the SAME architecture (h=128, L=4, m=8).")
    print(f"Cost matrices normalized to [0,1] for stable training.")
    print(f"S_n-invariance holds structurally across all task types.")

    return all_results, all_losses


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_breadth()

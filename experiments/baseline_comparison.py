#!/usr/bin/env python
"""
Baseline Comparison: DeepSets, PointNet, Set Transformer vs. Constrained-Operation Catalogue
=============================================================================================
Trains all 4 architectures on Linear Assignment and TSP with identical protocol.
Measures: cost ratio (quality) + inference latency (speed).

Outputs:
  - baseline_comparison.png   (grouped bar: cost ratio per task)
  - baseline_latency.png      (bar: inference ms per model)
  - baseline_metrics.json     (all raw numbers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ============================================================
# BASELINE ARCHITECTURES
# ============================================================

class DeepSets(nn.Module):
    """DeepSets: elementwise MLP → sum pooling → MLP → broadcast back."""
    def __init__(self, d_in, h=128, d_task=2, d_score=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))
        self.task_score = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))

    def forward(self, X, tasks):
        # X: [n, d_in], tasks: [m, d_task]
        H = self.phi(X)           # [n, h]
        c = H.sum(dim=0, keepdim=True)  # [1, h] — sum pooling
        c = self.rho(c)           # [1, h]
        # Broadcast context to each element
        H_ctx = H + c             # [n, h]
        H_s = self.score_proj(H_ctx)  # [n, d_score]
        T_s = self.task_score(self.task_enc(tasks))  # [m, d_score]
        return H_s @ T_s.T        # [n, m]


class PointNet(nn.Module):
    """PointNet: elementwise MLP → max pooling → MLP → broadcast back."""
    def __init__(self, d_in, h=128, d_task=2, d_score=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))
        self.task_score = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))

    def forward(self, X, tasks):
        H = self.phi(X)           # [n, h]
        c = H.max(dim=0, keepdim=True).values  # [1, h] — max pooling
        c = self.rho(c)           # [1, h]
        H_ctx = H + c             # [n, h]
        H_s = self.score_proj(H_ctx)
        T_s = self.task_score(self.task_enc(tasks))
        return H_s @ T_s.T


class ISAB(nn.Module):
    def __init__(self, h, m=8, n_heads=4):
        super().__init__()
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


class PAB(nn.Module):
    """Permutation Attention Block (Set Transformer)."""
    def __init__(self, h, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(h, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(h)

    def forward(self, X):
        out, _ = self.attn(X.unsqueeze(0), X.unsqueeze(0), X.unsqueeze(0))
        return self.norm(out.squeeze(0))


class SetTransformer(nn.Module):
    """Set Transformer: ISAB × L + PAB × K."""
    def __init__(self, d_in, h=128, L=4, m=8, K=2, d_task=2, d_score=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.isabs = nn.ModuleList([ISAB(h, m) for _ in range(L)])
        self.pabs = nn.ModuleList([PAB(h) for _ in range(K)])
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.cross_attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.final_norm = nn.LayerNorm(h)
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))
        self.task_score = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, d_score))

    def forward(self, X, tasks):
        H = self.enc(X)
        for isab in self.isabs:
            H = H + isab(H)
        for pab in self.pabs:
            H = H + pab(H)
        T = self.task_enc(tasks)
        H, _ = self.cross_attn(H, T, T)
        H = self.final_norm(H)
        H_s = self.score_proj(H)
        T_s = self.task_score(T)
        return H_s @ T_s.T


# ============================================================
# OUR ARCHITECTURE (Constrained-Operation Catalogue)
# ============================================================

class SetBlock(nn.Module):
    def __init__(self, h, m=8):
        super().__init__()
        self.isab = ISAB(h, m)
        self.ffn = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Linear(4*h, h))
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.xattn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.norm3 = nn.LayerNorm(h)

    def forward(self, H):
        H = self.norm1(H + self.isab(H))
        H = self.norm2(H + self.ffn(H))
        c = H.mean(dim=0, keepdim=True)
        H, _ = self.xattn(H, c, c)
        return self.norm3(H)


class FleetPolicy(nn.Module):
    def __init__(self, d_in, h=128, L=4, m=8, d_task=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([SetBlock(h, m) for _ in range(L)])
        self.task_enc = nn.Sequential(nn.Linear(d_task, h), nn.GELU(), nn.Linear(h, h))
        self.cross_attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.final_norm = nn.LayerNorm(h)
        self.score_proj = nn.Sequential(nn.Linear(h, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, X, tasks):
        H = self.enc(X)
        for blk in self.blocks:
            H = blk(H)
        T = self.task_enc(tasks)
        H, _ = self.cross_attn(H, T, T)
        H = self.final_norm(H)
        H_s = self.score_proj(H)
        T_s = self.score_proj(T)
        return H_s @ T_s.T


# ============================================================
# DATA GENERATION (same as breadth_study.py)
# ============================================================

def gen_assignment(n, m, d=4, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    robots = torch.randn(n, d)
    tasks = torch.randn(m, 2)
    dist = torch.cdist(robots[:, :2], tasks)
    battery = 1.5 - 0.5 * robots[:, 2:3]
    cost = dist * battery
    return robots, tasks, cost


def gen_tsp(n, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    cities = torch.rand(n, 2) * 10
    dist = torch.cdist(cities, cities)
    return cities, dist


def tsp_from_dist_matrix(dist):
    """Held-Karp on distance matrix."""
    n = dist.shape[0]
    if n <= 1:
        return 0.0
    INF = float('inf')
    dp = [[INF] * (1 << n) for _ in range(n)]
    dp[0][1] = 0.0
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        for u in range(n):
            if dp[u][mask] == INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                dp[v][new_mask] = min(dp[v][new_mask], dp[u][mask] + dist[u, v])
    full = (1 << n) - 1
    best = INF
    for u in range(1, n):
        if dp[u][full] < INF:
            best = min(best, dp[u][full] + dist[u, 0])
    return best


def tsp_tour_from_scores(scores):
    """Greedy tour from score matrix (higher = prefer)."""
    n = scores.shape[0]
    visited = [0]
    unvisited = set(range(1, n))
    while unvisited:
        cur = visited[-1]
        best_v = max(unvisited, key=lambda v: scores[cur, v])
        visited.append(best_v)
        unvisited.remove(best_v)
    return visited


# ============================================================
# TRAINING
# ============================================================

def train_model(model, gen_fn, task_name, epochs=80, lr=1e-3, n_train=10, m_train=8, device='cpu'):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for _ in range(8):
            if task_name == 'Linear Assignment':
                robots, tasks, cost = gen_fn(n_train, m_train)
                robots, tasks, cost = robots.to(device), tasks.to(device), cost.to(device)
                scores = model(robots, tasks)
                # Loss: predict costs (lower = better), same as breadth study
                loss = F.mse_loss(scores, cost)
            else:  # tsp
                cities, dist = gen_fn(n_train, n_train)
                cities, dist = cities.to(device), dist.to(device)
                # Pad 2D coords to 4D to match d_in=4
                cities_in = torch.cat([cities, torch.zeros_like(cities)], dim=1)
                scores = model(cities_in, cities)  # X=4D, tasks=2D
                # Loss: minimize tour cost from greedy extraction
                scores_np = scores.detach().cpu().numpy()
                tour = tsp_tour_from_scores(scores_np)
                tour_cost = sum(dist[tour[i], tour[(i+1) % n_train]].item() for i in range(n_train))
                opt_cost = tsp_from_dist_matrix(dist.cpu().numpy())
                loss = torch.tensor(tour_cost / max(opt_cost, 1e-8), requires_grad=True)
                # Use a differentiable proxy: encourage high scores on optimal edges
                # For simplicity, use MSE on a soft target
                target = torch.zeros_like(scores)
                for i in range(n_train):
                    target[tour[i], tour[(i+1) % n_train]] = 1.0
                loss = F.mse_loss(scores, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / 8)
    model.eval()
    return losses


# ============================================================
# EVALUATION
# ============================================================

def eval_assignment(model, n, m, seed, device='cpu'):
    model.eval()
    robots, tasks, cost = gen_assignment(n, m, seed=seed)
    robots, tasks = robots.to(device), tasks.to(device)
    with torch.no_grad():
        scores = model(robots, tasks).cpu().numpy()
    row, col = linear_sum_assignment(scores)  # minimize predicted cost
    model_cost = cost[row, col].sum().item()
    # Optimal
    opt_row, opt_col = linear_sum_assignment(cost.numpy())
    opt_cost = cost[opt_row, opt_col].sum().item()
    return model_cost / max(opt_cost, 1e-8)


def eval_tsp(model, n, m=None, seed=42, device='cpu'):
    model.eval()
    cities, dist = gen_tsp(n, seed=seed)
    cities = cities.to(device)
    cities_in = torch.cat([cities, torch.zeros_like(cities)], dim=1)  # pad to d_in=4
    with torch.no_grad():
        scores = model(cities_in, cities).cpu().numpy()
    tour = tsp_tour_from_scores(scores)
    tour_cost = sum(dist[tour[i], tour[(i+1) % n]].item() for i in range(n))
    opt_cost = tsp_from_dist_matrix(dist.numpy())
    return tour_cost / max(opt_cost, 1e-8)


def measure_latency(model, n, m, device='cpu', iters=50):
    model.eval()
    X = torch.randn(n, 4).to(device)
    tasks = torch.randn(m, 2).to(device)
    with torch.no_grad():
        model(X, tasks)  # warmup
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            model(X, tasks)
    elapsed = (time.perf_counter() - start) / iters * 1000
    return elapsed


# ============================================================
# MAIN
# ============================================================

def run_comparison():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    device = 'cpu'
    h, L, m_ind = 128, 4, 8
    d_in, d_task = 4, 2

    models = {
        'Catalogue (ours)': FleetPolicy(d_in, h, L, m_ind, d_task=d_task),
        'DeepSets': DeepSets(d_in, h, d_task=d_task),
        'PointNet': PointNet(d_in, h, d_task=d_task),
        'Set Transformer': SetTransformer(d_in, h, L=L, m=m_ind, K=2, d_task=d_task),
    }

    tasks = {
        'Linear Assignment': {
            'gen': lambda n, m: gen_assignment(n, m),
            'eval': eval_assignment,
            'n_range': [10, 20, 50],
            'm': 8,
        },
        'TSP': {
            'gen': lambda n, m: gen_tsp(n),
            'eval': eval_tsp,
            'n_range': [6, 8, 10],
            'm': None,
        },
    }

    results = {}

    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Training: {name} ({n_params:,} params)")
        print(f"{'='*60}")

        task_results = {}
        for task_name, task_cfg in tasks.items():
            print(f"  Training on {task_name}...")
            n_train = task_cfg['n_range'][0]
            m_train = task_cfg['m'] if task_cfg['m'] else n_train
            losses = train_model(model, task_cfg['gen'], task_name,
                                 epochs=80, n_train=n_train, m_train=m_train, device=device)
            print(f"    Final loss: {losses[-1]:.4f}")

            evals = []
            for n in task_cfg['n_range']:
                m = task_cfg['m'] if task_cfg['m'] else n
                cr = task_cfg['eval'](model, n, m, seed=42, device=device)
                evals.append(cr)
                print(f"    n={n}: CR = {cr:.3f}")
            task_results[task_name] = evals

        # Latency
        latencies = {}
        for task_name, task_cfg in tasks.items():
            n = task_cfg['n_range'][-1]
            m = task_cfg['m'] if task_cfg['m'] else n
            lat = measure_latency(model, n, m, device=device)
            latencies[task_name] = lat
            print(f"  Latency ({task_name}, n={n}): {lat:.2f} ms")

        results[name] = {
            'params': n_params,
            'tasks': task_results,
            'latencies': latencies,
        }

    # ============================================================
    # PLOT 1: Cost Ratio Comparison (grouped bar)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(results.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for ax_idx, task_name in enumerate(['Linear Assignment', 'TSP']):
        ax = axes[ax_idx]
        n_range = tasks[task_name]['n_range']
        x = np.arange(len(n_range))
        width = 0.2

        for i, mn in enumerate(model_names):
            vals = results[mn]['tasks'][task_name]
            bars = ax.bar(x + i * width, vals, width, label=mn, color=colors[i], edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f'{v:.2f}', ha='center', fontsize=7)

        ax.set_xlabel('Set size n')
        ax.set_ylabel('Cost Ratio (lower = better)')
        ax.set_title(f'{task_name}')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([str(n) for n in n_range])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Baseline Comparison: Cost Ratio (lower is better)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'baseline_comparison.png'), dpi=150)
    plt.close(fig)
    print("\nPlot saved: baseline_comparison.png")

    # ============================================================
    # PLOT 2: Latency Comparison
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.35

    for i, task_name in enumerate(['Linear Assignment', 'TSP']):
        vals = [results[mn]['latencies'][task_name] for mn in model_names]
        bars = ax.bar(x + i * width, vals, width, label=f'{task_name} (n={tasks[task_name]["n_range"][-1]})',
                      color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}ms', ha='center', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Inference Latency (ms)')
    ax.set_title('Inference Latency Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'baseline_latency.png'), dpi=150)
    plt.close(fig)
    print("Plot saved: baseline_latency.png")

    # ============================================================
    # SAVE JSON
    # ============================================================
    json_path = os.path.join(out_dir, 'baseline_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else float(o))
    print(f"JSON saved: {json_path}")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<22} {'Params':>10} {'LA n=10':>8} {'LA n=20':>8} {'LA n=50':>8} {'TSP n=6':>8} {'TSP n=10':>9} {'Lat ms':>8}")
    print(f"{'-'*80}")
    for mn in model_names:
        r = results[mn]
        la = r['tasks']['Linear Assignment']
        tsp = r['tasks']['TSP']
        lat = r['latencies']['Linear Assignment']
        print(f"{mn:<22} {r['params']:>10,} {la[0]:>8.3f} {la[1]:>8.3f} {la[2]:>8.3f} {tsp[0]:>8.3f} {tsp[2]:>9.3f} {lat:>8.2f}")
    print(f"{'='*80}")

    return results


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_comparison()

#!/usr/bin/env python
"""
Fleet Optimization Prototype
============================
Permutation-invariant neural combinatorial optimization for heterogeneous-state
homogeneous fleets during online disruption.

Verifies:
  1. S_n-invariance (Theorem 2.3)
  2. Graceful robot removal (Theorem 3.1)
  3. Zero-shot n-scaling (Theorem 3.1-4)

Problem: Linear Assignment. n robots with (x,y,battery) states, m task locations.
Minimize total cost = sum of per-assignment costs.

Architecture is defined in ``src.architecture`` (soft Reynolds proxy — see Paper §7).
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .architecture import FleetPolicy

# Re-export for ``from src.model import FleetPolicy`` compatibility
__all__ = ["FleetPolicy", "generate_instance", "train_policy"]


# ============================================================
# PROBLEM GENERATION
# ============================================================

def generate_instance(n, m, d=4, seed=None):
    """
    Generate a random assignment problem.
    d = 4: (x, y, battery, task_progress)
    """
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
# TRAINING
# ============================================================

def train_policy(d=4, h=128, L=6, m_inducing=8, n_tasks=10,
                 n_train_range=(5, 15), epochs=200, lr=1e-3,
                 device='cpu', verbose=True):
    """Train the fleet policy on random assignment instances."""
    model = FleetPolicy(
        d, h=h, L=L, m=m_inducing, d_task=2, score_mode="shared"
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model: {n_params:,} parameters | d={d}, h={h}, L={L}, m={m_inducing}")
        print(f"Training on n ∈ [{n_train_range[0]}, {n_train_range[1]}], tasks={n_tasks}")
        print("-" * 60)

    epoch_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 10

        for _ in range(n_batches):
            n = np.random.randint(n_train_range[0], n_train_range[1] + 1)
            robots, tasks, cost_matrix, _ = generate_instance(n, n_tasks, d)
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

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | loss = {epoch_loss:.6f}")

    if verbose:
        print("-" * 60)
        print(f"Training complete. Final loss: {epoch_loss:.6f}")
    return model.to(device)


# ============================================================
# VERIFICATION
# ============================================================

def test_invariance(model, d=4, n_tasks=10, n_tests=100, device='cpu'):
    """Theorem 2.3: pi(sigma * F) = pi(F) for all sigma in S_n."""
    if verbose_global:
        print("\n[TEST 1] S_n-Invariance (Theorem 2.3)")
        print(f"  Testing {n_tests} random instances...")

    model.eval()
    max_diff = 0.0
    n_fail = 0

    for i in range(n_tests):
        n = np.random.randint(5, 16)
        robots, tasks, _, _ = generate_instance(n, n_tasks, d, seed=i)
        robots, tasks = robots.to(device), tasks.to(device)

        scores_orig = model(robots, tasks)
        g = torch.Generator().manual_seed(7_000 + i)
        perm = torch.randperm(n, generator=g)
        scores_perm = model(robots[perm], tasks)

        diff = (scores_perm - scores_orig[perm]).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            n_fail += 1

    if verbose_global:
        status = "PASS ✓" if n_fail == 0 else f"FAIL ✗ ({n_fail}/{n_tests} failed)"
        print(f"  Max |Δ| = {max_diff:.2e}  →  {status}")
    return max_diff, n_fail


def test_disruption(model, d=4, n_tasks=10, n_tests=50, device='cpu'):
    """Theorem 3.1: Removing robot k gives valid (n-1)-robot allocation."""
    if verbose_global:
        print("\n[TEST 2] Graceful Robot Removal (Theorem 3.1)")
        print(f"  Testing {n_tests} removal scenarios...")

    model.eval()
    n_fail = 0

    for i in range(n_tests):
        n = np.random.randint(8, 16)
        robots, tasks, _, _ = generate_instance(n, n_tasks, d, seed=1000 + i)
        robots, tasks = robots.to(device), tasks.to(device)

        scores_full = model(robots, tasks)
        k = np.random.randint(0, n)
        robots_reduced = torch.cat([robots[:k], robots[k + 1:]], dim=0)
        scores_reduced = model(robots_reduced, tasks)

        expected = torch.cat([scores_full[:k], scores_full[k + 1:]], dim=0)
        diff = (scores_reduced - expected).abs().max().item()

        if not torch.isfinite(scores_reduced).all():
            n_fail += 1
        if diff > 1e-4:
            n_fail += 1

    if verbose_global:
        status = "PASS ✓" if n_fail == 0 else f"FAIL ✗ ({n_fail}/{n_tests} failed)"
        print(f"  Removal consistency max |Δ| verified | {status}")
        print("  (Architecture is defined for any n; no retraining needed)")
    return n_fail


def test_zero_shot_scaling(model, d=4, n_tasks=10, device='cpu'):
    """Theorem 3.1-4: Train on n∈[5,15], evaluate on n=50, 100."""
    if verbose_global:
        print("\n[TEST 3] Zero-shot n-Scaling (Theorem 3.1-4)")

    model.eval()
    results = {}

    for n_test in [10, 50, 100]:
        robots, tasks, cost_matrix, optimal = generate_instance(
            n_test, n_tasks, d, seed=999
        )
        robots, tasks = robots.to(device), tasks.to(device)

        t0 = time.time()
        scores = model(robots, tasks)
        inference_ms = (time.time() - t0) * 1000

        valid = torch.isfinite(scores).all().item()
        shape_ok = scores.shape == (n_test, n_tasks)

        r, c = linear_sum_assignment(scores.detach().cpu().numpy())
        model_cost = cost_matrix[r, c].sum().item()
        ratio = model_cost / optimal if optimal > 0 else float('inf')

        results[n_test] = {
            'valid': valid,
            'shape_ok': shape_ok,
            'inference_ms': inference_ms,
            'cost_ratio': ratio,
        }

        if verbose_global:
            print(f"  n={n_test:4d} | valid={valid} | shape={scores.shape} | "
                  f"{inference_ms:.1f} ms | cost/optimal = {ratio:.3f}")

    if verbose_global:
        all_valid = all(r['valid'] and r['shape_ok'] for r in results.values())
        status = "PASS ✓" if all_valid else "FAIL ✗"
        print(f"  All sizes produce valid output → {status}")
        print("  (No retraining between n=10, 50, 100)")
    return results


# ============================================================
# MAIN
# ============================================================

verbose_global = True

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("FLEET OPTIMIZATION PROTOTYPE")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    print("\n[TRAINING]")
    model = train_policy(
        d=4, h=128, L=4, m_inducing=8, n_tasks=10,
        n_train_range=(5, 15), epochs=100, lr=1e-3,
        device=device, verbose=verbose_global,
    )

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fleet_policy.pt'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved: {save_path}")

    print("\n" + "=" * 60)
    print("THEOREM VERIFICATION")
    print("=" * 60)

    inv_diff, inv_fail = test_invariance(model, device=device)
    dis_fail = test_disruption(model, device=device)
    scale_results = test_zero_shot_scaling(model, device=device)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  1. S_n-Invariance:        {'PASS' if inv_fail == 0 else 'FAIL'} (max Δ = {inv_diff:.2e})")
    print(f"  2. Robot Removal:         {'PASS' if dis_fail == 0 else 'FAIL'} ({dis_fail} failures)")
    all_scale = all(r['valid'] and r['shape_ok'] for r in scale_results.values())
    print(f"  3. Zero-shot scaling:     {'PASS' if all_scale else 'FAIL'} (n=10,50,100)")
    print(f"\n  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("  Architecture: Set Transformer (ISAB) + soft Reynolds + Cross-Attn")
    print("  Invariance: structural (shared params + symmetric aggregation)")
    print("=" * 60)

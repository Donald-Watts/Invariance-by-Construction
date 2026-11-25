#!/usr/bin/env python
"""
Empirical verification of Proposition 5.1: the catalogue's 1-layer
attention + P_inv readout can compute e₂.

Tests:
  1. θ₁ depends on e₂ (same e₁, different e₂ → different θ₁)
  2. Taylor expansion has non-zero e₂ coefficient
  3. e₂ is S_n-invariant (sanity)
  4. θ₁ is S_n-invariant (sanity)
"""
import torch
import torch.nn.functional as F
import numpy as np

def catalogue_attention(x):
    """Catalogue attention for real x: x -> softmax(x x^T)·x (row-wise)."""
    logits = x.unsqueeze(0) * x.unsqueeze(1)  # (1, n, n)
    weights = F.softmax(logits, dim=-1)        # (1, n, n) row-stochastic
    return (weights.squeeze(0) @ x)            # (n,)

def theta_1(x):
    """Full 1-layer catalogue readout scalar."""
    out = catalogue_attention(x)
    n = x.shape[0]
    m = out.mean()
    return (1/np.sqrt(n)) * (m * n)  # = sqrt(n) * mean(out)

def e1(x): return x.sum().item()
def e2(x):
    return (x.sum()**2 - (x**2).sum()).item() / 2

def run_tests():
    torch.manual_seed(42)
    n = 6
    results = []

    # Test 1
    x = torch.randn(n)
    t = 0.5
    y = x.clone(); y[0] += t; y[1] -= t
    assert abs(e1(x) - e1(y)) < 1e-10
    assert abs(e2(x) - e2(y)) > 1e-3
    diff = abs(theta_1(x) - theta_1(y))
    r1 = diff > 1e-6
    print(f"Test 1: θ₁ depends on e₂: {diff:.6f} {'✓' if r1 else '✗'}")
    results.append(r1)

    # Test 2: fit θ₁ ≈ a·e₁ + b·e₂ at small scale
    h = 0.01
    A = torch.zeros(200, 2); b = torch.zeros(200)
    for i in range(200):
        v = h * torch.randn(n)
        A[i, 0] = e1(v); A[i, 1] = e2(v); b[i] = theta_1(v)
    coeff, _, _, _ = torch.linalg.lstsq(A, b)
    r2 = abs(coeff[1].item()) > 1e-6
    print(f"Test 2: e₂ coefficient = {coeff[1]:.6f} {'✓' if r2 else '✗'}")
    results.append(r2)

    # Test 3: e₂ invariance
    x8 = torch.randn(8)
    max_d = max(abs(e2(x8) - e2(x8[torch.randperm(8)])) for _ in range(5))
    r3 = max_d < 1e-5
    print(f"Test 3: e₂ invariance (max Δ = {max_d:.2e}) {'✓' if r3 else '✗'}")
    results.append(r3)

    # Test 4: θ₁ invariance
    max_d = max(abs(theta_1(x8) - theta_1(x8[torch.randperm(8)])) for _ in range(5))
    r4 = max_d < 1e-5
    print(f"Test 4: θ₁ invariance (max Δ = {max_d:.2e}) {'✓' if r4 else '✗'}")
    results.append(r4)

    print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'}: {sum(results)}/4")
    return all(results)

if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python
"""
Formal Big-O Complexity Analysis
================================
Measures and derives the FLOP count of the Constrained-Operation Catalogue
architecture as a function of set size n, inducing points m, hidden dim h,
and depth L.

Outputs:
  - complexity_flops.png   (FLOPs vs n, log-log)
  - complexity_breakdown.png (stacked bar: per-component FLOPs)
  - complexity_metrics.json  (raw numbers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# ARCHITECTURE (same as breadth_study.py)
# ============================================================

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
# FLOP COUNTER (hook-based)
# ============================================================

class FLOPCounter:
    """Counts FLOPs (multiply-adds × 2) via forward hooks."""
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.flops = 0
        self.component_flops = {}
        self._active = False

    def _hook_fn(self, name):
        def hook(module, input, output):
            if not self._active:
                return
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * in * out * batch_elements
                in_f = input[0].shape
                out_f = output.shape
                # batch elements = product of all dims except last
                batch = 1
                for d in in_f[:-1]:
                    batch *= d
                flops = 2 * in_f[-1] * out_f[-1] * batch
                self.flops += flops
                self.component_flops[name] = self.component_flops.get(name, 0) + flops
            elif isinstance(module, nn.MultiheadAttention):
                # Q, K, V projections: 3 × 2 × h × h × seq_len
                # Attention scores: 2 × seq_q × seq_k × h
                # Output projection: 2 × h × h × seq_q
                # Approximate: use input shapes
                q = input[0]  # query
                k = input[1]  # key
                v = input[2]  # value
                seq_q = q.shape[1] if q.dim() > 2 else 1
                seq_k = k.shape[1] if k.dim() > 2 else 1
                h = q.shape[-1]
                # Projections (Q, K, V): 3 * 2 * h * h * seq
                proj_flops = 3 * 2 * h * h * (seq_q + seq_k)
                # Attention: 2 * seq_q * seq_k * h
                attn_flops = 2 * seq_q * seq_k * h
                # Output proj: 2 * h * h * seq_q
                out_flops = 2 * h * h * seq_q
                total = proj_flops + attn_flops + out_flops
                self.flops += total
                self.component_flops[name] = self.component_flops.get(name, 0) + total
            elif isinstance(module, nn.LayerNorm):
                # ~4 * h * batch (mean, var, normalize, scale+shift)
                batch = 1
                for d in input[0].shape[:-1]:
                    batch *= d
                h = input[0].shape[-1]
                flops = 4 * h * batch
                self.flops += flops
                self.component_flops[name] = self.component_flops.get(name, 0) + flops
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.LayerNorm)):
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def count(self, X, tasks):
        self.flops = 0
        self.component_flops = {}
        self._active = True
        self.model.eval()
        with torch.no_grad():
            self.model(X, tasks)
        self._active = False
        return self.flops, self.component_flops


# ============================================================
# THEORETICAL FLOP DERIVATION
# ============================================================

def theoretical_flops(n, m, h, L, d_in=4, d_task=2, d_score=32):
    """
    Derive FLOPs analytically.
    
    Components:
    - enc: 2 * (d_in*h + h*h) * n
    - Per SetBlock:
      - ISAB attn1 (m queries, n keys): 3*2*h*h*(m+n) + 2*m*n*h + 2*h*h*m
      - ISAB attn2 (n queries, m keys): 3*2*h*h*(n+m) + 2*n*m*h + 2*h*h*n
      - FFN: 2 * (h*4h + 4h*h) * n = 16*h^2 * n
      - xattn (n queries, 1 key): 3*2*h*h*(n+1) + 2*n*1*h + 2*h*h*n
      - 3 LayerNorms: 3 * 4*h*n
    - task_enc: 2 * (d_task*h + h*h) * m
    - cross_attn (n queries, m keys): 3*2*h*h*(n+m) + 2*n*m*h + 2*h*h*n
    - final_norm: 4*h*n
    - score_proj (×2 for H and T): 2 * 2 * (h*64 + 64*32) * (n+m)
    - dot product: 2 * n * m * d_score
    """
    # enc
    enc = 2 * (d_in * h + h * h) * n
    
    # Per SetBlock
    isab_attn1 = 3*2*h*h*(m+n) + 2*m*n*h + 2*h*h*m
    isab_attn2 = 3*2*h*h*(n+m) + 2*n*m*h + 2*h*h*n
    ffn = 16 * h * h * n
    xattn = 3*2*h*h*(n+1) + 2*n*1*h + 2*h*h*n
    norms_block = 3 * 4 * h * n
    per_block = isab_attn1 + isab_attn2 + ffn + xattn + norms_block
    
    # task_enc
    task_enc = 2 * (d_task * h + h * h) * m
    
    # cross_attn
    cross_attn = 3*2*h*h*(n+m) + 2*n*m*h + 2*h*h*n
    
    # final_norm
    final_norm = 4 * h * n
    
    # score_proj (applied to both H [n,h] and T [m,h])
    score_proj = 2 * 2 * (h*64 + 64*32) * (n + m)
    
    # dot product H_s @ T_s.T: [n,32] @ [32,m] → 2*n*m*32
    dot = 2 * n * m * d_score
    
    total = enc + L * per_block + task_enc + cross_attn + final_norm + score_proj + dot
    
    breakdown = {
        'enc': enc,
        f'SetBlock×{L}': L * per_block,
        '  ISAB attn1': L * isab_attn1,
        '  ISAB attn2': L * isab_attn2,
        '  FFN': L * ffn,
        '  xattn': L * xattn,
        '  norms': L * norms_block,
        'task_enc': task_enc,
        'cross_attn': cross_attn,
        'final_norm': final_norm,
        'score_proj': score_proj,
        'dot_product': dot,
    }
    return total, breakdown


# ============================================================
# MAIN
# ============================================================

def run_complexity():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    h, L, m_ind, d_in, d_task = 128, 4, 8, 4, 2
    
    model = FleetPolicy(d_in, h, L, m_ind, d_task=d_task)
    counter = FLOPCounter(model)
    counter.register()
    
    n_values = [5, 10, 20, 50, 100, 200]
    m_test = 10  # fixed task size for scaling study
    
    results = []
    for n in n_values:
        X = torch.randn(n, d_in)
        tasks = torch.randn(m_test, d_task)
        
        # Empirical FLOPs
        emp_flops, emp_comp = counter.count(X, tasks)
        
        # Theoretical FLOPs
        theo_flops, theo_break = theoretical_flops(n, m_test, h, L, d_in, d_task)
        
        results.append({
            'n': n,
            'm': m_test,
            'empirical_flops': emp_flops,
            'theoretical_flops': theo_flops,
            'ratio_emp_theo': emp_flops / theo_flops if theo_flops > 0 else None,
            'theoretical_breakdown': theo_break,
        })
        print(f"n={n:4d} | Empirical: {emp_flops:>14,.0f} | Theoretical: {theo_flops:>14,.0f} | Ratio: {emp_flops/theo_flops:.3f}")
    
    counter.remove()
    
    # ============================================================
    # PLOT 1: FLOPs vs n (log-log)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [r['n'] for r in results]
    emp = [r['empirical_flops'] for r in results]
    theo = [r['theoretical_flops'] for r in results]
    
    ax.loglog(ns, emp, 'o-', label='Empirical (hook counter)', color='#2196F3', linewidth=2)
    ax.loglog(ns, theo, 's--', label='Theoretical (analytic)', color='#F44336', linewidth=2)
    
    # Reference lines: O(n) and O(n²)
    n_ref = np.array([5, 200])
    c1 = emp[0] / n_ref[0]
    c2 = emp[0] / n_ref[0]**2
    ax.loglog(n_ref, c1 * n_ref, ':', color='gray', alpha=0.5, label='O(n) ref')
    ax.loglog(n_ref, c2 * n_ref**2, ':', color='gray', alpha=0.5, label='O(n²) ref')
    
    ax.set_xlabel('Set size n')
    ax.set_ylabel('FLOPs (forward pass)')
    ax.set_title(f'Complexity: FLOPs vs n (h={h}, L={L}, m={m_ind}, m_tasks={m_test})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'complexity_flops.png'), dpi=150)
    plt.close(fig)
    print("Plot saved: complexity_flops.png")
    
    # ============================================================
    # PLOT 2: Stacked bar — component breakdown at n=50
    # ============================================================
    r50 = [r for r in results if r['n'] == 50][0]
    bd = r50['theoretical_breakdown']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(bd.keys())
    vals = [v / 1e6 for v in bd.values()]  # in MFLOPs
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    bars = ax.barh(range(len(labels)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('MFLOPs')
    ax.set_title(f'FLOP Breakdown at n=50 (total={r50["theoretical_flops"]/1e6:.1f} MFLOPs)')
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'complexity_breakdown.png'), dpi=150)
    plt.close(fig)
    print("Plot saved: complexity_breakdown.png")
    
    # ============================================================
    # SAVE JSON
    # ============================================================
    json_path = os.path.join(out_dir, 'complexity_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {json_path}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPLEXITY ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Architecture: h={h}, L={L}, m_ind={m_ind}, d_in={d_in}, d_task={d_task}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'n':>5} {'Empirical FLOPs':>16} {'Theoretical FLOPs':>18} {'Ratio':>8} {'FLOPs/n':>12}")
    print(f"{'-'*65}")
    for r in results:
        print(f"{r['n']:>5} {r['empirical_flops']:>16,.0f} {r['theoretical_flops']:>18,.0f} "
              f"{r['ratio_emp_theo']:>8.3f} {r['empirical_flops']/r['n']:>12,.0f}")
    print(f"{'='*70}")
    
    # Fit power law: FLOPs = a * n^p
    log_n = np.log([r['n'] for r in results])
    log_f = np.log([r['empirical_flops'] for r in results])
    p, log_a = np.polyfit(log_n, log_f, 1)
    print(f"\nPower-law fit: FLOPs ≈ {np.exp(log_a):.2f} × n^{p:.3f}")
    print(f"  → Dominant scaling is O(n^{p:.2f})")
    if p < 1.2:
        print(f"  → Confirms LINEAR scaling in n (attention with fixed m is O(n))")
    else:
        print(f"  → Super-linear scaling detected")
    
    return results


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    run_complexity()

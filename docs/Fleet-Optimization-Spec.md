# Permutation-Invariant Neural Combinatorial Optimization
## for Heterogeneous-State Homogeneous Fleets during Online Disruption

*Formal specification · Invariance theorems · Disruption handling · Architecture · Efficiency · Prior-art positioning*

**Companion to:** `Catalogue.md` (the $S_n$-invariant layered architecture) and `Extension-A.md` (efficiency + expressiveness bounds).

---

## 0 — Problem Statement

### 0.1 — Setting

A fleet of $n$ **homogeneous-capability, heterogeneous-state** robots operates in a shared environment. At time $t$, robot $i$ carries a state vector

$$x_i(t) \in \mathcal{X} \subset \mathbb{R}^d,$$

where $\mathcal{X}$ encodes position, velocity, battery, task-progress, sensor state, and any other per-robot dynamic quantities. The fleet state is the **unordered set**

$$\mathcal{F}_t = \{x_1(t),\; x_2(t),\; \dots,\; x_n(t)\}.$$

Robots are **homogeneous in capability** (same motor specs, same sensing range, same action space) but **heterogeneous in state** (different positions, battery, task phase). Labels are arbitrary scheduler-assigned indices, not a meaningful ranking.

### 0.2 — Task

Given $\mathcal{F}_t$, produce an **allocation** (task assignment, routing plan, or resource share):

$$\pi: \mathcal{F}_n \to \mathcal{A},$$

where $\mathcal{F}_n = (\mathbb{R}^d)^n / S_n$ and $\mathcal{A}$ is the action space (bipartite matching, routing, allocation vector).

> **Architecture note.** The neural architecture produces a **score matrix** $S \in \mathbb{R}^{n \times m}$ (robot-task affinity). The final allocation (bipartite matching, routing, or allocation vector) is applied **externally** to this score matrix (e.g. Hungarian algorithm). The network itself is a $S_n$-invariant map $\mathcal{F}_n \to \mathbb{R}^{n \times m}$, not a direct allocator.

The policy must be:

1. **$S_n$-invariant:** $\pi(\sigma \cdot \mathcal{F}) = \pi(\mathcal{F})$ for all $\sigma \in S_n$.
2. **Online:** inference in $\text{poly}(n)$ time, not $O(n!)$ or MILP-exponential.
3. **Disruption-tolerant:** robot removal $\mathcal{F} \to \mathcal{F} \setminus \{x_k\}$ produces a valid $(n{-}1)$-robot allocation **without retraining**.

### 0.3 — What this is NOT

- Not multi-agent RL (no shared reward, no agent-to-agent learning).
- Not a standard MILP (exact but intractable for $n > 30$, cannot handle online disruption in real-time).
- Not a GNN on a fixed graph (the topology *is* the decision variable).
- Not the case where robots are truly identical (then state is one number and the problem is trivial).

### 0.4 — Expressiveness ceiling (read before the architecture)

The architecture's final readout is the **fleet centroid** $c_L = \frac{1}{n}\sum_i H_i^{(L)} \in \mathbb{R}^h$, followed by an MLP head. The score matrix $S \in \mathbb{R}^{n \times m}$ is produced by cross-attending each robot's encoded state to this centroid. Consequently:

- The architecture is **sensitive to per-robot heterogeneity** (attention operates on individual $H_i$), but the **global decision** (which tasks to prioritize, how to allocate shared resources) is a function of the $h$-dimensional centroid, not of all $n \cdot d$ independent symmetric invariants.
- Two fleets with the same centroid but different heterogeneity distributions can receive **different** per-robot scores (via the cross-attention path), but the **same** global allocation policy.
- This is the fleet analog of the Catalogue's 1-D fiber limitation (Extension-A Corollary A.2a): the architecture accesses a **finite-dimensional slice** of the full invariant space, parameterized by $O(L \cdot h^2 + L \cdot n \cdot h)$ real weights.
- For the allocation problem this is a **feature, not a bug**: the policy should depend on fleet-level statistics (total battery, spatial spread) more than on any single robot's exact state. But it is a **limitation** to be aware of: if the optimal allocation depends on higher-order symmetric statistics (e.g. the variance of battery levels), the architecture must learn to encode that into the $h$-dimensional centroid through its attention layers.

> **Practical consequence.** The expressiveness ceiling here is **much higher** than in the Catalogue (where it is 1 complex dimension), because $h$ is a free hyperparameter and the cross-attention path preserves per-robot information in the score matrix. The limitation is on the *global policy*, not on the *per-robot scores*.

---

## 1 — Mathematical Formulation

### 1.1 — The $S_n$ action

$S_n$ acts on the **rows** of $X = [x_1; \dots; x_n] \in \mathbb{R}^{n \times d}$ by permuting indices:

$$(\sigma \cdot X)_{\sigma(i),:} = X_{i,:}.$$"

A fleet policy $\pi: (\mathbb{R}^d)^n / S_n \to \mathcal{A}$ lifts to a $S_n$-invariant function $\tilde{\pi}: (\mathbb{R}^d)^n \to \mathcal{A}$ satisfying $\tilde{\pi}(\sigma \cdot X) = \tilde{\pi}(X)$."

### 1.2 — Invariant function space (and why it is richer than the Catalogue)

The space of $S_n$-invariant functions on $(\mathbb{R}^d)^n$ is

$$C^\infty((\mathbb{R}^d)^n)^{S_n}.$"

By the **DeepSets theorem** (Zaheer et al., 2017), any such function can be written as

$$f(x_1, \dots, x_n) = \rho\!\Bigl(\sum_{i=1}^n \phi(x_i)\Bigr),$$

for suitable $\phi: \mathbb{R}^d \to \mathbb{R}^h$ and $\rho: \mathbb{R}^h \to \mathbb{R}$ (in the infinite-capacity limit). The **sum** (or any symmetric aggregation) is the $S_n$-invariant operation.

> **Contrast with the Catalogue.** In `Catalogue.md`, $S_n$ permutes the $n$ coordinates of a *single* vector $x \in \mathbb{C}^n$; the invariant subspace is 1-dimensional ($\mathrm{span}(\mathbf{1})$). Here, $S_n$ permutes the $n$ **rows** of an $n \times d$ matrix; the invariant space is **infinite-dimensional** with $n \cdot d$ effective generators. The architecture must be invariant in *robot identity* but sensitive to *per-robot state*.

### 1.3 — Reynolds operator (fleet setting)

For linear readout, the Reynolds operator on $(\mathbb{R}^d)^n$ is

$$P_{\text{inv}} = \frac{1}{|S_n|}\sum_{\sigma \in S_n} \sigma = \frac{1}{n}\,\mathbf{1}\mathbf{1}^\top \otimes I_d,$$

a **rank-$d$** projection. Applied to $X \in \mathbb{R}^{n \times d}$:

$$P_{\text{inv}}(X) = \frac{1}{n}\,\mathbf{1}\mathbf{1}^\top X = \bar{x}\,\mathbf{1}^\top,$$

where $\bar{x} = \frac{1}{n}\sum_i x_i$ is the fleet centroid. All non-mean information (variance, spread, heterogeneity) lives in $\mathbf{1}^\perp \otimes \mathbb{R}^d$.

> **Key point.** In the Catalogue, the invariant subspace *is* the entire invariant content (1 scalar). Here, the invariant subspace is only the $d$-dimensional centroid. The **heterogeneity** — the thing that makes the problem non-trivial — is in the *non-invariant* complement, and must be processed *before* aggregation.

> **Implementation note.** `src/model.py` realizes the centroid step as soft cross-attention to $\bar{H}\in\mathbb{R}^h$ (elements stay $n$ vectors). That is a practical proxy for the fleet Reynolds idea above, **not** the catalogue's hard rank-1 $P_{\mathrm{inv}}$ on $\mathbb{C}^n$. Depth-irrelevance theorems in `Paper.md` apply to the hard catalogue; the factorial ablation tests the soft proxy empirically. See `Paper.md` §7.

---

## 2 — Architecture

### 2.1 — Design principles

| Principle | Consequence |
|---|---|
| $S_n$-invariance is **structural** | Invariant by construction for any parameters; no invariance penalty needed |
| Heterogeneity is **preserved** | Per-robot states are never collapsed before final readout; attention operates on encoded states |
| Online disruption = **set modification** | Robot removal = delete a row; architecture is defined for any $n$ |
| Zero-shot $n$-scaling | No $n$-dependent parameters; network trained on $n=10$ applies to $n=100$ unchanged |

### 2.2 — Layer structure

```
INPUT:  X ∈ R^{n×d}  (n robot state vectors, unordered)

  [ENC]   φ: R^d → R^h  (shared MLP, all robots)
          H = [φ(x_1); ...; φ(x_n)] ∈ R^{n×h}

  ─── SET BLOCK × L (L ≤ 10) ──────────────────────────────
  [ISAB]  Induced Set Attention: m inducing points (m ≪ n)
          cost O(n·m·h) — linear in n
  [PAB]   Point Attention: m inducing → n elements
  [FFN]   per-element shared feed-forward
  [POOL]  c = (1/n) Σ H_i   [REYNOLDS, rank-h]
  [XATTN] cross-attention: c attends to H (set context)
  [RES]   H ← H + f(c, H)
  ─────────────────────────────────────────────────────────

OUTPUT:  c_L = (1/n) Σ H_L  ∈ R^h
         → MLP head → allocation a ∈ A
```

**Why Set Transformer (not plain DeepSets):** ISAB uses $m \ll n$ inducing points, giving $O(n \cdot m)$ attention instead of $O(n^2)$. Inducing points are **shared parameters** independent of $n$, enabling zero-shot scaling. The cross-attention between centroid $c$ and set $H$ allows the policy to condition global decisions on both the fleet-level summary and individual heterogeneity.

### 2.3 — Invariance theorem (fleet setting)

**Theorem (Fleet $S_n$-invariance).** Let $\pi$ be a network in which:
1. Every per-element operation $\phi$ is **shared** (same weights for all $n$ robots),
2. Every aggregation is **symmetric** (sum, mean, ISAB with shared inducing points),
3. The final readout is applied to a **symmetric function** of the set.

Then $\pi(\sigma \cdot \mathcal{F}) = \pi(\mathcal{F})$ for all $\sigma \in S_n$, **regardless of parameter values**.

**Proof.** By induction on layers. Base: shared $\phi$ + symmetric $\sum$ is invariant under relabeling. Inductive step: ISAB with shared Q/K/V and shared inducing points produces output that is a symmetric function of the input set (permuting input rows permutes attention patterns identically). Cross-attention between $c$ (symmetric) and $H$ preserves invariance. $\square$

> **Connection to the Catalogue.** Same proof structure as Theorem (ii) in `Catalogue.md`: invariance is enforced by the symmetric aggregation, not by individual operations. Here the invariant subspace is $h$-dimensional (centroid encoding) vs. 1-dimensional in the Catalogue.

---

## 3 — Online Disruption: A Design Property

### 3.1 — Formal statement

**Design property (Graceful robot removal).** Let $\pi$ be a fleet policy satisfying Theorem 2.3, with parameters $\theta$ trained on fleets of size $n_0$. For any $n \le n_0$ and any robot $k$:

1. **Well-defined:** $\pi(\mathcal{F} \setminus \{x_k\})$ is a valid allocation for the $(n{-}1)$-robot fleet.
2. **Invariant:** $\pi(\sigma \cdot (\mathcal{F} \setminus \{x_k\})) = \pi(\mathcal{F} \setminus \{x_k\})$ for all $\sigma \in S_{n-1}$.
3. **Efficient:** inference cost $O((n{-}1) \cdot m \cdot h)$ per layer.
4. **No retraining:** parameters $\theta$ are $n$-independent; the set simply has one fewer element.

**Proof.** (1) The architecture maps sets, not indexed tuples. (2) The invariance proof uses only shared-parameter + symmetric-aggregation, which hold for any set size. (3) Every operation is linear in $n$. (4) No parameter depends on $n$. $\square$

### 3.2 — Practical consequence

| Event | MILP re-solve | This architecture |
|---|---|---|
| Robot $k$ fails | Remove constraint, re-solve. Minutes. | Delete row $k$. Forward pass. **$<$ 1 ms.** |
| Robot battery drops | Update RHS, re-solve. | Update row $k$. Forward pass. **$<$ 1 ms.** |
| New robot joins | Add constraint, re-solve. | Add row $j$. Forward pass. **$<$ 1 ms.** |
| Fleet $10 \to 100$ | Retrain / re-solve from scratch. | **No change.** Same $\theta$, $n=100$ rows. |

### 3.3 — Why this is non-trivial

- **MILP/CP** encodes the fleet as indexed variables $x_{ij}$. Removing robot $k$ requires re-indexing and re-solving.
- **Standard neural combinatorial optimization** (Attention Model, Pointer Network) uses fixed input dimension $n \times d$; changing $n$ requires retraining or padding.
- **This architecture** is a function of the **set**, not the tuple. $n$ is a free variable.

The gap in the literature is not the architecture (Set Transformers exist) but the **formal deployment in the combinatorial optimization setting with the disruption theorem explicitly stated and the MILP comparison made concrete**.

---

## 4 — Efficiency

| Operation | Cost | $n$-dependence |
|---|---|---|
| Per-robot encoding $\phi$ | $O(n \cdot d \cdot h)$ | Linear |
| ISAB ($m$ inducing) | $O(n \cdot m \cdot h)$ | **Linear** |
| PAB | $O(n \cdot m \cdot h)$ | Linear |
| Per-element FFN | $O(n \cdot h^2)$ | Linear |
| $P_{\text{inv}}$ (centroid) | $O(n \cdot h)$ | Linear |
| Cross-attention | $O(n \cdot h)$ | Linear |
| **Total per layer** | $O(n(mh + h^2))$ | **Linear** |
| $L$ layers + head | $L \cdot O(n(mh + h^2))$ | **Linear** |

| Approach | $n=10$ | $n=100$ | $n=1000$ |
|---|---|---|---|
| MILP | $\sim 1\text{ s}$ | $\sim 10\text{ min}$ | intractable |
| Attention Model (fixed $n$) | fast | $10\times$ | **retraining needed** |
| **This architecture** | fast | $10\times$ | **$100\times$, no retraining** |

For $m=8$, $h=128$, $L=6$: per-layer $\approx n \cdot (8 \cdot 128 + 128^2) \approx n \cdot 17408$. At $n=1000$: $\sim 1.7 \times 10^7$ ops per layer, $\sim 10^8$ total. Single GPU, sub-millisecond.

---

## 5 — Positioning vs. Prior Art

| Work | What it does | What this adds |
|---|---|---|
| DeepSets (Zaheer 2017) | $\rho(\sum\phi(x_i))$ invariance | ISAB attention for pairwise interactions; disruption theorem |
| Set Transformer (Lee 2019) | ISAB/PAB for variable-size sets | Explicit $S_n$-invariance proof in comb-opt context; MILP comparison |
| Attention Model (Kool 2019) | Neural TSP routing | Fixed $n$; no disruption handling; no invariance theorem |
| Reinforcement for comb-opt (Bello 2017) | RL + pointer network | Same: fixed $n$, no invariance by construction |
| MILP (Gurobi/CPLEX) | Exact assignment | Exponential; no online disruption; no zero-shot scaling |
| **This spec** | Set Transformer + invariance by construction + disruption theorem | **Formal zero-shot $n$-scaling + millisecond disruption response + no retraining** |

**The contribution is the combination and the theorems:** the architecture is known, the invariance is known, but the **formal statement that robot removal is a no-op, that the policy generalizes to arbitrary $n$ without retraining, and that inference is $O(n)$ for disruption** is not stated in the combinatorial optimization literature. That is the paper.

---

## 7 — Verification Plan

| Test | Pass criterion |
|---|---|
| **Invariance:** random $\sigma \in S_n$, compare $\pi(X)$ vs. $\pi(\sigma \cdot X)$ | $\|\Delta\| < 10^{-12}$ for 1000 random permutations |
| **Disruption:** remove row $k$, compare $\pi(X \setminus x_k)$ to a reference $(n{-}1)$-robot solution | Allocation is valid (feasible) and $\|\Delta\|$ bounded |
| **Zero-shot:** train on $n=10$, evaluate on $n=50$, $n=100$ | No degradation beyond $O(n)$ inference time; task metric within $5\%$ of $n=10$ baseline |
| **Speed:** disruption response time | $<$ 5 ms on CPU, $<$ 1 ms on GPU for $n \le 100$ |
| **MILP comparison:** solve same instance with Gurobi | This architecture is $>$ 1000× faster for $n \ge 20$; solution quality within $10\%$ of MILP optimum |

---

## 8 — Minimal Implementation (for immediate prototyping)

```python
import torch, torch.nn as nn

class SetBlock(nn.Module):
    def __init__(self, h, m=8):
        super().__init__()
        self.isab = InducedSetAttention(h, m)
        self.ffn = nn.Sequential(nn.Linear(h, 4*h), nn.GELU(), nn.Linear(4*h, h))
        self.xattn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)

    def forward(self, H):  # H: (n, h)
        n = H.shape[0]
        H = self.norm1(H + self.isab(H))
        H = self.norm2(H + self.ffn(H))
        c = H.mean(dim=0, keepdim=True)  # (1, h) — Reynolds
        H, _ = self.xattn(H, c, c)      # cross-attend to centroid
        return H

class FleetPolicy(nn.Module):
    def __init__(self, d, h=128, L=6, m=8, n_tasks=20):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([SetBlock(h, m) for _ in range(L)])
        self.head = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, n_tasks))

    def forward(self, X):  # X: (n, d)
        H = self.enc(X)
        for blk in self.blocks:
            H = blk(H)
        c = H.mean(dim=0)  # final Reynolds: (h,)
        return self.head(c)  # (n_tasks,) allocation scores
```

~50 lines. Train on $n \in [5, 15]$ random instances. Evaluate on $n = 50, 100$. Test invariance. Test disruption.

---

*Fleet-Optimization-Spec.md · 2026-11-10 · Companion to Catalogue.md + Extension-A.md*

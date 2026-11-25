# Extension A — Efficiency Theorem and Sharp Expressiveness Bound

*Supplement to Catalogue.md (v1 → final). Adds two theorems and a one-line pseudocode correction.*

---

## A.1 — Efficiency: the Reynolds operator is rank-1 for the permutation representation

### Theorem A.1 (O(n) Reynolds)

Let $\rho: S_n \to \mathrm{GL}(n,\mathbb{C})$ be the **permutation representation**, i.e. $\rho(\sigma)_{ij} = \delta_{i,\,\sigma(j)}$ (coordinate permutation). Then the Reynolds operator

$$P_{\mathrm{inv}} \;=\; \frac{1}{|S_n|}\sum_{\sigma \in S_n}\rho(\sigma)$$

is the **rank-1 orthogonal projection**

$$\boxed{\;P_{\mathrm{inv}} \;=\; \frac{1}{n}\;\mathbf{1}\,\mathbf{1}^\*,\;}$$

where $\mathbf{1} = (1,1,\dots,1)^\top \in \mathbb{C}^n$.

**Proof.** For any $x \in \mathbb{C}^n$, the $i$-th coordinate of $P_{\mathrm{inv}}\,x$ is

$$(P_{\mathrm{inv}}\,x)_i \;=\; \frac{1}{|S_n|}\sum_{\sigma \in S_n} (\rho(\sigma)\,x)_i \;=\; \frac{1}{|S_n|}\sum_{\sigma \in S_n} x_{\sigma^{-1}(i)}.$$'

Since $S_n$ acts **transitively** on $\{1,\dots,n\}$, the map $\sigma \mapsto \sigma^{-1}(i)$ is a bijection $S_n \to \{1,\dots,n\}$ with each $j \in \{1,\dots,n\}$ hit exactly $(n-1)!$ times. Hence

$$(P_{\mathrm{inv}}\,x)_i \;=\; \frac{(n-1)!}{n!}\sum_{j=1}^{n} x_j \;=\; \frac{1}{n}\sum_{j=1}^{n}x_j \;=\; \frac{1}{n}\,\mathbf{1}^\* x,$$'

independent of $i$. Therefore $P_{\mathrm{inv}}\,x = \frac{1}{n}(\mathbf{1}^\* x)\,\mathbf{1} = \frac{1}{n}\,\mathbf{1}\,\mathbf{1}^\*\, x$ for all $x$, i.e.

$$P_{\mathrm{inv}} = \frac{1}{n}\,\mathbf{1}\,\mathbf{1}^\*.$$'

$\blacksquare$

### Corollary A.1a (Computational cost)

| Version | Cost | $n=8$ | $n=10$ | $n=50$ |
|---|---|---|---|---|
| Full sum over $S_n$ (Catalogue §5.1, line 598) | $O(n!\cdot n)$ | 322 560 | 36 288 000 | $\approx 3.6\times 10^{67}$ |
| **Rank-1 projection (Thm A.1)** | $O(n)$ | **8** | **10** | **50** |

The rank-1 form eliminates the **entire factorial bottleneck**. The architecture now scales linearly in $n$ for the invariance step, and the per-layer cost is dominated by $O(n^3)$ (matrix exponential, eigensplit) rather than $O(n!\cdot n)$.

### Corollary A.1b (No Monte-Carlo approximation needed)

Catalogue §7 states: *"Beyond $n = 8$, use Monte-Carlo Reynolds with $M \ge 10^4$ samples; invariance degrades from exact to $\varepsilon$-approximate."* This is **unnecessary** for the permutation representation. The rank-1 form is **exact** for all $n$ at cost $O(n)$. Monte-Carlo approximation is only needed for **general finite groups** where no closed-form projection is available.

### Pseudocode correction (replaces line 598 of §5.1)

```python
# ── [3] Reynolds operator — sole S_n enforcement within layer ──
# OLD (O(n!·n), unnecessary for permutation rep):
#   x = jnp.stack([rho @ x for rho in S_n], axis=0).mean(axis=0)
#
# NEW (O(n), exact by Theorem A.1):
x = (1.0 / x.shape[0]) * (jnp.conj(jnp.ones(x.shape[0])) @ x) * jnp.ones(x.shape[0])
#   equivalently:  x = ((jnp.conj(ones) @ x) / n) * ones
```

One line. Exact. $O(n)$. No group enumeration.

> **Scope note.** This theorem is specific to the **permutation representation** of $S_n$ on $\mathbb{C}^n$. For other representations (e.g. the $\binom{n}{2}$-dimensional edge representation in Appendix A, or a general finite group $G$), the Reynolds operator is a **full-rank** projection with no closed-form rank-1 reduction, and the $O(|G|\cdot n)$ or Monte-Carlo cost is genuine.

---

## A.2 — Sharp expressiveness: what the architecture actually computes

### Setup

By Theorem (ii) of Catalogue.md, after the first $P_{\mathrm{inv}}$ the state is $c_1\mathbf{1}$ with $c_1 \in \mathbb{C}$. Each subsequent layer maps

$$c_\ell\,\mathbf{1} \;\xrightarrow{\;\mathrm{LAYER}_{\ell+1}\;}\; c_{\ell+1}\,\mathbf{1},$$'

so the total architecture computes

$$c_L \;=\; \Phi_L\bigl(c_1\bigr), \qquad \Phi_L \;=\; \phi_L \circ \phi_{L-1} \circ \cdots \circ \phi_2,$$'

where each $\phi_\ell : \mathbb{C} \to \mathbb{C}$ is the per-layer scalar map.

### Lemma A.2 (Per-layer functional form)

Fix the parameters of layer $\ell$: $W_\ell \in \mathrm{U}(n)$, $B_\ell \in \mathbb{C}^{n\times n}$, $b_\ell \in \mathbb{C}^n$, $A_\ell \in \mathbb{R}$ (gauge angle), and the eigensplit routing (fixed thresholds). Define

$$u_\ell \;=\; e^{iA_\ell}\bigl(W_\ell + B_\ell W_\ell\bigr)\mathbf{1} \;\in\; \mathbb{C}^n, \qquad v_\ell \;=\; b_\ell \;\in\; \mathbb{C}^n.$$

(These absorb the $W$-rotation, gauge twist, and residual into a single affine map $c \mapsto c\,u_\ell + v_\ell$ on the pre-attention state, up to the eigensplit routing which is a **fixed** linear projection $P_\ell$ depending only on $W_\ell$'s eigenvectors, not on $c$.)

After the eigensplit projector $P_\ell$ (a fixed $n\times n$ matrix), the pre-attention state is

$$z(c) \;=\; P_\ell\bigl(c\,u_\ell + v_\ell\bigr) \;=\; c\,p_\ell + q_\ell,$$'

where $p_\ell = P_\ell u_\ell$ and $q_\ell = P_\ell v_\ell$ are fixed vectors.

The attention logits are

$$\ell_{ij}(c) \;=\; \mathrm{Re}\bigl(z_i(c)\,\overline{z_j(c)}\bigr) \;=\; \mathrm{Re}\bigl((c\,p_i + q_i)\,(c\,\overline{p_j} + \overline{q_j})\bigr),$$'

which is **quadratic** in $c$ and $\bar{c}$:

$$\ell_{ij}(c) \;=\; a_{ij}\,|c|^2 \;+\; \mathrm{Re}\bigl(b_{ij}\,c\bigr) \;+\; d_{ij},$$'

with fixed coefficients $a_{ij} = \mathrm{Re}(p_i \overline{p_j})$, $b_{ij} = p_i \overline{q_j} + \overline{q_i}\, p_j$, $d_{ij} = \mathrm{Re}(q_i \overline{q_j})$.

The softmax weights are

$$\alpha_{ij}(c) \;=\; \frac{\exp\bigl(\ell_{ij}(c)\bigr)}{\sum_k \exp\bigl(\ell_{kj}(c)\bigr)},$$'

and the post-attention state is $\sum_j \alpha_{ij}(c)\,z_j(c)$, a **smooth** (indeed real-analytic) function of $c$ and $\bar{c}$.

Finally, $P_{\mathrm{inv}}$ (rank-1 by Thm A.1) extracts the scalar:

$$\phi_\ell(c) \;=\; \frac{1}{n}\,\mathbf{1}^\* \bigl[\text{post-attention output}(c)\bigr].$$

**Conclusion.** Each $\phi_\ell : \mathbb{C} \to \mathbb{C}$ is a **smooth (real-analytic) function of $c$ and $\bar{c}$**, built from exponentials of at most quadratic polynomials in $c$, $\bar{c}$, with $O(n^2)$ fixed complex coefficients. $\blacksquare$

### Theorem A.2 (Quantitative expressiveness bound)

The $L$-layer architecture computes

$$c_L \;=\; \Phi_L(c_0), \qquad c_0 = \tfrac{1}{\sqrt{n}}\,\mathbf{1}^\* x,$$'

where $\Phi_L : \mathbb{C} \to \mathbb{C}$ is a **smooth (real-analytic) function of one complex variable** with at most

$$\mathcal{P}(L, n) \;=\; \sum_{\ell=1}^{L} \bigl(2n^2 + 2n + 1\bigr) \;\approx\; 2Ln^2$$'

**real parameters** (counting: $W_\ell$ contributes $n^2$ real params via the skew-Hermitian $A_\ell$; $B_\ell$ contributes $2n^2$; $b_\ell$ contributes $2n$; $A_\ell$ contributes $1$; the eigensplit thresholds are fixed constants, not free parameters).

**Comparison with the full invariant space.** The space of $S_n$-invariant functions on $\mathbb{C}^n$ is the ring $\mathbb{C}[e_1, \dots, e_n]$ (or $C^\infty(\mathbb{C}^n)^{S_n}$), which is **infinite-dimensional** (or has $n$ algebraically independent generators). The architecture accesses a **finite-dimensional family** of size $O(Ln^2)$ within the 1-complex-dimensional fiber over $\mathbf{1}^*$. Specifically:

| Quantity | Full invariant space | Architecture |
|---|---|---|
| Parameters | $\infty$ (or $n$ generators $e_1,\dots,e_n$) | $\mathcal{P}(L,n) \approx 2Ln^2$ |
| Input dependence | $x \in \mathbb{C}^n$ (all $n$ coordinates) | $c_0 = \frac{1}{\sqrt{n}}\mathbf{1}^\* x$ (1 scalar) |
| Function class | all smooth $S_n$-invariant functions | smooth $\Phi_L : \mathbb{C} \to \mathbb{C}$ with $O(Ln^2)$ params |
| $n=8,\;L=20$ | infinite | $\approx 2560$ real params |

The architecture is a **$\mathcal{P}(L,n)$-parameter subfamily** of the 1-dimensional fiber of the invariant function space. It is not the full fiber (which would require arbitrary smooth $\mathbb{C} \to \mathbb{C}$ functions), but it is a rich, learnable subfamily: compositions of $L$ softmax-of-quadratic maps.

### Corollary A.2a (Non-realizable functions)

No $L$-layer architecture with the catalogue's per-layer structure can realize an $S_n$-invariant function $f : \mathbb{C}^n \to \mathbb{C}$ that depends on $e_2 = \sum_{i<j} x_i x_j$ (or any elementary symmetric polynomial of degree $\geq 2$) **independently** of $e_1 = \sum_i x_i$. The architecture's output is a function of $e_1$ alone (composed through $L$ nonlinearities). In particular, two inputs $x, x'$ with $\mathbf{1}^\* x = \mathbf{1}^\* x'$ but $e_2(x) \neq e_2(x')$ are **indistinguishable** by the architecture.

**Example.** $n = 3$: $x = (1, 1, 0)$ and $x' = (0.5, 0.5, 1)$. Both have $\mathbf{1}^\* x = \mathbf{1}^\* x' = 2$, so $c_0$ is identical. But $e_2(x) = 1$ and $e_2(x') = 1.25$. The architecture assigns the same output to both. $\blacksquare$

> This is a **consequence of the architecture's design** (single $\mathbf{1}^*$-readout per layer), not a fundamental limitation of $S_n$-invariant functions. A deeper architecture that tracks all $n$ isotypic components (Appendix A direction) could in principle distinguish $e_2$ from $e_1$.

---

## A.3 — Revised complexity budget (replaces §7 row for $P_{\mathrm{inv}}$)

| Operation | Old cost | **New cost (Thm A.1)** | Note |
|---|---|---|---|
| $W = e^A$ | $O(n^3)$ | $O(n^3)$ | unchanged |
| Gauge twist | $O(n)$ | $O(n)$ | unchanged |
| Residual | $O(n^2)$ | $O(n^2)$ | unchanged |
| Eigensplit | $O(n^3)$ | $O(n^3)$ | unchanged |
| Attention | $O(n^2)$ | $O(n^2)$ | unchanged |
| **$P_{\mathrm{inv}}$** | **$O(n!\cdot n)$** | **$O(n)$** | **rank-1 projection (Thm A.1)** |
| **Total per layer** | **$O(n!\cdot n + n^3)$** | **$O(n^3)$** | **dominated by $W$ and eigensplit** |
| Total ($L$ layers) | $L\cdot O(n!\cdot n)$ | $L\cdot O(n^3)$ | $L \le 20$ |
| $n=100,\;L=20$ | $\approx 10^{158}$ ops | **$2 \times 10^7$ ops** | feasible on a single GPU |

The $n!$ bottleneck is **gone**. The architecture is now polynomial in $n$ for all $n$.

---

## A.4 — What this adds to the catalogue

| Addition | Impact |
|---|---|
| **Theorem A.1** (rank-1 $P_{\mathrm{inv}}$) | Removes the $O(n!)$ cost; architecture scales to $n = 100+$; Monte-Carlo approximation no longer needed for $S_n$ |
| **Lemma A.2 + Theorem A.2** | Replaces the qualitative "1D slice" of §4.4 with a **quantitative** $O(Ln^2)$-parameter bound and an explicit non-realizability corollary |
| **Corollary A.2a** (example) | Concrete pair of inputs the architecture **cannot** distinguish — useful for test design and for communicating the expressiveness limit |
| **Corrected pseudocode** | One-line change; exact; $O(n)$ |
| **Revised complexity table** | $n = 100$ goes from infeasible ($10^{158}$) to feasible ($10^7$) |

These additions are **strictly strengthening**: they do not change the invariance theorem (which already holds), they do not add operations, and they do not alter the minimal viable subset. They tighten the computational cost and sharpen the expressiveness statement.

---

*Extension A · 2026-11-10 · Supplement to Catalogue.md v1-final*

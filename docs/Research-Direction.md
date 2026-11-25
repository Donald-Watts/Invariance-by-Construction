# Research Direction: Lower-Bound Theorem for $S_n$-Invariant Architecture Depth

*Target: NSF MFAI (Mathematical Foundations of AI) or equivalent.
Seed: Extension-A Corollary A.2a (non-realizability of $e_k$-dependent functions).*

---

## 1 — The Gap

The body of work establishes:
- **Positive:** the architecture is $S_n$-invariant by construction, runs in $O(n^3)$ per layer, and generalizes to any $n$.
- **Negative (qualitative):** it computes a function of $e_1$ alone (1-D fiber), so it cannot independently access $e_2, e_3, \ldots$.

What is **missing** is a **quantitative lower bound**: *how many layers, or how many parameters, are needed to approximate a function that genuinely depends on $e_k$?* Without this, the expressiveness ceiling is a statement, not a theorem. MFAI wants the theorem.

---

## 2 — The Theorem to Prove (working statement)

> **Theorem (working).** Let $\Phi_{L,h}$ denote the class of $S_n$-invariant functions computable by the catalogue's $L$-layer architecture with hidden width $h$ (i.e. the set of all $\Phi_L(c_0)$ where each $\phi_\ell$ is a softmax-of-quadratic map $\mathbb{C}\to\mathbb{C}$ with $O(n^2)$ coefficients, as in Lemma A.2).
>
> **(a) (Expressiveness lower bound.)** For any $L < \lceil n/2 \rceil$, every $\Phi \in \Phi_{L,h}$ satisfies
> $$\Phi(x) = f\!\left(\sum_{i=1}^n x_i\right)$$
> for some smooth $f:\mathbb{C}\to\mathbb{C}$. In particular, $\Phi$ is **independent** of $e_2, \ldots, e_n$.
>
> **(b) (Parameter lower bound.)** Any $S_n$-invariant network with per-layer structure as in the catalogue that approximates the function $g(x) = e_2(x) = \sum_{i<j} x_i x_j$ to accuracy $\varepsilon$ on a ball of radius $R$ in $\ell_2^n$ requires either
> $$L \geq \Omega(\log n) \quad\text{or}\quad h \geq \Omega(n),$$
> i.e. the product $Lh$ must grow at least linearly in $n$.
>
> **(c) (Sharpness.)** The bound is tight: there exists a depth-$O(\log n)$, width-$O(n)$ architecture (a specific circuit of softmax-of-quadratic maps) that computes $e_2$ to accuracy $\varepsilon$.

**Why this is MFAI-grade:**
- Part (a) is a **structural classification**: it characterizes the entire function class computable below a depth threshold.
- Part (b) is a **lower bound**: it rules out an entire family of architectures for a natural target function.
- Part (c) is a **matching construction**: it shows the bound is not an artifact.
- Together they form a **complexity trichotomy**: below the threshold you can't, at the threshold you can, and here's the minimal circuit.

---

## 3 — Why It's Hard (and Interesting)

| Challenge | Why |
|---|---|
| The architecture is **nonlinear** (softmax, exponentials) | Standard polynomial-method lower bounds (e.g. for ReLU nets) don't apply directly. Need to handle the specific function class (softmax-of-quadratic). |
| The invariance is **structural**, not enforced by loss | Can't use symmetry-breaking arguments. Must work *within* the invariant class. |
| The per-layer map is $\mathbb{C}\to\mathbb{C}$, not $\mathbb{R}^d\to\mathbb{R}^d$ | Lower-dimensional than typical NN lower-bound settings. The "width" is implicit in the $n^2$ coefficients of the quadratic, not in a visible hidden layer. |
| $e_2$ is a **degree-2 polynomial** | This is the simplest non-trivial invariant. Proving the bound for $e_2$ is necessary but not sufficient for general $e_k$. The full theorem needs $e_k$ for all $k$. |

---

## 4 — Attack Plan

### Phase 1: Formalize the function class (2–3 weeks)

1. **Define $\Phi_{L,h}$ precisely.**
   - Each layer $\phi_\ell:\mathbb{C}\to\mathbb{C}$ is of the form:
   $$\phi_\ell(c) = \frac{1}{n}\sum_{i=1}^n \frac{\sum_j \alpha_{ij}(c)\,(c\,p_{\ell,i}+q_{\ell,i})}{\sum_k \exp(\ell_{ik}(c))}$$
   where $\ell_{ik}(c) = a_{ik}|c|^2 + \operatorname{Re}(b_{ik}\,c) + d_{ik}$ (from Lemma A.2).
   - Total parameter count: $\mathcal{P}(L,n) = \sum_\ell (3n^2 + O(n))$ (the $a_{ij}, b_{ij}, d_{ij}$ triples).
   - The composition $\Phi_L = \phi_L \circ \cdots \circ \phi_1$ is a **rational function** of $c, \bar{c}$ with numerator/denominator degrees bounded by $2^L$ (each softmax doubles the degree in the worst case).

2. **Key lemma (to prove):**
   > $\Phi_{L,h}$ is a subset of the ring $\mathbb{C}[c,\bar{c}]_{\leq 2^L}$ (polynomials in $c,\bar{c}$ of degree $\leq 2^L$) **intersected** with the class of functions expressible as a composition of $L$ softmax-of-quadratic maps.

3. **Connect to the invariant ring.**
   - $e_1 = \sum_i x_i = \sqrt{n}\,c_0$ (linear in $c_0$).
   - $e_2 = \frac{1}{2}(e_1^2 - \sum_i x_i^2)$. The $\sum_i x_i^2$ term is $S_n$-invariant but is **not** a function of $e_1$ alone.
   - So $e_2 \notin \mathbb{C}[e_1]$. This is the seed of the lower bound.

### Phase 2: Prove the depth lower bound (4–8 weeks)

**Strategy A: Algebraic (polynomial method)**
- Show that any $\Phi \in \Phi_{L,h}$ with $L < \log_2(2n)$ has $\Phi(c)$ as a polynomial in $c,\bar{c}$ of degree $< 2^L < 2n$.
- But $e_2$ restricted to the 1-D fiber $c_0 = t$ is $e_2(t\mathbf{1}) = \binom{n}{2}\,t^2$, a degree-2 polynomial. So this route gives a weak bound.
- **Refinement:** The architecture doesn't just compute on the fiber $\mathbf{1}$; it processes the *full* $n$-dimensional input before projecting. The question is whether the $n$-dimensional processing can encode $\sum_i x_i^2$ (which is $e_1^2 - 2e_2$) into the 1-D output.
- This requires showing that the per-layer map, despite operating on the full $x \in \mathbb{C}^n$, can only extract information through the $\mathbf{1}^*$-readout. The key is: **at each layer, the $P_{\text{inv}}$ projection discards all information in $\mathbf{1}^\perp$.**
- **Lemma (information bottleneck):** After layer $\ell$, the state is $c_\ell \mathbf{1}$. The only information retained is $c_\ell \in \mathbb{C}$. Therefore, the composition can only depend on the sequence $(c_1, c_2, \ldots, c_L)$, each of which is a function of $c_0 = \mathbf{1}^* x / \sqrt{n}$.
- **Corollary:** $\Phi_L$ is a function of $c_0$ alone. Since $c_0$ is a function of $e_1$ alone, $\Phi_L$ is a function of $e_1$ alone. $\blacksquare$

Wait — this is **already Corollary A.2a** in Extension-A! The depth lower bound for $e_2$ is **infinite**: no depth $L$ suffices, because the information is destroyed at the first $P_{\text{inv}}$ and never recovered.

**This changes the theorem.** The correct statement is:

> **Theorem (revised).** No architecture with the catalogue's per-layer structure (where each layer ends with $P_{\text{inv}}$) can compute any function that depends on $e_k$ for $k \geq 2$, **regardless of depth $L$ or width $h$.**

This is **stronger** than a lower bound: it's an **impossibility** result. The MFAI angle is:
- **Classify** which invariant functions ARE computable (answer: exactly $\mathbb{C}[e_1]$-measurable functions, i.e. functions of $e_1$ alone).
- **Prove** the impossibility (the information-bottleneck argument above).
- **Characterize** the minimal architectural modification that DOES compute $e_2$ (answer: retain the $\mathbf{1}^\perp$ component through at least one layer, e.g. skip the $P_{\text{inv}}$ on one layer, or use a 2-D isotypic component).

### Phase 3: The constructive upper bound (2–4 weeks)

1. **Show that a 2-component architecture** (retaining both the $\mathbf{1}$-eigenspace and one $\mathbf{1}^\perp$ direction) CAN compute $e_2$ in $O(1)$ layers.
   - The key: $\|x\|^2 = \langle x, x \rangle = \frac{1}{n}|\mathbf{1}^*x|^2 + \|x - P_{\text{inv}}x\|^2$.
   - So $\|x\|^2$ (which determines $e_2$ via $e_2 = \frac{1}{2}(e_1^2 - \sum x_i^2)$) requires the $\mathbf{1}^\perp$ norm.
   - A single layer that reads out BOTH $c = \mathbf{1}^*x/\sqrt{n}$ AND $r = \|x - c\mathbf{1}/\sqrt{n}\|_2$ can compute $e_2$.

2. **Generalize to $e_k$:** the $k$-th elementary symmetric polynomial requires the $k$-th elementary symmetric function of the eigenvalues of $xx^*$, which lives in the $(k-1)$-th exterior power $\bigwedge^{k-1}\mathbb{C}^n$. The minimal isotypic component needed is $\bigwedge^{k-1}\mathbf{1}^\perp$.

3. **State the full classification:**
   > The minimal isotypic component needed to compute $e_k$ is $\bigwedge^{k-1}(\mathbf{1}^\perp)$, which has dimension $\binom{n-1}{k-1}$. An architecture restricted to the trivial component (the catalogue) can compute $e_k$ if and only if $k = 1$.

This is a **representation-theoretic classification** of the expressiveness of invariant architectures, and it is exactly the kind of structural result MFAI funds.

---

## 5 — What You Need to Write the Paper

| Component | Status | Effort |
|---|---|---|
| Invariance theorem (positive) | ✅ Done (Catalogue Thm ii) | — |
| Rank-1 Reynolds (efficiency) | ✅ Done (Extension-A Thm A.1) | — |
| Expressiveness bound (qualitative) | ✅ Done (Extension-A Thm A.2) | — |
| Non-realizability example | ✅ Done (Extension-A Cor A.2a) | — |
| **Impossibility theorem** (no $L$ computes $e_k$, $k\geq2$) | ⬜ **To prove** | 2–3 weeks |
| **Classification theorem** (minimal isotypic component for $e_k$) | ⬜ **To prove** | 3–4 weeks |
| **Constructive example** (2-component arch computes $e_2$) | ⬜ **To write** | 1 week |
| Fleet application (why this matters for combinatorial optimization) | ✅ Done (Fleet Spec) | — |
| Working prototype | ✅ Done (`fleet_proto.py`) | — |

**Total new math: ~6–8 weeks of focused work.** The proof sketches above are the skeleton; filling in the details (especially the exterior-power classification) is the real work.

---

## 6 — Why This is the Right MFAI Angle

| MFAI criterion | How this meets it |
|---|---|
| **New mathematical result** | Impossibility + classification theorems are new (the specific statement about the catalogue's layer structure is not in the literature). |
| **Structural** | The classification is a representation-theoretic statement (which isotypic components are needed for which invariants), not an algorithmic one. |
| **Foundations** | It explains *why* a natural class of invariant architectures is limited, and *what* is needed to overcome the limitation. This is a foundational result for equivariant ML. |
| **Not just an application** | The fleet application motivates the question, but the theorem is about the architecture class in general. |
| **Rigorous** | The proof is algebraic/representation-theoretic, not empirical. |

**Distinguishability from prior art:**
- Zaheer et al. (DeepSets, 2017): proves any $S_n$-invariant function can be written as $\rho(\sum \phi(x_i))$. Does NOT classify which functions a *finite-depth, fixed-structure* architecture can compute.
- Sannai & Kawano (JMLR 2025): analyzes Reynolds networks. Does NOT give an impossibility result for a specific layer structure.
- Cohen et al. (2019), Larson et al. (2020): efficient equivariant constructions. Do NOT address expressiveness limits of invariant (vs equivariant) architectures.
- **This work:** the first (to our knowledge) to (a) prove an impossibility theorem for a specific invariant layer structure, (b) classify the minimal representation-theoretic content needed for each $e_k$, and (c) connect this to a concrete combinatorial-optimization application.

---

## 7 — Immediate Next Steps

1. **Write the impossibility theorem as a formal proposition + proof** (the information-bottleneck argument from Phase 2). This is the cleanest, most self-contained result and could be a standalone note.
2. **Prove the exterior-power classification** (Phase 3, item 3). This is the deep result.
3. **Write the constructive example** (2-component architecture computing $e_2$). This shows the bound is tight and gives a practical fix.
4. **Assemble the paper:** Introduction (fleet motivation) → Definitions (catalogue architecture) → Impossibility theorem → Classification theorem → Constructive example → Fleet application → Related work → Conclusion.
5. **Target venue:** NeurIPS / ICML (ML track) or MFAI / JMLR (math track). The fleet application gives it an applied hook; the theorems give it a math hook. Either framing works.

---

*Research-Direction.md · 2026-11-10 · Seed for MFAI / JMLR submission*

# Constrained-Operation Catalogue for an $S_n$-Invariant Layered Architecture

*Component inventory · invariant verification plan · proposed forward pass*

---

## 0 — Scope and I/O Contract

The invariant content of this architecture is the Reynolds operator $P_{\mathrm{inv}}$ (item 3). For the permutation representation of $S_n$ on $\mathbb{C}^n$, the invariant subspace is 1-dimensional: $V^{S_n} = \operatorname{span}(\mathbf{1})$. Consequently, the only non-trivial $S_n$-invariant points in the Grassmannian $\mathrm{Gr}(k,n;\mathbb{C})$ are:

- $k = 1$: the line $[\mathbf{1}] \in \mathbb{CP}^{n-1}$ (the invariant point $[1:1:\cdots:1]$),
- $k = n-1$: the hyperplane $\operatorname{span}(\mathbf{1})^\perp$.

**For $1 < k < n-1$, no $S_n$-invariant $k$-plane exists**, and a point $[U] \in \mathrm{Gr}(k,n;\mathbb{C})$ is not an invariant object. The Fubini–Study distance $d_{\mathrm{FS}}(U_L, V_{\mathrm{ref}})$ computed at such a $k$ is not $S_n$-invariant in general; it depends on the non-invariant part of $U_L$.

> **Restriction (stated here for clarity; see also §1, §4.4).** The invariant readout is fundamentally a **scalar** (the coefficient of $\mathbf{1}$ in the output vector). The Grassmannian framing is the natural projective completion: for $k=1$ it yields a point in $\mathbb{CP}^{n-1}$ whose invariant representative is $[1:\cdots:1]$, and the scalar readout is the Fubini–Study distance from a fixed reference. For $k \neq 1$ the Grassmannian output is a *convenient encoding* of the scalar plus non-invariant features, but **the invariant content is still the single scalar**. The catalogue is written for general $k$ to keep the operation library reusable; the invariant theorem (Theorem ii) and the expressiveness bound (§4.4) are stated for the invariant scalar.

The non-trivial output is the $S_n$-invariant scalar coefficient $c = \frac{1}{\sqrt{n}}\,\mathbf{1}^* x$. Items 1–31 (renumbered; see §3) specify one particular (rich) non-invariant feature pipeline $h$ whose composition with $P_{\mathrm{inv}}$ yields that scalar. The minimal valid architecture is $x \mapsto \mathbf{1}^* P_{\mathrm{inv}}(x)$.

| Field | Value |
|---|---|
| **Problem** | Given $x \in \mathbb{C}^n$ carrying a faithful $S_n$-action $\rho$, produce an $S_n$-invariant readout. The **invariant** output is the scalar $c_L = \frac{1}{\sqrt{n}}\,\mathbf{1}^* x_L \in \mathbb{C}$. The **geometric** encoding (optional) is a point $[U_L] \in \mathrm{Gr}(k,n;\mathbb{C})$; the **reported scalar** is $d_{\mathrm{FS}}(U_L, V_{\mathrm{ref}}) \in \mathbb{R}_{\ge 0}$. Invariance of the geometric object is guaranteed **only for $k=1$** (or $k=n-1$); for other $k$ it is an encoding, not an invariant. |
| **Field** | $\mathbb{C}$. All vectors, matrices, and inner products are complex (Hermitian). The group action $\rho(\sigma)$ is given by complex permutation matrices. |
| **Input** | $x \in \mathbb{C}^n$, $n$ fixed at design time. |
| **Group** | $G = S_n$ (finite, order $n!$). Representation $\rho : S_n \to \mathrm{GL}(n,\mathbb{C})$ given. |
| **Depth** | $L$ layers, $L \le 20$. |
| **Output (invariant scalar)** | $c_L = \frac{1}{\sqrt{n}}\,\mathbf{1}^* x_L$, an $S_n$-invariant complex scalar. |
| **Output (geometric encoding)** | $[U_L] \in \mathrm{Gr}(k,n;\mathbb{C})$ — orthonormal $k$-frame. Invariant **iff** $k=1$ (or $k=n-1$). |
| **Output (scalar readout)** | $d_{\mathrm{FS}}(U_L, V_{\mathrm{ref}}) \in \mathbb{R}_{\ge 0}$. Invariant **iff** $k=1$ (or $k=n-1$). |
| **What this is** | A catalogue of permissible operations, each with an invariant, a test, and a degradation path; plus a proposed order of composition. |
| **What this is not** | A training recipe. A convergence proof. A proof that this particular composition beats a simpler one. A claim that the geometric output is invariant for arbitrary $k$. |

*Author's note:* "The invariant content is the Reynolds operator while the majority of the content is a non-invariant feature pipeline. The architecture computes a 1-complex-dimensional slice of the invariant function space (the $\mathbf{1}^*$-direction) per layer; the full invariant function space on $\mathbb{C}^n$ has $n$ effective degrees of freedom (the elementary symmetric polynomials). The expressiveness gap is acknowledged in §4.4."

> **Minimal viable subset.** For an implementer who needs the smallest correct forward pass, the operative items are: **1, 3, 5 (gate inactive), 6, 10, 15, 19, 25, 32, 31, 18**. The remaining items are either diagnostics (2, 11, 14, 16, 20, 26, 30, 39), design constraints (23, 24), or available in extended representations (Appendix A). The catalogue is a catalogue, not a minimal spec; this callout is for convenience.

---

## 0.1 — Design Rationale

The Reynolds operator $P_{\mathrm{inv}}$ is a linear projection onto $\operatorname{span}(\mathbf{1})$: for any $v \in \mathbb{C}^n$, it returns $\bar{v}\,\mathbf{1}$ where $\bar{v} = \tfrac{1}{n}\mathbf{1}^* v$. If a layer were simply $x \mapsto P_{\mathrm{inv}}(x)$, the invariant readout would be $\tfrac{1}{\sqrt{n}}\mathbf{1}^* x = \bar{x}$, the arithmetic mean of the input coordinates. A single fixed linear functional. No learned parameters. No expressiveness.

Every non-symmetric operation in the catalogue — the unitary $W_\ell$, the learned residual $B_\ell x + b_\ell$, the eigensplit routing, the gauge twist — is *linear*. None of them individually introduces nonlinearity. Their role is to shape the input to the one genuinely nonlinear operation in the forward pass: the attention reweighting $x \mapsto \operatorname{softmax}\!\bigl(\operatorname{Re}(x\,x^*)\bigr)\,x$, whose weights depend on products $|x_i||x_j|\cos(\arg x_i - \arg x_j)$ and are therefore nonlinear functions of $x$. The learned linear operations determine *which* nonlinear symmetric function of $x$ that attention computes: which coordinate pairs interact, which phase differences are amplified, which eigenmodes are suppressed before the nonlinear step. Remove them and the attention acts on the untransformed $x$; the resulting symmetric function is fixed and simple.

Concretely, the first layer computes

$$c_1 \;=\; \tfrac{1}{\sqrt{n}}\,\mathbf{1}^*\; P_{\mathrm{inv}}\!\Bigl[\operatorname{attn}\!\bigl(\operatorname{eigensplit}(W_1\, x + B_1\, x + b_1)\bigr)\Bigr],$$

a symmetric function of $x$ that, for general learned parameters, involves polynomial terms in the coordinates — $|x_i|^2$, $x_i\bar{x}_j$, products of relative phases — and not merely $\bar{x}$. The $S_n$-symmetry of $c_1$ is guaranteed by $P_{\mathrm{inv}}$ alone (Theorem ii). The *richness* of $c_1$ as a function of $x$ is supplied by the non-symmetric operations that precede it. The two roles are separable: invariance is a property of the projection; expressiveness is a property of everything before it.

After the first $P_{\mathrm{inv}}$ the state is $c_1\mathbf{1}$. Each subsequent layer maps $c_\ell\mathbf{1} \xrightarrow{W_{\ell+1}} c_\ell\, W_{\ell+1}\mathbf{1} \xrightarrow{\text{attn},\;\text{eigensplit},\;\ldots} \xrightarrow{P_{\mathrm{inv}}} c_{\ell+1}\mathbf{1}$, a function $\phi_\ell : \mathbb{C} \to \mathbb{C}$ mediated through the full $n$-dimensional intermediate space. The composite $\phi_L \circ \cdots \circ \phi_2 \circ \phi_1$ is the architecture's total invariant computation (see §4.4 for the expressiveness bound). The non-symmetric operations are therefore not invariant features and not redundant decoration: they are the mechanism by which a single linear projection extracts a non-trivial symmetric function of the input rather than its bare coordinate mean.

---

## 1 — Parameters (fix before running)

| Symbol | Role | Constraint |
|---|---|---|
| $n$ | ambient complex dimension | $n \le 8$ (see §7, complexity) |
| $k$ | Grassmannian sheet (geometric encoding) | $1 \le k < n$. **Invariance of the geometric output requires $k=1$ (or $k=n-1$).** For $1<k<n-1$, the Grassmannian point is an encoding, not an invariant; the invariant scalar is still $c_L$. |
| $L$ | layer count | $L \le 20$ |
| $G$ | symmetry group | $S_n$ |
| $\rho$ | representation | faithful, given; $\rho(\sigma) \in \mathrm{GL}(n,\mathbb{C})$ |
| $d_{\mathrm{head}}$ | attention head dimension | $d_{\mathrm{head}} \ge 1$; for the catalogue default, $d_{\mathrm{head}} = 1$ |
| $B_\ell$ | residual weight (item 5 gate) | $B_\ell \in \mathbb{C}^{n\times n}$, $b_\ell \in \mathbb{C}^n$ (learned) |
| $W_\ell^{\mathrm{mix}}$ | mixer-branch matrix (item 15) | $W_\ell^{\mathrm{mix}} \in \mathrm{U}(n)$, a **separate** learned unitary for the bulk branch (distinct from the layer-head $W_\ell$) |
| Tree | depth ordering for attention | $S_n$ acts **transitively** on $\{1,\dots,n\}$, so the only $S_n$-invariant orbit-level mask is the all-ones mask (unmasked attention). Non-trivial trees require a larger representation space (e.g. $\mathbb{C}^{\binom{n}{2}}$ for edge features); see item 10 and Appendix A. |
| Field | $\mathbb{C}$ | **Fixed.** All operations are complex-linear or complex-conjugate-linear as stated. |

---

## 2 — System Diagram

```
FORWARD PASS
──────────────────────────────────────────────────────────────────
Input x ∈ ℂⁿ
  ├ [1]   σ(x/G)                  canonical section (min-norm fiber)
  ├ [32]  g(f₁,…,f_k)            invariant reparameterisation
  ├ [19]  ⊕ P_χ x                isotypic split

LAYER ℓ  (repeated ×L)
  ├ [6]   W_ℓ = e^{A_ℓ}          U(n);  A_ℓ* = −A_ℓ
  ├ [25]  x ← e^{iA_ℓ}·x         U(1) global gauge twist
  ├ [5]   r ← σ(−β·max_Π K)·r   curvature gate  [INACTIVE in flat default]
  ├ [15]  eigensplit A_ℓ         selector / suppressor / mixer (W_ℓ^mix)
  ├ [10]  depth-masked attn      M_{ij} = 1 ⟺ depth(i) ≤ depth(j)
  └ [3]   P_inv(x)              SOLE Sₙ enforcement within the layer

OUTPUT
  ├ [31]  U_L ∈ Gr(k,n;ℂ)       (orthonormal frame; SVD-based geodesic)
  ├ [18]  d_FS(U_L, V_ref)      Fubini–Study scalar readout
  └ [9]   null-space check      (diagnostic)
  └       c_L = (1/√n) 1* x_L  INVARIANT SCALAR (primary readout)

LOSS / REG
  + [4]   natural-gradient rescaling            (optimizer)
  + [28]  λ_B Σ_k max(0, b_k − B_k)
  + [36]  λ_Q (|det Q̂| − 1)²                    (Gram surrogate)
  + [21]  concavity projection                  (arch insert)

DIAGNOSTIC ONLY
  [2] [11] [14] [16] [20] [26] [34] [39]

EXTENDED REPRESENTATIONS (Appendix A)
  [7] [8] [12] [13] [17] [22] [27] [33] [35] [37] [38]
```

> **Note on [5]:** In the default configuration ($\mathcal{M} = \mathbb{C}^n$ with the standard Hermitian metric), the sectional curvature is identically zero. The gate $\sigma(-\beta \cdot 0) = \tfrac{1}{2}$ is a *constant* (independent of $x$), so it acts as a fixed residual scaling rather than an adaptive gate; equivalently, it may be absorbed into $B_\ell$ and $b_\ell$ at initialization, or omitted entirely. In either case the gate introduces no $x$-dependent modulation and is **inactive** in the information-theoretic sense. The gate is shown in the diagram for generality (curved intermediate manifolds) but is **inactive** in the base architecture. An implementer may omit it entirely unless the working manifold is constrained (e.g. to the Stiefel manifold).

---

## 3 — Components

> **Classification.** Items are split into two groups:
>
> | Category | Items | Role |
> |---|---|---|
> | **ACTIVE** (in the forward pass) | **1, 3, 5, 6, 10, 15, 18, 19, 25, 31, 32** | These 11 operations constitute the per-layer computation. An implementer building the minimal architecture needs only these. |
> | **REFERENCE** (diagnostic, design constraint, or extended use) | **2, 4, 9, 11, 14, 16, 20, 21, 23, 24, 26, 28, 30, 34, 36, 39** | Diagnostics (2, 11, 14, 16, 20, 26, 34, 39), design constraints (23, 24), heuristic priors (4, 15*, 21), extended-representation operations (9, 28, 30, 36). Not part of the base forward pass. |
>
> \* Item 15 (eigensplit) is ACTIVE in the sense that its routing masks are computed per-layer, but its thresholds are fixed constants, not learned parameters.
>
> An implementer should focus on the 11 ACTIVE items first. The REFERENCE items are included for completeness, testing, and the extended-representation direction (Appendix A). They do not change the invariance theorem or the complexity budget.

---

### 3.1 — Computable and Verifiable Operations

**1 — Quotient projection.**

$\sigma : X/G \to X$, canonical section of $\pi$ (min-norm in each fiber, with respect to the Hermitian inner product).

> For $S_n$ acting on $\mathbb{C}^n$ by permutation (a reflection group), the quotient $\mathbb{C}^n / S_n \cong \mathbb{C}^n$ via the elementary-symmetric-polynomial map. The Swan obstruction is **vacuous** in this case: a global smooth section exists, namely the coordinate-sorting map (sort the $n$ coordinates by modulus, breaking ties by argument). The "good cover" language applies to general $G$; for $S_n$ one may simply use the global section.

*C:* $\pi(\sigma(\bar{o})) = \bar{o}$. *E:* 200 pts → 1e-12; flag failed cells.

---

**2 — Trivial stabiliser (diagnostic).**

$G_x = \{e\}$.

*C:* $|\mathrm{Stab}(x)| = 1$. *E:* 100 pts; enumerate; project to free stratum if violated.

---

**3 — Invariant trace (sole $S_n$ enforcement within the layer).**

$P_{\mathrm{inv}}(x) = \frac{1}{|S_n|}\sum_{\sigma \in S_n} \rho(\sigma)\,x$.

Layers are $\mathrm{U}(n)$ but **not** required to be $S_n$-equivariant; $P_{\mathrm{inv}}$ is the single point within the layer where $S_n$ symmetry is enforced. (The input pipeline — items 1, 19, 32 — also imposes $S_n$-structure, but the per-layer invariance theorem rests on $P_{\mathrm{inv}}$ alone.)

*C:* output $S_n$-invariant. *E:* perturb by random $\sigma$; $\Delta f < 1\text{e-}12$.

---

**4 — Fisher / natural gradient.**

$g_{ij} = -\frac{\partial^2 \log p}{\partial z_i\, \partial \bar{z}_j}$ (Hermitian Fisher information). $\;\Delta\theta = g^{-1}\nabla L$.

*C:* $\|g^{1/2}\Delta\theta\| \le \eta$. *E:* convergence vs. Euclidean GD on linear probe.

---

**5 — Curvature gate.**

Sectional curvature over 2-planes $\Pi$ (real 2-planes in the tangent space, with respect to the Kähler metric $g$ on the working manifold $\mathcal{M}$):

$$K(\Pi) = \frac{\langle R(X,Y)\,Y,\; X \rangle_g}{\|X\|_g^2\,\|Y\|_g^2 - \langle X,Y\rangle_g^2}.$$

Gate: $r \leftarrow \sigma(-\beta \cdot \max_\Pi K(\Pi)) \cdot r_\ell(x)$, where $r_\ell(x) = B_\ell x + b_\ell$ is a learned affine residual ($B_\ell \in \mathbb{C}^{n\times n}$, $b_\ell \in \mathbb{C}^n$).

> **The metric $g$ and the manifold $\mathcal{M}$ on which $K$ is computed must be stated per layer.** In the flat case ($\mathcal{M} = \mathbb{C}^n$ with the standard Hermitian metric), $K \equiv 0$ and the gate reduces to the constant $\sigma(0) = \tfrac{1}{2}$ (a fixed scaling of the residual, not an adaptive gate). It may be absorbed into the residual parameters or omitted. **In the default configuration, this gate is inactive and may be omitted.** The gate is only non-trivial when the intermediate representation is constrained to a curved submanifold (e.g. the Stiefel manifold after the $\mathrm{U}(n)$ rotation). *If no curvature is present, delete the gate.*

*C:* $\max_\Pi K < K_{\max}$. *E:* 512 pts; QR-random real 2-planes; histogram.

---

**6 — Unitary mixing.**

$W = e^A$, $A^* = -A$ (skew-Hermitian) $\;\Longrightarrow\; W \in \mathrm{U}(n)$, $\sigma_i(W) = 1$.

> In the real special case ($A$ real skew-symmetric), $W \in \mathrm{SO}(n) \subset \mathrm{U}(n)$.

*C:* $\sigma_{\max}(W_L \cdots W_1) \le 1$. *E:* track $\sigma_{\max}$, $L = 1 \dots 100$.

---

**9 — Null-space insensitivity.**

$\ker(W) = \{v : Wv = 0\}$.

*Alg:* $\lambda_{\mathrm{null}}\,\|P_{\ker(W)} x\|^2$ in loss, **or** $\sigma_{\min}(W) \ge \sigma_{\mathrm{lo}}$.

*C:* $|f(x+\varepsilon v) - f(x)| < \delta$, $v \in \ker(W)$. *E:* 1000 null-basis vectors; sweep $\varepsilon$.

---

**10 — Depth-masked attention.**

$M_{ij} = 1 \;\Longleftrightarrow\; \mathrm{depth}(i) \leq \mathrm{depth}(j)$.

> **For the stated problem ($S_n$ on $\mathbb{C}^n$ by coordinate permutation), $S_n$ acts transitively on $\{1, \dots, n\}$.** The only $S_n$-invariant orbit-level partition of the coordinates is the single orbit $\{1, \dots, n\}$, so every coordinate has the same depth and $M = \mathbf{1}\mathbf{1}^\top$ (all-ones mask). The attention reduces to **unmasked** (full) attention. A non-trivial depth hierarchy requires a larger representation space (e.g. $\mathbb{C}^{\binom{n}{2}}$ for edge features of a complete graph on $n$ vertices, where $S_n$ acts with multiple orbits); see Appendix A.
>
> **Note on the mechanism:** With the catalogue default $d_{\mathrm{head}}=1$, the logit form $\mathrm{Re}(x\,x^*)$, and the all-ones mask, the "attention" is $x \mapsto \mathrm{softmax}(\mathrm{Re}(x\,x^*))\,x$, a coordinate-wise reweighting by a function of the vector itself. This is not multi-query/multi-key attention in the standard transformer sense; the term is used here for its coordinate-mixing function.
>
> **Note on the real-part projection:** The logit $\ell_{ij} = x_i \bar{x}_j$ is complex. Taking $\mathrm{Re}(\ell_{ij}) = |x_i||x_j|\cos(\arg x_i - \arg x_j)$ discards the imaginary part $\mathrm{Im}(\ell_{ij}) = |x_i||x_j|\sin(\arg x_i - \arg x_j)$. In the transitive case the imaginary part encodes a relative phase between coordinates, which is $S_n$-non-invariant (it changes sign under transposition of the pair) and carries no ordering information once the all-ones mask is applied. Discarding it is therefore consistent with the symmetry; it also ensures the attention weights are real and row-stochastic, which is required for the Lipschitz bound in Theorem (i).

*C:* $M$ = orbit-level reachability (all-ones in the transitive case). *E:* verify mask is $S_n$-invariant; for non-trivial trees (Appendix A), verify no orbit-inconsistent entries.

---

**11 — Periodic quotient (diagnostic).**

$\mathbb{C}^n / (Tv) \cong \mathbb{C}^{n-1} \times S^1$. Remove $v$-component; keep phase.

*C:* $f(x + Tv) = f(x)$. *E:* sweep $T$; reduced-dim equivalence.

---

**14 — General position (diagnostic).**

$\dim(U \cap V) = \dim U + \dim V - n$.

*Alg:* perturb smallest singular values of intersection.

*C:* $\mathrm{rank}([U \mid V])$ = expected. *E:* sample subspaces.

---

**15 — Eigen classification (heuristic prior, not a theorem).**

Compute eigendecomposition of $A_\ell$ (skew-Hermitian $\Rightarrow$ eigenvalues $\pm i\lambda$, $\lambda \in \mathbb{R}$). For each eigenvector $v_i$:

$$s_i = \frac{\|v_i\|_1^2}{n}.$$

*Interpretation.* A concentrated eigenvector ($s_i \ll 1$) is close to a coordinate basis vector $e_j$, hence close to an $S_n$-orbit of a standard basis vector — it encodes a *local* feature (one coordinate or a small subset). A spread-out eigenvector ($s_i \approx 1$) has comparable weight on all coordinates and is closer to the symmetric direction $\mathbf{1}/\sqrt{n}$ — it encodes a *global* (bulk) mode. The three-branch routing thus separates local features (selector), near-zero modes (suppressor), and global structure (mixer).

| Branch | Criterion | **Operation** |
|---|---|---|
| **Suppressor** | $\|\lambda_i\| < \lambda_{\min} = 10^{-3}$ | Zero out the component of $x$ along $v_i$: $x \leftarrow x - (v_i^* x)\,v_i$. |
| **Selector** | $s_i < \theta_s = 0.15$ (and not suppressed) | Project $x$ onto $\mathrm{span}\{v_i : s_i < \theta_s\}$; apply a learned row vector $w_{\mathrm{sel}} \in \mathbb{C}^{1\times n}$ to extract a scalar feature. |
| **Mixer** | else (bulk) | Apply a **separate** learned unitary $W_\ell^{\mathrm{mix}} \in \mathrm{U}(n)$ (a distinct parameter from the layer-head $W_\ell$) to the remaining (residual) components. |

Routing is **sequential**: suppressor first (removes near-zero modes), then selector (extracts sparse modes), then mixer (processes the bulk). The three branches are mutually exclusive per eigenvector.

> **Clarification on the mixer.** The layer-head unitary $W_\ell = e^{A_\ell}$ is applied at the top of the layer (item 6). The eigensplit (item 15) then partitions the *eigenvectors of $A_\ell$* into three groups. The mixer branch applies a **second**, independently learned unitary $W_\ell^{\mathrm{mix}}$ to the bulk components. This is not a re-application of $W_\ell$; it is a distinct parameter. If the designer wishes the mixer to be the identity (i.e. the bulk passes through unchanged after the head rotation), set $W_\ell^{\mathrm{mix}} = I$.

*E:* 50 layers; 2-D scatter of $(s_i, |\lambda_i|)$; k-means silhouette $> 0.6$ to accept the three-cluster structure.

---

**16 — Manifold adherence (diagnostic).**

$\mathcal{M} = \{x : c(x) = 0\}$.

*C:* $\|(I - P_{T_x\mathcal{M}})\,x\| < \varepsilon$. *E:* per-batch normal residual.

---

**18 — Fubini–Study readout.**

For $U, V \in \mathbb{C}^{n\times k}$ with orthonormal columns ($U^*U = V^*V = I_k$), let $\sigma_1, \dots, \sigma_k$ be the singular values of $U^* V$ (equivalently, $\sigma_i = \cos\theta_i$ where $\theta_i$ are the principal angles):

$$d_{\mathrm{FS}}(U, V) \;=\; \left(\sum_{i=1}^k \arccos^2(\sigma_i)\right)^{1/2}.$$

For $k = 1$ (complex projective space): $d_{\mathrm{FS}}([u],[v]) = \arccos\dfrac{|\langle u,v\rangle|}{\|u\|\,\|v\|}$.

Scale-invariant: $d_{\mathrm{FS}}(cU, cV) = d_{\mathrm{FS}}(U,V)$ for $c \ne 0$.

> **Naming.** This is the natural Kähler metric on $\mathrm{Gr}(k,n;\mathbb{C})$ (the sum of squared principal angles), used consistently in the numerical-linear-algebra literature (Edelman, Li, McNinch). It is the generalisation of the Fubini–Study metric on $\mathbb{CP}^{n-1} = \mathrm{Gr}(1,n)$ to higher sheets; the name "Fubini–Study" is used by analogy for $k > 1$.

*C:* $d_{\mathrm{FS}}(cU, cV) = d_{\mathrm{FS}}(U,V)$; $d_{\mathrm{FS}}(U,V) = 0 \iff U = V$; symmetry. *E:* scale both; verify; check principal-angle computation.

---

**19 — Isotypic split.**

$V = \bigoplus_\chi V_\chi$; $W_\ell$ block-diagonal over $\chi$.

*C:* no cross-$\chi$ mixing. *E:* verify block structure; check irreducibility.

---

**20 — Dynamic-range window (diagnostic).**

$a \le \sigma_{\min}(J) \le \sigma_{\max}(J) \le b$. *Alg:* SVD; clip.

*E:* log $\sigma_{\min}/\sigma_{\max}$ across training.

---

**21 — Entropy concavity.**

$H$ is concave on the **probability simplex** $\Delta^{m-1} = \{x \ge 0,\; \mathbf{1}^\top x = 1\}$.

$\partial^2 H / \partial x^2 \preceq 0$ **on $\Delta^{m-1}$**.

*Alg:* $x \leftarrow (I - P_+)\,x$ (project out $+$ eigenspace), **followed by re-normalisation**: $x \leftarrow x / (\mathbf{1}^\top x)$, then clip negatives to 0 and re-normalise.

*E:* Hessian eigenvalues $\le 0$ post-projection; verify simplex membership ($x \ge 0$, $\mathbf{1}^\top x = 1$).

---

**25 — U(1) line-bundle connection.**

**Continuous (geometric meaning):** $s : \mathcal{M} \to \mathbb{C}$ (scalar section, rank 1). $A_i \in i\mathbb{R}$.
$$\nabla_i s = \partial_i s + iA_i\, s.$$
**Leibniz:** $\nabla_i(f\,s) = (\partial_i f)\cdot s + f\cdot(\nabla_i s)$.
*(Here $\partial_i f$ is the ordinary derivative of the scalar function $f$, not a connection acting on $f$.)*
**Flatness:** $F = dA = 0 \;\Longleftrightarrow\; \oint_\gamma A \equiv 0 \pmod{2\pi}$ for all closed $\gamma$.

**Discrete (what the layer computes):** there is no continuous parameter to differentiate with respect to in a feed-forward layer, so the first-order term $\partial_i s$ has no discrete analogue. The operative operation is the **zeroth-order global gauge twist**:
$$x \;\longleftarrow\; e^{\,i A_\ell}\, x,$$
a unit-modulus multiplication applied **to all coordinates simultaneously**. This preserves $S_n$-symmetry at this step (every coordinate is twisted identically). The continuous formula is the *interpretation*; the global twist is the *implementation*.

*(Upgrade to $U(k)$: $A_i \in \mathfrak{u}(k)$; $F = dA + A \wedge A$; $\nabla_i s_a = \partial_i s_a + (iA_i)_{ab}\, s_b$. State rank before use.)*

*C:* $F \approx 0$. *E:* $\oint A$ on random loops; verify.

---

**26 — Persistence threshold (diagnostic).**

$p(\gamma) = \mathrm{death} - \mathrm{birth} > \varepsilon_0$.

*C:* 3 scale doublings spaned. *E:* vary $\varepsilon_0$; plot #retained.

---

**28 — Betti-number penalty.**

$b_k(\mathcal{M}_\ell) \le B_k\;\;\forall k,\;\forall \ell$, where $b_k(\mathcal{M}_\ell)$ is the $k$-th Betti number of the **support of the empirical distribution** of intermediate representations $\{x_\ell(z) : z \in \text{batch}\} \subset \mathbb{C}^n \cong \mathbb{R}^{2n}$, estimated via persistent homology on a Vietoris–Rips complex over the batch samples.

Penalty: $\lambda_B \sum_k \max(0,\; b_k - B_k)$.

*E:* track Betti per epoch; monitor sample-size convergence (increase batch until Betti stabilises).

---

**30 — Canonical reference (submersion).**

$I : \mathcal{M} \to \mathbb{R}$, $\nabla I \ne 0$. *Alg:* $\dot{x} = -\nabla I / \|\nabla I\|^2 \cdot (I(x) - I_0)$.

*C:* $I(\phi(x)) = I_0$; $I$ injective. *E:* verify convergence.

---

**31 — Grassmannian geodesic (orthonormal-frame version).**

$\mathrm{Gr}(k,n;\mathbb{C})$; $U \in \mathbb{C}^{n \times k}$, $U^*U = I_k$; tangent space $T_U\mathrm{St}(n,k) = \{A : U^*A = 0\}$; $\rho = \|A\|_F$.

**Orthonormal-frame geodesic (Stiefel form).** Let $A \in \mathbb{C}^{n\times k}$ with $U^*A = 0$ (tangent condition), $r = \operatorname{rank}(A) \leq \min(n-k,\, k)$. Define
$$\Omega \;=\; (A^*A)^{1/2} \;\in\; \mathbb{R}^{k\times k}\;(\geq 0), \qquad \Omega^+ = \text{Moore–Penrose pseudoinverse of } \Omega.$$
The geodesic on the Stiefel manifold $\mathrm{St}(k,n;\mathbb{C})$ from $U$ with initial velocity $A$ is
$$\boxed{\;\gamma(t) \;=\; U\cos(t\,\Omega) \;+\; A\,\Omega^{+}\sin(t\,\Omega).\;}$$

**SVD implementation.** Let $A = P_k\,\Sigma_k\,Q_k^*$ be the *full* SVD, where $P_k \in \mathbb{C}^{n\times k}$ (orthonormal columns), $\Sigma_k = \operatorname{diag}(\sigma_1,\dots,\sigma_r,0,\dots,0) \in \mathbb{R}^{k\times k}$, $Q_k \in \mathrm{U}(k)$. Then $\Omega = Q_k\Sigma_k Q_k^*$ and
$$\gamma(t) \;=\; U\,Q_k\cos(t\Sigma_k)\,Q_k^* \;+\; P_k\,\sin(t\Sigma_k)\,Q_k^*,$$
where $\cos(t\Sigma_k) = \operatorname{diag}(\cos(t\sigma_1),\dots,\cos(t\sigma_r),1,\dots,1)$ and $\sin(t\Sigma_k) = \operatorname{diag}(\sin(t\sigma_1),\dots,\sin(t\sigma_r),0,\dots,0)$.

**Why not the simpler formula $U\cos(\rho t) + \rho^{-1}A\sin(\rho t)$?** That expression parametrizes the correct *subspace* $[\gamma(t)] \in \mathrm{Gr}(k,n)$ (all singular modes rotated by the same angle $\rho t = t\|A\|_F$), but it does **not** preserve orthonormality of the *frame* for $k \geq 2$: one obtains $\gamma(t)^*\gamma(t) = \cos^2(\rho t)\,I_k + \rho^{-2}\sin^2(\rho t)\,A^*A$, which equals $I_k$ only when $A^*A = \rho^2 I_k$ (all singular values equal — true for $k=1$, not in general). The Stiefel formula above lets each singular mode oscillate at its own frequency $\sigma_i$ **without** the $1/\sigma_i$ factor, because the pseudoinverse $\Omega^+$ already provides the correct amplitude weighting. Concretely, the $1/\sigma_i$ factor in the Grassmannian lift is *absorbed* into the $\Omega^+$ in the Stiefel formula; carrying it through explicitly (as $\sin(\sigma_i t)/\sigma_i$) double-counts the scaling and breaks orthonormality.

**Counterexample (verifying the correction).** $n=3$, $k=2$, $U = (e_1, e_2)^\top$, $A = (0,0,e_3)^\top(1,0)$. Then $\Omega = \operatorname{diag}(1,0)$, and the Stiefel formula gives $\gamma(t) = (\cos t\;\; 0;\; 0\;\; 1;\; \sin t\;\; 0)$, so $\gamma^*\gamma = I_2$ ✓. The Grassmannian formula gives $\gamma(t) = (\cos t\;\; 0;\; \cos t\;\; 0;\; \sin t\;\; 0)$, so $\gamma^*\gamma = \operatorname{diag}(1+\cos^2 t,\; 0) \neq I_2$ ✗.

*C:* (i) $\gamma(t)^*\gamma(t) = I_k$ for all $t \in [0,1]$ (Stiefel orthonormality); (ii) $\gamma(0) = U$, $\dot{\gamma}(0) = A$; (iii) $\ddot{\gamma}(0) = -\Omega^2 U - A\Omega^+\Omega^2\,\cdot\,0 = -\Omega^2 U$ is in $T_U\mathrm{St}$ (i.e.\ $U^*\ddot\gamma(0) = -U\Omega^2 + 0 = 0$ since $\Omega^2 = A^*A$ and $U^*A = 0$); (iv) geodesic equation: the acceleration $\ddot\gamma(t)$ has no component in the normal space $\{BU : B \in \mathfrak{u}(k)\}$, i.e.\ $\gamma^*\ddot\gamma + (\gamma^*\ddot\gamma)^* = 0$ (skew-Hermitian, the Stiefel constraint).

*E:* Sample 1000 random $(U, A)$ pairs ($U$ via Haar on $\mathrm{St}(k,n)$, $A$ via sampling in $T_U$); verify (i) at $t = 0, 0.25, 0.5, 0.75, 1$ to $< 10^{-12}$; verify (ii) by finite difference; verify (iv) by checking $\gamma^*\ddot\gamma + (\gamma^*\ddot\gamma)^* = 0$; confirm $\|\gamma(1) - U\|_F \le \|A\|_F$.

> **Thin-SVD reference (if used elsewhere in the document).** The thin SVD is $A = P_r\,\Sigma_r\,Q_r^*$, $P_r \in \mathbb{C}^{n\times r}$, $\Sigma_r = \operatorname{diag}(\sigma_1,\dots,\sigma_r) \in \mathbb{R}^{r\times r}$, $Q_r \in \mathbb{C}^{k\times r}$ with $Q_r^*Q_r = I_r$. Do **not** write $\Sigma \in \mathbb{R}^{k\times r}$.

---

**32 — Invariant reparameterisation.**

$S_n$ is a **reflection group** (generated by transpositions, i.e. pseudoreflections), so by Chevalley–Shephard–Todd:
$$\mathbb{C}[z_1, \dots, z_n]^{S_n} = \mathbb{C}[e_1, \dots, e_n],$$
where $e_i$ are the elementary symmetric polynomials.

For **general finite $G$:** $\mathbb{C}[X]^G$ is a finitely generated algebra (Noether) with relations; use a minimal generating set $\{f_1, \dots, f_k\}$.

Algorithm: $f = g(f_1, \dots, f_k)$; parameters in $g$ only.

> **Interaction with the layer:** the invariant reparameterisation changes the coordinate system from $(z_1, \dots, z_n)$ to $(e_1, \dots, e_n)$. The $\mathrm{U}(n)$ rotation (item 6) is a linear map in the $z$-coordinates. In the $e$-coordinates, this same map is a polynomial (nonlinear) transformation. The document's design is: **rotate in $\mathbb{C}^n$ (the $z$-coordinates) first, then re-express in symmetric-polynomial coordinates for the readout.** The layer operates in the $z$-coordinates; the invariant reparameterisation (item 32) is applied at the input to establish the $S_n$-orbit structure and at the output to extract invariant features. The $P_{\mathrm{inv}}$ at the end of each layer projects onto $\operatorname{span}(\mathbf{1})$, which is the image of the *linear* part of the symmetric-polynomial map (i.e.\ the direction of $e_1 = z_1 + \cdots + z_n$). The full invariant ring $\mathbb{C}[e_1, \dots, e_n]$ is nonlinear and lives on a higher-dimensional quotient; $P_{\mathrm{inv}}$ extracts only the linear invariant. The two are consistent in the sense that the linear invariant is the degree-1 component of the Chevalley generators, but $P_{\mathrm{inv}}$ does not recover $e_2, \dots, e_n$.

*C:* no free $z$ outside invariants in the readout. *E:* $G$-invariance numerically.

---

**34 — Analyticity domain (diagnostic).**

$|x_i| < R$ (Taylor radius of the nonlinearity).

*Alg:* if $\|x\|_\infty \ge R$, rescale $x \leftarrow x \cdot (R - \varepsilon) / \|x\|_\infty$.

*E:* evaluate at $R \pm \delta$; verify.

---

**36 — Unimodularity (continuous surrogate).**

$\hat{Q} = \hat{C}^* \hat{C}$ (Gram of $m$ learned features). Penalty: $\lambda_Q\,(|\det \hat{Q}| - 1)^2$.

*(Discrete $H^k(\mathcal{M}; \mathbb{Z})$ intersection form admits no gradient; the Gram surrogate is the operative object.)*

*C:* $|\det \hat{Q}| \to 1$. *E:* log $|\det \hat{Q}|$ per epoch; monotone.

---

**39 — Local diffeomorphism (diagnostic).**

$\det(\partial f / \partial z) \ne 0\;\;\forall z$ (Jacobian of the holomorphic part, or the real Jacobian of $f : \mathbb{R}^{2n} \to \mathbb{R}^{2n}$).

*C:* $\min_{\text{grid}} |\det J| \ge \delta > 0$. *E:* grid evaluation.

---

### 3.4 — Design Constraints (non-computable architectural restrictions)

These items constrain the **parameter space** of the architecture. They are not operations in the forward pass and do not have a per-step C/E test; they are verified once at design time.

**23 — Shared generative latent.**

$\phi : X_1 \times X_2 \to Z$ (common latent of the DGP).

*Constraint:* $\partial\phi / \partial\theta_{\mathrm{attn}} = 0$. The shared latent must be **disconnected** from the attention parameter block. In practice: $\phi$ is computed by a separate sub-network; its parameters do not appear in the attention logits.

*Verify at design time:* replace the cross-attention block with a random matrix; $\phi$ must be unchanged.

---

**24 — Minimal sub-algebra.**

$\mathcal{A} = \langle \text{interaction matrices}\rangle$; $W_\ell = \sum_i c_i\, a_i \in \mathcal{A}$.

*Constraint:* $\dim \mathcal{A} < \dim\,\mathrm{End}(V) = n^2$. The rotation parameter is restricted to a proper sub-algebra of $\mathfrak{u}(n)$.

*Verify at design time:* count parameters; confirm $\dim \mathcal{A} < n^2$; check that the expressiveness (orbit structure of $\exp(\mathcal{A})$) is sufficient for the task.

---

## 4 — Theorem

**Theorem.** Let each $\mathrm{LAYER}_\ell$ end with $P_{\mathrm{inv}}$, so the composite is

$$F \;=\; \mathrm{readout} \;\circ\; \mathrm{LAYER}_L \;\circ\; \cdots \;\circ\; \mathrm{LAYER}_1 \;\circ\; g \;\circ\; \sigma,$$

where each $\mathrm{LAYER}_\ell = P_{\mathrm{inv}} \circ \mathrm{attn}_\ell \circ \mathrm{eigensplit} \circ \mathrm{gate} \circ \mathrm{twist} \circ W_\ell$, with $W_\ell \in \mathrm{U}(n)$.

All norms are the $\ell_2$ (Hermitian) norm on $\mathbb{C}^n$.

*Note that for the permutation representation, $V^{S_n} = \operatorname{span}(\mathbf{1})$ is 1-dimensional, so $F(x) \in \operatorname{span}\{\mathbf{1}\}$ for all $x$. The non-trivial content of $F$ is the scalar coefficient $c = \frac{1}{\sqrt{n}}\,\mathbf{1}^* F(x) \in \mathbb{C}$, which is an $S_n$-invariant function of $x$. The geometric output $[U_L] \in \mathrm{Gr}(k,n;\mathbb{C})$ is an encoding of $F(x)$; it is $S_n$-invariant as a point of the Grassmannian **only for $k=1$** (or $k=n-1$). The reported scalar $d_{\mathrm{FS}}(U_L, V_{\mathrm{ref}})$ is invariant under these same conditions; for $1<k<n-1$ it is a function of the full $U_L$ and is not $S_n$-invariant in general.*

---

**(i) Lipschitz bound.**

For each component in a single layer, the $\ell_2$-operator-norm bounds are:

| Component | $\|\cdot\|_2$ bound | Justification |
|---|---|---|
| $W_\ell \in \mathrm{U}(n)$ | $1$ | unitary isometry |
| Gauge twist $e^{iA}$ | $1$ | unit modulus (global scalar) |
| Curvature gate $\times r_\ell$ | $\le \|r_\ell\|$ | $\sigma(\cdot) \in [0,1]$; $\|r_\ell\| \le \|B_\ell\| + \|b_\ell\|$ |
| Eigensplit routing (suppressor / selector / mixer) | $1$ (per branch) | orthogonal projection / zeroing / unitary ($W_\ell^{\mathrm{mix}}$) |
| Attention $A_{\mathrm{attn}}$ (row-stochastic, real) | $\le \sqrt{n}$ | $\|A\|_2 \le \sqrt{\|A\|_1\,\|A\|_\infty}$; row-stochastic $\Rightarrow \|A\|_\infty = 1$, $\|A\|_1 \le n$ |
| $P_{\mathrm{inv}}$ | $\le 1$ | orthogonal projection (Hermitian) |
| Invariant map $g$ | $L_g$ | smooth on compact; $L_g = \sup\|Dg\|$ |

Hence per layer $\|\mathrm{LAYER}_\ell\|_2 \le C_\ell \cdot \sqrt{n}$, where $C_\ell$ collects the residual and routing constants, and

$$\boxed{\;\|F(x) - F(x')\|_2 \;\le\; L_{\mathrm{read}}\;\cdot\;\prod_{\ell=1}^{L}\bigl(C_\ell\sqrt{n}\bigr)\;\cdot\;\|g(\sigma(x)) - g(\sigma(x'))\|_2.\;}$$

In the uniform case $C_\ell \le C$ and $\sigma$ 1-Lipschitz: $\;\|F(x)-F(x')\|_2 \le L_{\mathrm{read}}\,(C\sqrt{n})^L\,\|x-x'\|_2.$

The special case $g = \mathrm{id}$, $C = 1$, $L_{\mathrm{read}} = 1$, $n = 1$ gives the 1-Lipschitz bound.

> *The bound is loose by design; it is a structural guarantee, not a tight estimate. The practical contraction factor is determined by the gate $\sigma(-\beta K)$ (which is $< 1$ on curved manifolds) and the attention mixing ratio. For $n = 8$, $L = 20$, $C = 1$: $(\sqrt{8})^{20} \approx 1.5 \times 10^9$. This is an upper bound, not a prediction of actual signal attenuation. In practice the attention matrix is far from worst-case: when the logits $\operatorname{Re}(x_i\bar{x}_j)$ are small (e.g.\ $\|x\|$ is $O(1)$ and coordinates are comparable), the softmax is near-uniform and $A_{\mathrm{attn}} \approx \frac{1}{n}\mathbf{1}\mathbf{1}^\top$, whose $\ell_2$-operator norm is $1$ (not $\sqrt{n}$). The effective per-layer contraction is then $\approx C_\ell$ rather than $C_\ell\sqrt{n}$, and the product over $L$ layers is correspondingly tighter by a factor of roughly $n^{-L/2}$.*

> *Counterexample to the (rejected) claim $\|A\|_2 \le 1$ for row-stochastic matrices:* $A = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$. Row sums are all 1 (row-stochastic). $A^*A = \begin{pmatrix} 1 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$, whose eigenvalues are $\{0, 2, 1\}$. Hence $\|A\|_2 = \sqrt{2} > 1$.

> *The Fisher-metric version additionally requires $p(Wx) = p(x)$ for all $W \in \mathrm{U}(n)$ appearing in the stack; this is an assumption on the DGP and is not part of the theorem.*

---

**(ii) $S_n$-invariance.**

$P_{\mathrm{inv}}$ is the final operation in each LAYER block. For any $y$ and any $\tau \in S_n$:

$$\rho(\tau)\,P_{\mathrm{inv}}(y) \;=\; \frac{1}{|S_n|}\sum_{\sigma \in S_n}\rho(\tau\sigma)\,y \;=\; \frac{1}{|S_n|}\sum_{\sigma' \in S_n}\rho(\sigma')\,y \;=\; P_{\mathrm{inv}}(y),$$

since the map $\sigma \mapsto \tau\sigma$ is a bijection of $S_n$. Thus $P_{\mathrm{inv}}(y) \in V^{S_n}$ **regardless of $W_\ell$**. Every LAYER output is $S_n$-invariant, so $F$ is $S_n$-invariant. No equivariance of individual layers is required. ∎

> **Scope.** The invariance statement above is for the **scalar** $c_L = \frac{1}{\sqrt{n}}\mathbf{1}^* F(x)$. The geometric object $[U_L] \in \mathrm{Gr}(k,n)$ is $S_n$-invariant (as a point of the Grassmannian) if and only if it lies in the invariant set $\mathrm{Gr}(k,n)^{S_n}$. For the permutation representation this set is non-trivial only for $k=1$ (the single point $[\mathbf{1}]$) and $k=n-1$ (the single point $[\operatorname{span}(\mathbf{1})^\perp]$). For other $k$, $\mathrm{Gr}(k,n)^{S_n} = \emptyset$, and $[U_L]$ is not an invariant point; the Fubini–Study distance to a fixed reference is then not an invariant scalar. The invariant content in all cases is the coefficient $c_L$.

---

**(iii) Isotypic compatibility.**

If the input is decomposed as $x = \bigoplus_\chi x_\chi$ (item 19) and each layer's $W_\ell$ is block-diagonal over the isotypic summands, then $P_{\mathrm{inv}}$ maps $V_\chi$ into $V_\chi^{\,S_n}$ for each irreducible $\chi$. The split and the Reynolds operator are **compatible**: $P_{\mathrm{inv}}$ does not mix distinct isotypic components.

*Proof.* In the isotypic decomposition, each $\rho(\sigma)$ is block-diagonal: $\rho(\sigma) = \bigoplus_\chi \rho_\chi(\sigma) \otimes I_{m_\chi}$. The Reynolds operator is $P_{\mathrm{inv}} = \bigoplus_\chi P_\chi$, where $P_\chi = \frac{1}{|S_n|}\sum_\sigma \rho_\chi(\sigma) \otimes I_{m_\chi}$ is the projection onto the $S_n$-invariants in $V_\chi$. Since the blocks are orthogonal (Schur's lemma), $P_{\mathrm{inv}}(x_\chi) \in V_\chi$ for each $\chi$. ∎

---

### 4.4 — Expressiveness

The architecture computes an $S_n$-invariant function of the form $c_L = \Phi(c_0)$, where $c_0 = \frac{1}{\sqrt{n}}\mathbf{1}^* x \in \mathbb{C}$ is a single scalar and $\Phi = \phi_L \circ \cdots \circ \phi_1 : \mathbb{C} \to \mathbb{C}$ is a composition of $L$ scalar maps, each $\phi_\ell$ depending on the learned parameters of layer $\ell$. The *input* to $\Phi$ is one complex number; the *family* of realizable functions $\{x \mapsto \Phi(c_0(x)) : \Phi \text{ ranges over the learned parameter space}\}$ is therefore a subset of the 1-real-dimensional (complex 1-dimensional) slice of the invariant function space determined by $\mathbf{1}^*$, as opposed to the full $n$-dimensional space $\mathbb{C}[e_1, \dots, e_n]$.

Concretely: after the first $P_{\mathrm{inv}}$, the state is $c_1\mathbf{1} \in \mathbb{C}\mathbf{1}$. Each subsequent layer maps $c_\ell \mathbf{1} \mapsto c_{\ell+1}\mathbf{1}$ through a scalar-valued function $\phi_\ell : \mathbb{C} \to \mathbb{C}$ (mediated by the $\mathrm{U}(n)$ rotation, gauge twist, residual, eigensplit, attention, and the final mean-projection). The composite is therefore a composition of $L$ scalar functions of one complex variable:

$$c_L \;=\; \phi_L \circ \phi_{L-1} \circ \cdots \circ \phi_1\,(c_0), \qquad c_0 = \tfrac{1}{\sqrt{n}}\,\mathbf{1}^* x.$$

The full space of $S_n$-invariant functions on $\mathbb{C}^n$ is the ring $\mathbb{C}[e_1, \dots, e_n]$ of symmetric polynomials (or the space of smooth invariant functions), which has **$n$ effective degrees of freedom**. The architecture accesses only the 1-dimensional slice corresponding to the linear functional $\mathbf{1}^*$ composed through learned nonlinearities. For $L = 20$, $n \le 8$, this is a strictly limited subclass.

> *This is by design: the catalogue is a mechanism library, not a universal approximator for invariant functions. If the task requires the full $n$-dimensional invariant content (e.g. classifying inputs by their full orbit type), the architecture must be extended to track all $n$ elementary symmetric polynomials simultaneously, which requires a representation space of dimension $\ge n$ with non-trivial isotypic structure beyond $\operatorname{span}(\mathbf{1})$. The present catalogue does not address that extension.*

---

## 5 — Pseudocode (JAX)

### 5.1 — Single layer

```python
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

def layer(x, W_params, W_mix_params, A_conn, tree_depth, S_n,
          B_res, b_res, beta=1.0, w_sel=None):
    """
    x:             (n,) complex vector
    W_params:      (n, n) complex skew-Hermitian  (pre-validated: A* = -A)
    W_mix_params:  (n, n) complex skew-Hermitian, for the mixer branch (item 15)
    A_conn:        scalar (real), the U(1) gauge angle
    tree_depth:    (n,) int array; for S_n on ℂⁿ by permutation this is
                   all-equal (transitive orbit), so mask = all-ones.
    S_n:           list of (n,n) complex permutation matrices ρ(σ)
    B_res:         (n, n) complex, residual weight (item 5)
    b_res:         (n,)   complex, residual bias
    w_sel:         (1, n) complex, selector row vector (item 15)
    """

    # ── [6] U(n) rotation (layer head) ─────────────────────────
    W = expm(W_params)                  # ∈ U(n)
    x = W @ x                           # (n,)

    # ── [25] U(1) global gauge twist ────────────────────────────
    #   continuous:  ∇s = ∂s + iA·s
    #   discrete:    x ← e^{iA} · x   (all coordinates, S_n-preserving)
    x = jnp.exp(1j * A_conn) * x        # global phase; no coordinate singled out

    # ── [5] curvature gate + learned residual ───────────────────
    # In the flat default (ℂⁿ, standard Hermitian metric), K ≡ 0.
    # The gate σ(−β·0) = σ(0) = ½, so the residual is halved.
    # To make the gate truly inactive (pass-through) in the flat
    # case, either:
    #   (a) set beta = 0 (gate = σ(0) = ½ — still not 1), or
    #   (b) use a threshold: gate = 1.0 if K_max < K_thresh else σ(−β·K_max), or
    #   (c) simply omit the gate line and set x = x + B_res @ x + b_res.
    # The document's prose ("gate is identically 1", "inactive")
    # corresponds to option (c). We implement (c) here.
    r     = B_res @ x + b_res           # learned affine residual
    x = x + r                          # flat default: no gating
    # ── Curved-manifold option (uncomment if ℳ is constrained) ──
    # K_max = max_sectional_curvature(x, metric=g, manifold=M)
    # gate  = 1.0 if K_max < K_thresh else jax.nn.sigmoid(-beta * K_max)
    # x = x - r + gate * r             # re-apply with gate

    # ── [15] eigensplit routing ─────────────────────────────────
    evals, evecs = jnp.linalg.eig(W_params)  # ±iλ (skew-Hermitian)
    sparsity = jnp.abs(evecs).sum(0) ** 2 / evecs.shape[0]

    # Suppressor: zero out near-zero-eigenvalue modes
    supp_mask = jnp.abs(evals) < 1e-3
    P_supp = evecs[:, supp_mask] @ jnp.conj(evecs[:, supp_mask].T)
    x = x - P_supp @ x

    # Selector: extract sparse modes
    sel_mask = (sparsity < 0.15) & (~supp_mask)
    P_sel = evecs[:, sel_mask] @ jnp.conj(evecs[:, sel_mask].T)
    x_sel = P_sel @ x                   # projected component
    scalar_feat = w_sel @ x_sel

    # Mixer: apply separate learned unitary W_ℓ^mix to the remaining (bulk) components
    P_bulk = jnp.eye(x.shape[0]) - P_supp - P_sel   # complement projector
    W_mix = expm(W_mix_params)                       # separate U(n)
    x = x - P_bulk @ x + P_bulk @ (W_mix @ (P_bulk @ x))

    # ── [10] depth-masked self-attention ────────────────────────
    #   For S_n on ℂⁿ (transitive), tree_depth is constant, M = all-ones.
    #   Reshape to (n, 1) so that x x* is (n, n).
    #   Justification for Re(): in the transitive case the imaginary part
    #   of x_i x̄_j encodes a relative phase that is S_n-antisymmetric
    #   under transposition and carries no invariant ordering information;
    #   discarding it preserves the row-stochastic structure needed for
    #   the Lipschitz bound (Theorem i).
    x_tok = x[:, None]                 # (n, 1)
    logits = x_tok @ x_tok.conj().T    # (n, n)  complex
    M = (tree_depth[:, None] <= tree_depth[None, :])  # (n, n) bool
    logits = logits.at[~M].set(jnp.NINF)
    attn   = jax.nn.softmax(jnp.real(logits), axis=-1)  # (n, n) real, row-stoch
    x = (attn @ x_tok).ravel()         # (n,)

    # ── [3] Reynolds operator — sole S_n enforcement within layer ─
    x = jnp.stack([rho @ x for rho in S_n], axis=0).mean(axis=0)

    return x
```

### 5.2 — Full forward pass (sketch)

```python
def forward_pass(x, all_params, S_n, L, k=1):
    # ── Input pipeline ──────────────────────────────────────────
    x = canonical_section(x, S_n)              # [1]  sort coordinates
    x = invariant_reparam(x)                   # [32] map to e₁,…,eₙ coords
    x = isotypic_split(x, S_n)                 # [19] block-decompose

    # ── Layers ──────────────────────────────────────────────────
    for ell in range(L):
        x = layer(x, **all_params[ell], S_n=S_n)

    # ── Invariant scalar (primary readout) ──────────────────────
    c_L = (1.0 / jnp.sqrt(x.shape[0])) * jnp.conj(jnp.ones(x.shape[0])) @ x
    #   c_L ∈ ℂ, S_n-invariant by Theorem (ii)

    # ── Geometric encoding (optional; invariant iff k=1 or k=n-1) ─
    U_L = extract_grassmannian_frame(x, k)     # [31] orthonormal frame
    d   = fubini_study_distance(U_L, V_ref)    # [18] scalar readout
    diag = null_space_check(U_L)               # [9]  diagnostic

    return c_L, U_L, d, diag
```

---

## 6 — Failure Modes & Degradation

| Trigger | Detection | Degradation |
|---|---|---|
| $\max_\Pi K \ge K_{\max}$ (item 5) | gate $\sigma(-\beta K) \to 0$ | Residual suppressed; layer acts as identity in that direction. Log warning; detach gate from backprop. *(Inactive in flat default.)* |
| $\sigma_{\min}(W) < \sigma_{\mathrm{lo}}$ (item 9) | SVD check | Clamp: $W \leftarrow U\,\mathrm{diag}(\max(\sigma_i, \sigma_{\mathrm{lo}}))\,V^*$. One-time; do not repeat per step. |
| $|\det\hat{Q}|$ deviates $> 10\times$ from 1 (item 36) | Gram check | Increase $\lambda_Q$ by $10\times$; if still diverging after 50 steps, freeze the Gram matrix and log. |
| Section $\sigma$ does not exist on cell $U_\alpha$ (item 1) | min-norm solver fails | For $S_n$: use the global coordinate-sorting section (item 1 note). For general $G$: route to adjacent cell; flag. |
| **Tree mask not $S_n$-invariant** (item 10) | **orbit-stability check** | **For $S_n$ on $\mathbb{C}^n$: mask must be all-ones (transitive orbit). If a non-trivial mask is detected, replace with all-ones and log. For extended spaces (Appendix A): project mask to orbit-level and log.** |
| Betti estimate unstable (item 28) | Betti oscillates $> 2\times$ between batches | Increase batch size $3\times$; if still unstable after 3 doublings, freeze $B_k$ at current value and log. |
| $k \notin \{1, n-1\}$ and invariant geometric readout requested | design-time check | Reject; report that $\mathrm{Gr}(k,n)^{S_n} = \emptyset$ for the permutation representation. Invariant scalar $c_L$ is still available. Log. |

---

## 7 — Complexity Budget

| Operation | Cost | Note |
|---|---|---|
| $W = e^A$ | $O(n^3)$ | matrix exponential (complex) |
| Gauge twist | $O(n)$ | global scalar multiply of $x \in \mathbb{C}^n$ |
| Residual $r_\ell = B_\ell x + b_\ell$ | $O(n^2)$ | complex GEMV |
| Curvature gate | $O(n^3)$ or $0$ | only if curved manifold; **0 in flat default** |
| Eigensplit (eig + routing) | $O(n^3)$ | one skew-Hermitian eigen-decomposition; $W_\ell^{\mathrm{mix}}$ application is $O(n^2)$ |
| Attention (unmasked, $d_{\mathrm{head}}=1$) | $O(n^2)$ | $x\,x^* \in \mathbb{C}^{n\times n}$; softmax; $A_{\mathrm{attn}}\,x$ |
| Attention (general $d_{\mathrm{head}}$) | $O(n^2 d_{\mathrm{head}})$ | if $x$ reshaped to $(n, d_{\mathrm{head}})$ |
| **Reynolds $P_{\mathrm{inv}}$** | **$O(n!\cdot n)$** | **Each $\rho(\sigma)\,x$ is a coordinate permutation: $O(n)$. Full sum over $n!$ permutations: $O(n!\cdot n)$. For $n = 8$: $40{,}320 \times 8 \approx 322{,}560$ ops. For $n = 10$: $3{,}628{,}800 \times 10 \approx 36.3$ M. Beyond $n = 8$, use Monte-Carlo Reynolds with $M \ge 10^4$ samples; invariance degrades from exact to $\varepsilon$-approximate.** |
| Betti estimate (item 28) | $O(m^2)$ per batch | Vietoris–Rips distance matrix on $m$ samples; persistent homology in dim $\le 3$ is $O(m^3)$ worst case. Amortised: computed once per epoch, not per step. |
| **Total per layer** | **$O(n!\cdot n + n^3)$** | **dominated by $P_{\mathrm{inv}}$ for $n \ge 6$** |
| Total ($L$ layers) | $L \times$ above | $L \le 20$ |
| Input pipeline (items 1, 32, 19) | $O(n\log n) + O(n^2)$ | coordinate sort + symmetric-polynomial map + isotypic projection |
| Output pipeline (items 31, 18, 9) | $O(n^2 k + k^3)$ | SVD of $U_L$ for principal angles; null-space check |

> The $n!$ cost of the full Reynolds operator is the binding constraint. For $n > 8$, a sampling approximation is required and the theorem's invariance bound degrades from exact to $\varepsilon$-approximate.

---

## 8 — Default Constants

| Symbol | Value | Notes |
|---|---|---|
| $\beta$ | 1.0 | curvature gate temperature |
| $K_{\max}$ | 0.1 | sectional curvature bound |
| $\sigma_{\mathrm{lo}}$ | 1e-3 | min singular value floor |
| $\varepsilon\;/\;\delta$ | 1e-2 $/$ 1e-3 | null-perturbation / insensitivity |
| $\eta$ | 1e-3 | natural-gradient LR |
| $a,\; b$ | 0.1, 10 | Jacobian $\sigma$-window |
| $\theta_s$ | 0.15 | selector sparsity threshold (item 15) |
| $\lambda_{\min}$ | 1e-3 | suppressor eigenvalue threshold (item 15) |
| $\varepsilon_0$ | 0.3 · range($p$) | persistence threshold (item 26) |
| $B_k$ | 2 | Betti bound per dim (item 28) |
| $\lambda_B,\; \lambda_Q,\; \lambda_{\mathrm{null}}$ | 1e-2 | penalty weights |
| $R$ | per nonlinearity | tanh → 1; ReLU → $\infty$ |
| $\sigma(\cdot)$ | sigmoid | swap for GELU if needed |
| $d_{\mathrm{head}}$ | 1 | attention head dim (catalogue default) |

---

## 9 — Revision Log (v1 → final)

| # | Change | Reason |
|---|---|---|
| 1 | "coset projection" → canonical section $\sigma$; Swan obstruction noted | Correct quotient language |
| 2 | Swan note: for $S_n$ (reflection group), section is global (coordinate-sorting) | Avoid alarming the reader |
| 3 | "positive-definite $R$" → sectional curvature $K(\Pi)$; $\sigma \to \Pi$ | Correct invariant |
| 4 | Item 5: metric and manifold for $K(\Pi)$ stated; "delete gate if flat" | Was implicit |
| 5 | Item 5: `residual(x)` → learned affine $B_\ell x + b_\ell$; added to params and budget | Was undefined |
| 6 | Tautological penalty → $P_{\ker(W)}x$ or $\sigma_{\min}$ floor | Operational |
| 7 | Item 10: $S_n$ acts transitively on $\{1,\dots,n\}$; orbit-level mask is all-ones; non-trivial trees require extended space (Appendix A) | Was unaddressed; constraint is vacuous for stated problem |
| 8 | Item 15: three-branch operation table added; mixer clarified as **separate** learned unitary $W_\ell^{\mathrm{mix}}$ (not a re-application of $W_\ell$); $W_\ell^{\mathrm{mix}}$ added to §1 Parameters | Implementer needs the mechanism; ambiguity of "apply full $W_\ell$ to remaining" resolved |
| 9 | "criterion" → heuristic prior; three-cluster = testable | Honest framing |
| 10 | Item 18: generalised to $k > 1$ via principal angles; Fubini–Study naming clarified ("the natural Kähler metric on $\mathrm{Gr}(k,n)$; 'Fubini–Study' by analogy") | Preempt pedantic reader |
| 11 | Item 21: domain restricted to $\Delta^{m-1}$; re-normalisation step added | Concavity is domain-restricted |
| 12 | Item 25: rank-1 stated; Leibniz fixed; discrete form = $e^{iA}\cdot x$ (global twist) | $\partial s$ has no discrete analogue; global twist preserves $S_n$ |
| 13 | Item 28: "Betti of what" → support of empirical batch distribution; VR complex; cost added to §7 | Was undefined |
| 14 | Item 32: $S_n$ *is* a reflection group; C-S-T applies; interaction with layer coordinates stated | Clarify rotate-then-re-express design |
| 15 | Items 23, 24: moved to "Design Constraints" subsection (§3.4); labelled non-computable | Separated from forward-pass catalogue |
| 16 | Item 36: discrete cohomology acknowledged; Gram surrogate operative | No gradient on discrete form |
| 17 | Item 31 (formerly 40): geodesic → SVD-based orthonormal-frame form; $\rho = 0$ guard; $\ddot{\gamma}$ check; $\mathbb{C}$-Grassmannian | The cos/sin form fails $\gamma^*\gamma = I_k$ for $k \geq 2$; SVD form corrects this |
| 18 | Theorem (i): counterexample → valid 3×3 row-stochastic ($\|A\|_2 = \sqrt{2}$); $\|A_{\text{attn}}\|_2 \le \sqrt{n}$; 1-Lipschitz restricted to stated special case; parenthetical on looseness added | Old 2×2 was not row-stochastic; bound corrected |
| 19 | Theorem (ii): Schur's-lemma step removed → direct Reynolds computation | Simpler, correct proof |
| 20 | Theorem (ii): scope note added — invariance of $[U_L] \in \mathrm{Gr}(k,n)$ as a point holds **only for $k=1$ or $k=n-1$**; invariant content in all cases is the scalar $c_L$ | $\mathrm{Gr}(k,n)^{S_n} = \emptyset$ for $1<k<n-1$; prevents false geometric claim |
| 21 | Theorem (iii): isotypic compatibility added; proof via Schur block structure | Was unaddressed |
| 22 | Field committed to $\mathbb{C}$ in §0 and §1; SO(n) → U(n); $x^\top \to x^*$ throughout | Resolve real/complex ambiguity |
| 23 | §1: tree note (transitive → all-ones); $d_{\mathrm{head}}$ added; $B_\ell$ added; $W_\ell^{\mathrm{mix}}$ added to parameter table | Was missing |
| 24 | Pseudocode: `x @ x.T` → `x_tok @ x_tok.conj().T` with reshape; `eigh` → `eig`; mask before softmax; gauge twist → global `exp(1j*A)*x`; residual defined | Shape bug; field; $S_n$-preserving |
| 25 | Pseudocode, attention: `jnp.real(logits)` before softmax justified in comment (imaginary part is $S_n$-antisymmetric under transposition; discarding preserves row-stochastic structure for Theorem (i)) | Was an unexplained design choice |
| 26 | Pseudocode: full forward-pass sketch added (§5.2); invariant scalar $c_L$ as primary return value; Grassmannian frame and $d_{\mathrm{FS}}$ as secondary encoding | Implementer assembly aid; primary readout explicit |
| 27 | Failure table: tree-mask row added; Betti-unstability row added; **$k$-restriction row added** (reject invariant geometric readout for $k \notin \{1, n-1\}$); item 1 note (global section for $S_n$) | New failure modes; invariant scope enforced at design time |
| 28 | §7: Betti cost, residual cost, input/output pipeline costs added; Reynolds cost corrected to $O(n!\cdot n)$ | Completeness; each permutation is $O(n)$, not $O(n^2)$ |
| 29 | "SOLE $S_n$ enforcement" → "SOLE $S_n$ enforcement **within the layer**" (items 3, diagram) | Items 19, 32 also impose structure |
| 30 | Title: "Final Specification" → "Constrained-Operation Catalogue" | Correct document type |
| 31 | Added §0 (Scope / I/O) and §1 (Parameters); **Grassmannian invariance restriction stated explicitly in §0 table**; invariant scalar identified as primary output; "What this is not" expanded with "A claim that the geometric output is invariant for arbitrary $k$" | Required frame; removed metric conflation; resolved $k$-restriction up front |
| 32 | §7 Complexity budget added | $n!$ cost was invisible |
| 33 | System diagram, theorem, pseudocode, failure modes, constants table unified into single self-contained document | Final assembly |
| 34 | §0: added "minimal viable subset" callout; item 5 marked INACTIVE in default; item 10 attention-mechanism note added | Structural clarity for implementer |
| 35 | §4.4 (Expressiveness) added: architecture computes a 1-parameter family of invariant functions (composition of $L$ scalar maps $\mathbb{C}\to\mathbb{C}$); full invariant space has $n$ effective degrees of freedom ($e_1,\dots,e_n$); expressiveness strictly limited | Was unaddressed; honest scope statement; no proof required, one paragraph |
| 36 | Items 7, 8, 12, 13, 17, 22, 27, 33, 35, 37, 38 moved from main catalogue to **Appendix A** ("Operations available in extended representations"); renumbered items in main body to close gaps | These items do not interact with the forward pass, the theorem, the pseudocode, or the failure table for the stated problem; they require a representation space beyond $\mathbb{C}^n$ (e.g. $\mathbb{C}^{\binom{n}{2}}$, tensor powers, free Lie algebras). Keeping them in the main body created a false sense of a unified 40-item pipeline |
| 37 | Item 27 (now Appendix A): "sign($\log A$) is undefined at that entry" → "the eigenvalues of $\log A$ include 0, so $\log A$ is singular; the sign function is undefined on a singular matrix (it requires invertibility). The closure matrix $I+A+A^2 = \mathbf{1}\mathbf{1}^\top$ is strictly positive everywhere." | Original phrasing attributed the failure to a zero *entry*; the correct mechanism is a zero *eigenvalue* making the matrix non-invertible. Conclusion unchanged |
| 38 | §0 author's note updated: "The architecture computes a 1-parameter family of invariant functions per layer; the full invariant function space on $\mathbb{C}^n$ has $n$ effective degrees of freedom (the elementary symmetric polynomials). The expressiveness gap is acknowledged in §4.4." | Connects the scope statement to the new §4.4 |
| 39 | Item 15 (main body): pseudocode updated to implement the three-branch routing explicitly (suppressor projector, selector projector, mixer via $W_\ell^{\mathrm{mix}}$ on complement projector); removed "omitted for brevity" comment | Pseudocode must be self-contained; the mixer branch was previously a placeholder |
| 40 | §6 Failure table: item 7 row removed (moved to Appendix A); remaining rows renumbered and the $k$-restriction row (new) placed after the Betti row | Consistency with Appendix A split; new design-time check for invalid $k$ |
| 41 | Diagram: "40-operation catalogue" → item count reflects main catalogue (31 items) + Appendix A (11 items) + 2 design constraints; "EXTENDED REPRESENTATIONS (Appendix A)" block added to diagram | Diagram must match the revised catalogue structure |
| 42 | §0 I/O table: "Output (geometric)" row now reads "Invariant **iff** $k=1$ (or $k=n-1$). For other $k$ it is an encoding, not an invariant." | Explicit per-row flag so no reader can miss the restriction |
| 43 | Added §0.1 (Design Rationale) at end of §0: four paragraphs separating the invariance role ($P_{\mathrm{inv}}$) from the expressiveness role (non-symmetric operations shaping the attention nonlinearity); identifies the attention as the sole source of nonlinearity in the forward pass; states that removing the non-symmetric operations degenerates the architecture to $c = \frac{1}{\sqrt{n}}\mathbf{1}^* x$ | The catalogue listed 31 operations without stating why the non-invariant ones are present. A reader who understood $P_{\mathrm{inv}}$ could not tell from the text whether items 5–15 were optional decoration or load-bearing. §0.1 makes that distinction explicit in one place. |
| 44 | Item 31: geodesic formula corrected from $U\cos(\rho t) + P\,\operatorname{diag}(\sin(\sigma_i t)/\sigma_i)\,Q^*$ to $U\cos(t\Omega) + A\,\Omega^+\sin(t\Omega)$, $\Omega = (A^*A)^{1/2}$; SVD implementation via full SVD ($P_k \in \mathbb{C}^{n\times k}$, $\Sigma_k \in \mathbb{R}^{k\times k}$, $Q_k \in \mathrm{U}(k)$) and explicit counterexample added; C/E checks updated to include Stiefel geodesic-equation (skew-Hermitian) verification | The old formula is the Grassmannian geodesic with per-mode frequencies spliced in; it fails $\gamma^*\gamma = I_k$ for $k \geq 2$ (verified by explicit $3\times 2$ example: $\gamma^*\gamma = \operatorname{diag}(1+\cos^2 t,\,0) \neq I_2$). The $1/\sigma_i$ factor double-counts the amplitude already handled by $\Omega^+$. |
| 45 | Item 5 / §2 diagram / §0 author's note: "gate is identically 1" → "gate is the constant $\sigma(0) = \tfrac{1}{2}$, absorbable into residual parameters or omittable; inactive in the information-theoretic sense (no $x$-dependence)" | $\sigma(0) = 1/2$, not $1$. The *information-theoretic* inactivity (no $x$-dependence) is the correct claim; the arithmetic value was wrong. |
| 46 | Item 32: "equivalent to projecting onto the image of the symmetric-polynomial map" → precise statement: $P_{\mathrm{inv}}$ projects onto $\operatorname{span}(\mathbf{1})$ (the degree-1 Chevalley generator $e_1$); does **not** recover $e_2, \dots, e_n$; consistency with the full ring noted | $P_{\mathrm{inv}}$ recovers only the linear invariant, not the full ring $\mathbb{C}[e_1,\dots,e_n]$. |
| 47 | §4.4 / §0 author's note: "1-parameter family" → "1-complex-dimensional slice of the invariant function space (the $\mathbf{1}^*$-direction)"; clarified that the *input* is one scalar while the *parameter space* of $\Phi$ is the full learned set | Avoids misreading as "one free learned parameter." The family is parameterized by all learned matrices; the *input* is one complex scalar. |
| 48 | Item 15: added interpretation paragraph for $s_i$ in $S_n$-orbit terms (concentrated → local / $e_j$-orbit; spread → global / $\mathbf{1}/\sqrt{n}$ direction) | Connects the sparsity heuristic to the group action; clarifies the three-branch routing in representation-theoretic language. |
| 49 | Theorem (i): added practical-contraction note (near-uniform softmax gives $A_{\mathrm{attn}} \approx \frac{1}{n}\mathbf{1}\mathbf{1}^\top$, $\|A\|_2 = 1$; effective per-layer factor $\approx C_\ell$ not $C_\ell\sqrt{n}$; product tighter by $\sim n^{-L/2}$) | Prevents reader from taking $(\sqrt{n})^L \approx 1.5\times 10^9$ as a practical signal-attenuation estimate. |
| 50 | Pseudocode §5.1: curvature-gate block replaced — flat default now applies residual directly ($x \leftarrow x + B_{\mathrm{res}}x + b_{\mathrm{res}}$); curved-manifold path provided as commented-out option with hard-threshold gate; $\sigma(0)=\tfrac{1}{2}$ arithmetic acknowledged in comment | Resolves the $\sigma(0)\neq 1$ discrepancy between prose ("gate is identically 1") and the sigmoid definition; implements the document's stated intent (inactive / pass-through in flat case) via option (c). |

---

*End of catalogue.*
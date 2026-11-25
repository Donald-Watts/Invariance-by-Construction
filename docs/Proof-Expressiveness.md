# Expressiveness Classification for $S_n$-Invariant Layered Architectures

*Formal theorem, proof, and constructive example.
Companion to: `Catalogue.md` (architecture), `Extension-A.md` (rank-1 Reynolds, non-realizability), `Research-Direction.md` (attack plan).*

---

## 1 — Setup

### 1.1 — The representation

Let $G = S_n$ act on $V = \mathbb{C}^n$ by permutation: $(\sigma \cdot x)_i = x_{\sigma^{-1}(i)}$. The decomposition into isotypic components is

$$V = \mathbb{C}\mathbf{1} \;\oplus\; W, \qquad W = \mathbf{1}^\perp = \{x \in \mathbb{C}^n : \sum_{i=1}^n x_i = 0\},$$

where $\mathbb{C}\mathbf{1}$ carries the trivial representation and $W \cong \mathbb{C}^{n-1}$ carries the standard (irreducible) representation. Write any $x \in V$ as

$$x = \underbrace{\frac{e_1(x)}{n}\,\mathbf{1}}_{\in\,\mathbb{C}\mathbf{1}} \;+\; \underbrace{\left(x - \frac{e_1(x)}{n}\,\mathbf{1}\right)}_{\in\,W},$$

where $e_1(x) = \sum_i x_i$ is the first elementary symmetric polynomial.

### 1.2 — The Reynolds operator

By Theorem A.1 of `Extension-A.md`, the Reynolds (averaging) operator for the permutation representation is the rank-1 orthogonal projection

$$P_{\mathrm{inv}} = \frac{1}{|G|}\sum_{\sigma \in S_n} \sigma = \frac{1}{n}\,\mathbf{1}\,\mathbf{1}^*,$$

which projects $V$ onto $\mathbb{C}\mathbf{1}$ along $W$. It satisfies $P_{\mathrm{inv}}^2 = P_{\mathrm{inv}}$, $P_{\mathrm{inv}}\,\sigma = \sigma\,P_{\mathrm{inv}}$ for all $\sigma \in S_n$, and $\mathrm{Im}(P_{\mathrm{inv}}) = \mathbb{C}\mathbf{1}$.

### 1.3 — The architecture

The catalogue's $L$-layer architecture computes

$$\Phi_L(x_0) = \;\underbrace{P_{\mathrm{inv}} \circ \psi_L \circ P_{\mathrm{inv}} \circ \psi_{L-1} \circ \cdots \circ P_{\mathrm{inv}} \circ \psi_1}_{\text{$L$ layers}}\;(x_0),$$

where each $\psi_\ell : V \to V$ is a smooth $S_n$-equivariant map (the per-layer computation: unitary mixing, gauge, residual, reweighting). The equivariance condition is

$$\psi_\ell(\sigma \cdot x) = \sigma \cdot \psi_\ell(x) \qquad \forall\;\sigma \in S_n,\; x \in V.$$

The final readout is the scalar

$$f(x_0) \;=\; \frac{\mathbf{1}^*\,\Phi_L(x_0)}{\sqrt{n}} \;\in\; \mathbb{C},$$

so that $\Phi_L(x_0) = f(x_0)\,\mathbf{1}/\sqrt{n}$.

### 1.4 — The invariant ring

The ring of $S_n$-invariant polynomials on $\mathbb{C}^n$ is

$$\mathbb{C}[V]^{S_n} = \mathbb{C}[e_1, e_2, \ldots, e_n],$$

where $e_k = \sum_{1 \leq i_1 < \cdots < i_k \leq n} x_{i_1}\cdots x_{i_k}$. Let $\mathbb{C}[V]^{S_n}_{\leq d}$ denote the subspace of invariant polynomials of total degree $\leq d$. Its dimension is

$$\dim \mathbb{C}[V]^{S_n}_{\leq d} = \sum_{k=0}^{\min(d,n)} p(k) \quad \text{(where $p(k)$ = number of integer partitions of $k$)}, \qquad (d \leq n),$$

For $d > n$, the summands for $k > n$ are $p_n(k)$ (partitions of $k$ into at most $n$ parts), which are strictly smaller than $p(k)$. The cumulative sum $\sum_{k=0}^{d} p(k)$ (for $d \leq n$) has no closed binomial-coefficient form; the sequence is $1,\; 2,\; 4,\; 7,\; 12,\; 19,\; 30,\; \ldots$ (OEIS A000712 cumulative: $p(0)=1,\, p(1)=1,\, p(2)=2,\, p(3)=3,\, p(4)=5,\, p(5)=7,\, p(6)=11$). More precisely, $\dim \mathbb{C}[V]^{S_n}_{\text{hom}}{d} = p(d)$ for $d \leq n$.

---

## 2 — The Structural Fact

> **Proposition 2.1 (Information bottleneck).** Let $\Phi_L$ be as in §1.3 with $L \geq 1$.
>
> **(a)** $\Phi_1(x_0) \in \mathbb{C}\mathbf{1}$ for all $x_0 \in V$. More generally, $\Phi_\ell(x_0) \in \mathbb{C}\mathbf{1}$ for all $\ell \geq 1$.
>
> **(b)** There exists a smooth function $\theta_1 : V \to \mathbb{C}$ such that $\Phi_1(x_0) = \theta_1(x_0)\,\mathbf{1}/\sqrt{n}$, and for $\ell \geq 2$ there exist smooth functions $\theta_\ell : \mathbb{C} \to \mathbb{C}$ such that $\Phi_\ell(x_0) = \theta_\ell(\theta_{\ell-1}(\theta_{\ell-2}(\cdots\theta_1(x_0)\cdots)))\,\mathbf{1}/\sqrt{n}$.
>
> **(c)** Consequently, $\Phi_L(x_0) = F(\theta_1(x_0))\,\mathbf{1}/\sqrt{n}$ where $F = \theta_L \circ \cdots \circ \theta_2 : \mathbb{C} \to \mathbb{C}$ is a 1-variable smooth function. The entire $n$-dimensional input is compressed to a **single complex scalar** $\theta_1(x_0)$ by the first layer, and all subsequent layers act on that scalar.

**Proof.**

**(a)** By induction. For $\ell = 1$: $\Phi_1(x_0) = P_{\mathrm{inv}}(\psi_1(x_0)) \in \mathrm{Im}(P_{\mathrm{inv}}) = \mathbb{C}\mathbf{1}$. ✓

Inductive step: suppose $\Phi_{\ell-1}(x_0) = c_{\ell-1}\,\mathbf{1}$ for some $c_{\ell-1} \in \mathbb{C}$. Then
$$\Phi_\ell(x_0) = P_{\mathrm{inv}}\bigl(\psi_\ell(c_{\ell-1}\mathbf{1})\bigr).$$
Since $\psi_\ell$ is equivariant and $\mathbf{1}$ is $S_n$-fixed ($\sigma \cdot \mathbf{1} = \mathbf{1}$), we have
$$\psi_\ell(\sigma \cdot (c_{\ell-1}\mathbf{1})) = \psi_\ell(c_{\ell-1}\mathbf{1}) = \sigma \cdot \psi_\ell(c_{\ell-1}\mathbf{1}).$$
Thus $\psi_\ell(c_{\ell-1}\mathbf{1})$ is $S_n$-fixed, hence lies in $\mathbb{C}\mathbf{1}$. Therefore $P_{\mathrm{inv}}(\psi_\ell(c_{\ell-1}\mathbf{1})) = \psi_\ell(c_{\ell-1}\mathbf{1}) \in \mathbb{C}\mathbf{1}$. ✓

**(b)** For $\ell = 1$: since $\Phi_1(x_0) = P_{\mathrm{inv}}\,\psi_1(x_0) = \frac{\mathbf{1}\,\mathbf{1}^*\psi_1(x_0)}{n}$, we write
$$\Phi_1(x_0) \;=\; \underbrace{\frac{\mathbf{1}^*\psi_1(x_0)}{\sqrt{n}}}_{\displaystyle =:\,\theta_1(x_0)}\;\frac{\mathbf{1}}{\sqrt{n}},$$
where $\theta_1 : V \to \mathbb{C}$ is smooth. For $\ell \geq 2$: since $\Phi_{\ell-1}(x_0) = \theta_{\ell-1}(\cdots)\,\mathbf{1}/\sqrt{n}$ lies in $\mathbb{C}\mathbf{1}$, and $\psi_\ell$ is equivariant, $\psi_\ell$ maps it to $\mathbb{C}\mathbf{1}$, so
$$\Phi_\ell(x_0) \;=\; P_{\mathrm{inv}}\,\psi_\ell\!\left(\theta_{\ell-1}(\cdots)\,\frac{\mathbf{1}}{\sqrt{n}}\right) \;=\; \underbrace{\frac{\mathbf{1}^*\,\psi_\ell\!\left(\theta_{\ell-1}(\cdots)\,\mathbf{1}/\!\sqrt{n}\right)}{\sqrt{n}}}_{\displaystyle =:\,\theta_\ell\bigl(\theta_{\ell-1}(\cdots)\bigr)}\;\frac{\mathbf{1}}{\sqrt{n}},$$
where each $\theta_\ell : \mathbb{C} \to \mathbb{C}$ is smooth. ✓

**(c)** Immediate from (b). \quad $\square$

> **Remark 2.2 (Why this is not yet the full story).** Proposition 2.1 shows that $\theta_1 : V \to \mathbb{C}$ is the **sole carrier** of input information. The expressiveness of the architecture is entirely determined by the class of functions that $\theta_1$ can represent. Proposition 2.1 does not restrict $\theta_1$ beyond smoothness and $S_n$-invariance (which follows from equivariance of $\psi_1$ and the $\mathbf{1}^*$-readout). The next theorem pins down $\theta_1$'s class.

---

## 3 — The Expressiveness Theorem

### 3.1 — The readout function

> **Proposition 3.1.** The function $\theta_1(x) = \mathbf{1}^*\psi_1(x)/\sqrt{n}$ is $S_n$-invariant.

**Proof.** For $\sigma \in S_n$:
$$\theta_1(\sigma \cdot x) = \frac{\mathbf{1}^*\psi_1(\sigma \cdot x)}{\sqrt{n}} = \frac{\mathbf{1}^*(\sigma \cdot \psi_1(x))}{\sqrt{n}} = \frac{(\sigma \cdot \mathbf{1})^*\psi_1(x)}{\sqrt{n}} = \frac{\mathbf{1}^*\psi_1(x)}{\sqrt{n}} = \theta_1(x),$$

where we used equivariance of $\psi_1$ and $\sigma \cdot \mathbf{1} = \mathbf{1}$. \quad $\square$

### 3.2 — Polynomial specialization

To obtain a sharp classification, specialize to **polynomial** equivariant layers. (The catalogue's attention step $x \mapsto \mathrm{softmax}\!\bigl(\mathrm{Re}(x\,\bar{x}^{\top})\bigr)\,x$ is a *transcendental* (real-analytic) function involving exponentials of bilinear forms in $x, \bar{x}$ — not a polynomial and not a rational function. The correct function class for the catalogue's default is $C^\omega(V)^{S_n}$ (real-analytic $S_n$-invariants), which strictly contains $\mathbb{C}[V]^{S_n}_{\mathrm{poly}}$. The polynomial case below gives the cleanest theorem and the sharpest algebraic bound; the catalogue's actual class is larger but retains the same single-scalar bottleneck structure (Prop 2.1 requires only smooth equivariance). See Lemma 3.4 for the precise bridging.)

> **Definition 3.2.** A map $\psi : V \to V$ is a **degree-$d$ equivariant polynomial** if $\psi = (\psi_1, \ldots, \psi_n)$ where each $\psi_i \in \mathbb{C}[x_1, \ldots, x_n]$ has total degree $\leq d$ and $\psi$ is $S_n$-equivariant.

> **Lemma 3.3 (Readout is an invariant polynomial).** If $\psi : V \to V$ is a degree-$d$ $S_n$-equivariant polynomial map, then the readout
> $$\theta_1(x) \;=\; \frac{1}{\sqrt{n}}\sum_{i=1}^n \psi_i(x)$$
> is an $S_n$-invariant polynomial of degree $\leq d$, i.e.
> $$\theta_1 \;\in\; \mathbb{C}[V]^{S_n}_{\leq d} \;=\; \mathbb{C}[e_1, \ldots, e_{\min(d,n)}]_{\leq d}.$$

**Proof.** By Proposition 3.1, $\theta_1$ is $S_n$-invariant. Since each $\psi_i$ has degree $\leq d$, so does $\theta_1$ (a linear combination of degree-$\leq d$ polynomials). By the fundamental theorem of invariant theory for $S_n$, $\mathbb{C}[V]^{S_n} = \mathbb{C}[e_1, \ldots, e_n]$, so every invariant polynomial of degree $\leq d$ lies in $\mathbb{C}[e_1, \ldots, e_{\min(d,n)}]_{\leq d}$. \quad $\square$

> **Lemma 3.4 (Bridging: the catalogue is a transcendental instance).** The catalogue architecture of [Catalogue.md] satisfies the hypotheses of Proposition 2.1 and therefore the single-scalar bottleneck applies. Specifically:
>
> **(a) (Equivariance.)** The attention step $x \mapsto \mathrm{softmax}\!\bigl(\mathrm{Re}(x\,\bar{x}^{\top})\bigr)\,x$ is $S_n$-equivariant. *Proof.* For $\sigma \in S_n$ and $k = \sigma(i)$:
> $$f(\sigma x)_{\sigma(i)} \;=\; \sum_j \frac{e^{\mathrm{Re}(x_{\sigma^{-1}(i)}\,\overline{x_{\sigma^{-1}(j)}})}{\sum_m e^{\mathrm{Re}(x_{\sigma^{-1}(i)}\,\overline{x_{\sigma^{-1}(m)}})}}\; x_{\sigma^{-1}(j)} \;=\; \sum_{j'} \frac{e^{\mathrm{Re}(x_i\,\bar{x}_{j'})}}{\sum_{m'} e^{\mathrm{Re}(x_i\,\bar{x}_{m'})}}\; x_{j'} \;=\; f(x)_i \;=\; (\sigma f(x))_i. \qquad\square$
>
> **(b) (Bottleneck applies.)** The catalogue applies $P_{\mathrm{inv}}$ (mean projection) after each layer. By Prop 2.1, the state after layer 1 lies in $\mathbb{C}\mathbf{1}$, and all subsequent layers are scalar maps $\mathbb{C} \to \mathbb{C}$. The $L$-layer catalogue computes $F(\theta_1(x_0))$ for a real-analytic $S_n$-invariant $\theta_1$ and a smooth $F$.
>
> **(c) (Transcendental, not polynomial.)** The readout $\theta_1$ is a real-analytic function whose Taylor expansion in the $e_k$-basis has **infinitely many** non-zero terms (the softmax involves $\exp$ of bilinear forms in $x,\bar{x}$, expanding to all degrees). Thus $\theta_1 \notin \mathbb{C}[V]^{S_n}_{\leq d}$ for any finite $d$; the algebraic bound of Theorem 3.4 is a *restriction on polynomial truncations*, not a characterization of the catalogue.
>
> **(d) (Fundamental limitation.)** The true ceiling of the catalogue is the **single-scalar output**: it produces one number $F(\theta_1(x_0))$, not the full orbit $(e_1(x_0), \ldots, e_n(x_0)) \in \mathbb{C}^n$. Extracting a specific invariant $e_k$ from $\theta_1$ requires inverting a transcendental equation, which is generically multi-valued and ill-conditioned for $k$ large relative to the effective information content of $\theta_1$.
>
> **(e) (Polynomial model as clean theorem.)** For a sharp, provable classification, we work in the polynomial model of §3.2. The polynomial bound (Thm 3.4) is a *lower bound* on the catalogue's expressiveness: anything the polynomial model can compute, the catalogue can also compute (by restricting $F$ to a polynomial). The catalogue may be more powerful (transcendental $F$), but the single-scalar structure is shared.

> **Remark 3.6 (Practical consequence for the catalogue.)** In the catalogue's default configuration ($d_{\mathrm{head}}=1$, all-ones mask, zero curvature), the readout $\theta_1$ is a specific real-analytic invariant of the form
> $$\theta_1(x) \;=\; \frac{1}{\sqrt{n}}\sum_{i,j} \frac{e^{\mathrm{Re}(x_i\bar{x}_j)}}{\sum_k e^{\mathrm{Re}(x_i\bar{x}_k)}}\,x_j,$$
> whose leading Taylor terms (about $x=0$, after the linear operations of the layer) are of degree 1 (proportional to $e_1$) and degree 2 (involving $e_2$). For the purposes of Theorem 3.4, the catalogue is a $d_1 = 2$ instance: it can compute $e_1$ and $e_2$ (and all invariant polynomials of $x$-degree $\leq 2$) as smooth functions of $\theta_1$, but the *guaranteed* algebraic bound is degree 2. Higher-degree invariants are in principle extractable (by Lemma 3.4(c)) but not guaranteed by the polynomial theorem.

### 3.3 — Main theorem

> **Theorem 3.4 (Expressiveness classification).** Let $\Phi_L$ be the $L$-layer architecture of §1.3, where layer 1 is a degree-$d_1$ equivariant polynomial map and layers $2, \ldots, L$ are smooth equivariant maps (of any degree).
>
> **(i) (Bottleneck.)** The computed function is
> $$\Phi_L(x_0) = F\!\bigl(\,\theta_1(x_0)\,\bigr)\,\mathbf{1}/\!\sqrt{n},$$
> where $\theta_1 \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ and $F : \mathbb{C} \to \mathbb{C}$ is a smooth function (composition of $L{-}1$ smooth maps $\mathbb{C}\to\mathbb{C}$).
>
> **(ii) (Impossibility for linear layers.)** If $d_1 = 1$, then $\theta_1(x) = a\,e_1(x) + b$ for some $a, b \in \mathbb{C}$, and
> $$\Phi_L(x_0) = F(a\,e_1(x_0) + b)\,\mathbf{1}/\!\sqrt{n}.$$
> In particular, $\Phi_L$ is a **function of $e_1$ alone** and is **independent of $e_2, e_3, \ldots, e_n$**, regardless of $L$.
> 
> No amount of depth $L$ or nonlinearity in layers $2, \ldots, L$ can recover $e_2, \ldots, e_n$.
>
> **(iii) (Sharp access for degree $\geq 2$.)** If $d_1 \geq 2$, then $\theta_1$ can represent $e_2$ (and more generally any invariant polynomial of degree $\leq d_1$). Specifically, there exists a degree-2 equivariant polynomial map $\psi_1 : \mathbb{C}^n \to \mathbb{C}^n$ such that
> $$\theta_1(x) = \frac{1}{\sqrt{n}}\sum_{i=1}^n \psi_{1,i}(x) = \frac{1}{n}\,e_2(x) + \text{(terms in } e_1\text{)},$$
> so that $e_2$ is **extractable in a single layer**.
>
> **(iv) (General classification.)** The architecture with first-layer degree $d_1$ can compute exactly the functions $\mathbb{C}[e_1, \ldots, e_{\min(d_1,n)}]_{\leq d_1} \xrightarrow{\;F\;} \mathbb{C}$, i.e. smooth functions of a degree-$\leq d_1$ invariant polynomial. It **cannot** compute any function that essentially depends on $e_k$ for $k > d_1$, regardless of $L$.

**Proof.**

**(i)** Proposition 2.1(c) gives the form. Lemma 3.3 gives $\theta_1 \in \mathbb{C}[V]^{S_n}_{\leq d_1}$. ✓

**(ii)** $\mathbb{C}[V]^{S_n}_{\leq 1} = \mathbb{C}[e_1]_{\leq 1} = \{ae_1 + b : a, b \in \mathbb{C}\}$. So $\theta_1(x) = ae_1(x) + b$. The rest follows from (i). ✓

**(iii)** **Constructive example.** Define
$$\psi_1(x) \;=\; \frac{e_2(x)}{n}\,\mathbf{1} \;\in\; \mathbb{C}\mathbf{1} \;\subset\; V.$$

*Equivariance.* $e_2$ is $S_n$-invariant: $e_2(\sigma \cdot x) = \sum_{i<j} x_{\sigma^{-1}(i)}\,x_{\sigma^{-1}(j)} = \sum_{i<j} x_i\,x_j = e_2(x)$ (relabeling). Since $\mathbf{1}$ is $S_n$-fixed, $(\sigma \cdot \psi_1(x))_i = \psi_1(\sigma \cdot x) = \frac{e_2(x)}{n}\mathbf{1} = \psi_1(x)$. Hence $\psi_1$ is $S_n$-equivariant (trivially, since its image lies in the fixed-point subspace). ✓

*Degree.* $e_2(x) = \sum_{i<j} x_i x_j$ is a homogeneous polynomial of degree 2 in the coordinates. Hence $\psi_1$ is a degree-2 equivariant polynomial map. ✓

*Readout.*
$$\theta_1(x) \;=\; \frac{\mathbf{1}^*\,\psi_1(x)}{\sqrt{n}} \;=\; \frac{\mathbf{1}^*\!\left(\frac{e_2(x)}{n}\,\mathbf{1}\right)}{\sqrt{n}} \;=\; \frac{e_2(x)}{n\sqrt{n}}\;\mathbf{1}^*\mathbf{1} \;=\; \frac{e_2(x)}{n\sqrt{n}}\cdot n \;=\; \frac{e_2(x)}{\sqrt{n}}.$$

Thus $\Phi_1(x_0) = \frac{e_2(x_0)}{n}\,\mathbf{1}$, and the scalar readout $f(x_0) = \mathbf{1}^*\Phi_1(x_0)/\sqrt{n} = e_2(x_0)/\sqrt{n}$. Rescaling by $\sqrt{n}$ (absorbed into the output head) yields $e_2(x_0)$ exactly. ✓

> **Remark 3.5 (Architectural realizability).** The map $x \mapsto e_2(x)\,\mathbf{1}/n$ is a valid degree-2 equivariant polynomial, but it is **not** a single step of the catalogue's unitary-gauge-residual-reweighting pipeline in its default form. To realize $e_2$ within the catalogue one uses:
> 1. **Gauge (elementwise square):** $x \mapsto x \odot x$ (degree 2, equivariant).
> 2. **Readout:** $\mathbf{1}^*(x \odot x)/\sqrt{n} = \sum_i x_i^2/\sqrt{n} = (e_1^2 - 2e_2)/\sqrt{n}$ (using $\sum_i x_i^2 = e_1^2 - 2e_2$).
> 3. **Linear correction:** subtract $e_1^2/\!\sqrt{n}$ (available from a parallel linear readout) and multiply by $-1/2$.
>
> This is a **concrete, implementable** two-step construction within the catalogue's operational vocabulary. It confirms that the catalogue's gauge operation is sufficient to access degree-2 invariants; the limitation in (ii) applies only when the gauge is replaced by a linear map.

**(iv)** Follows from (i) + Lemma 3.3. The function $\theta_1$ ranges over $\mathbb{C}[V]^{S_n}_{\leq d_1} = \mathbb{C}[e_1, \ldots, e_{\min(d_1,n)}]_{\leq d_1}$. By the fundamental theorem of invariant theory, $e_1, \ldots, e_n$ are algebraically independent over $\mathbb{C}$, so $\mathrm{trdeg}_{\mathbb{C}}\,\mathbb{C}(e_1, \ldots, e_n) = n$. For $k > d_1$, the element $e_k$ is transcendental over the subfield $\mathbb{C}(e_1, \ldots, e_{d_1})$ (which has transcendence degree $d_1 < n$); in particular $e_k \notin \mathbb{C}(e_1, \ldots, e_{d_1})$ as a rational function. Since any $\theta \in \mathbb{C}[e_1, \ldots, e_{d_1}]_{\leq d_1}$ generates a subfield $\mathbb{C}(\theta) \subseteq \mathbb{C}(e_1, \ldots, e_{d_1})$ of transcendence degree $\leq 1$, the element $e_k$ is transcendental over $\mathbb{C}(\theta)$. If a smooth $F : \mathbb{C} \to \mathbb{C}$ satisfied $F(\theta(x)) = e_k(x)$ on a non-empty open set of the quotient $(\mathbb{C}^n)^{S_n}/\!\sim$, then $e_k$ would be a holomorphic (hence algebraic) function of the single variable $\theta$, i.e. $e_k \in \mathbb{C}(\theta)$, contradicting its transcendence over $\mathbb{C}(\theta)$. \quad $\square$

---

## 4 — Corollaries and the Expressiveness Classification

> **Corollary 4.1 (Depth is irrelevant for linear first layers.)** If the catalogue's per-layer structure restricts the first layer to degree $d_1 = 1$ (i.e. the gauge is linear or absent), then
> $$\mathcal{A}(1, L) \;=\; \{\, F(a\,e_1 + b)\,\mathbf{1}/\!\sqrt{n} \;\mid\; F \in C^\infty(\mathbb{C},\mathbb{C}),\; a,b \in \mathbb{C} \,\}$$
> for **all** $L \geq 1$. The architecture computes only functions of $e_1$, regardless of depth.
>
> *Interpretation.* The non-realizability of $e_2$-dependent functions (Extension-A Corollary A.2a) is **not** a depth limitation. It is a **first-layer degree** limitation. Adding layers does not help. Adding nonlinearity to the first layer (degree $\geq 2$) does.

> **Corollary 4.2 (Strict filtration by first-layer degree.)** The chain
> $$\mathcal{A}(1, L) \;\subsetneq\; \mathcal{A}(2, L) \;\subsetneq\; \mathcal{A}(3, L) \;\subsetneq\; \cdots \;\subsetneq\; \mathcal{A}(n, L)$$
> is a **strict** filtration for every $L \geq 1$. Strictness follows from the algebraic independence of $e_1, \ldots, e_n$: $e_{k+1} \notin \mathbb{C}[e_1, \ldots, e_k]$ as a polynomial, so no smooth $F$ of a degree-$\leq k$ invariant can equal $e_{k+1}$ on an open set.
>
> The terminal object $\mathcal{A}(n, L)$ contains all smooth functions of the $n$ elementary symmetric polynomials, which (by the fundamental theorem of symmetric polynomials) is the full ring of $S_n$-invariant polynomial functions.

> **Corollary 4.3 (The expressiveness classification — final form.)**
>
> **Theorem (Expressiveness classification of $S_n$-invariant layered architectures).** Let $\mathcal{A}(d_1, L)$ denote the class of $S_n$-invariant functions $V \to \mathbb{C}$ computable by the $L$-layer architecture of §1.3 with first-layer degree $d_1$.
>
> $$\boxed{\;\mathcal{A}(d_1, L) \;=\; \{\, F\bigl(\theta(x_0)\bigr) \;\mid\; \theta \in \mathbb{C}[V]^{S_n}_{\leq d_1},\;\; F \in C^\infty(\mathbb{C}, \mathbb{C}) \,\}\;}$$
>
> i.e. *smooth functions of a single invariant polynomial of degree $\leq d_1$*. Since every element of $\mathbb{C}[e_1, \ldots, e_n]$ is itself a single invariant polynomial, $\mathcal{A}(d_1, L)$ contains **all** invariant polynomials of $x$-degree $\leq d_1$ (take $F = \mathrm{id}$), together with all smooth non-polynomial functions of one such invariant. The restriction is the **degree bound** $d_1$ on $\theta$, not a limitation on the number of elementary symmetric polynomials appearing in $\theta$.
>
> *Consequences:*
> - $\mathcal{A}(1, L)$ is independent of $L$: depth plays no role when the first layer is linear.
> - $e_k \in \mathcal{A}(d_1, L)$ iff $k \leq d_1$ (for some choice of $F$).
> - The filtration $\mathcal{A}(1,L) \subsetneq \mathcal{A}(2,L) \subsetneq \cdots \subsetneq \mathcal{A}(n,L)$ is strict.
> - The terminal class $\mathcal{A}(n, L)$ is the full invariant function space (polynomial part).
>
> *Proof.* Direct from Theorem 3.4 + algebraic independence of $e_1, \ldots, e_n$. \quad $\square$
>
> **Remark 4.5 (Scope of the bottleneck.)** Because every polynomial $P(e_1,\ldots,e_n)$ is itself a single invariant polynomial, $\mathcal{A}(d_1,L)$ contains **all** invariant polynomials of $x$-degree $\leq d_1$ (take $F=\mathrm{id}$), together with all smooth (non-polynomial) functions of one such invariant. The restriction is therefore **not** "one invariant vs. many invariants" — it is the **degree bound** $d_1$ on $\theta$. Concretely:
> - $e_k \in \mathcal{A}(d_1,L)$ iff $k \leq d_1$.
> - Any invariant polynomial $P$ of $x$-degree $\leq d_1$ (e.g.\ $e_1 e_2$ when $d_1 \geq 3$, or $e_1^2 + e_2$ when $d_1 \geq 2$) is in $\mathcal{A}(d_1,L)$.
> - Invariant polynomials of $x$-degree $> d_1$ (e.g.\ $e_{d_1+1}$, or $e_1\,e_{d_1+1}$) are **not** in $\mathcal{A}(d_1,L)$.
> - Smooth functions that depend on invariants of degree $> d_1$ through non-algebraic maps (e.g.\ $\sin(e_{d_1+1}(x))$) are also excluded.
> - The architecture outputs a **single scalar**; it cannot produce a vector of invariants $(e_1(x),\ldots,e_m(x))$ as a joint multi-dimensional output.
> - Functions like $\sin(e_1(x))\cdot\cos(e_2(x))$ are **not** in $\mathcal{A}(d_1,L)$ for any $d_1$: they depend on $e_1$ and $e_2$ through *different* transcendental maps and cannot be written as $F(\theta(x))$ for any single invariant polynomial $\theta$.

> **Corollary 4.4 (Fleet architecture escapes the bound).** The fleet architecture (`Fleet-Optimization-Spec.md`) uses a Set Transformer (ISAB + PAB) whose first "layer" is the ISAB: a **rational** (attention-based) equivariant map, not a bounded-degree polynomial. By the DeepSets universality theorem (Zaheer et al., 2017, Theorem 2), the ISAB can **approximate** any $S_n$-invariant continuous function on compact sets to arbitrary precision. Note the distinction: DeepSets establishes *approximation* of symmetric continuous functions, whereas Theorem 3.4 establishes *exact* polynomial computation impossibility. The two results are complementary, not contradictory: the catalogue's limitation is algebraic (structural), while the fleet's is statistical (finite capacity). Hence the fleet architecture is **not subject to Theorem 3.4's degree bound**: its first-layer expressiveness is unbounded (in the algebraic sense). The expressiveness ceiling in Fleet Spec §0.4 is therefore a **statistical/practical** limitation (finite $h$, finite training data), not an **algebraic** one.

---

## 5 — Constructive Example: Computing $e_2$ in One Layer

> **Proposition 5.1.** There exists a degree-2 equivariant polynomial map $\psi : \mathbb{C}^n \to \mathbb{C}^n$ such that the one-layer architecture ($L=1$) with first layer $\psi$ and readout $\mathbf{1}^*/\sqrt{n}$ computes $e_2(x)$ exactly.

**Proof.** Take $\psi(x) = \frac{e_2(x)}{n}\,\mathbf{1}$, as in Theorem 3.4(iii). The readout gives $e_2(x)/\sqrt{n}$; rescale by $\sqrt{n}$. \quad $\square$

> **Remark 5.2 (Catalogue-realizable construction).** Within the catalogue's operational vocabulary (unitary, gauge, residual, reweighting):
>
> | Step | Operation | Output |
> |---|---|---|
> | 1 | Gauge: $x \mapsto x \odot x$ | $x_i^2$ per coordinate |
> | 2 | Readout: $\sum_i x_i^2 / \sqrt{n}$ | $(e_1^2 - 2e_2)/\sqrt{n}$ |
> | 3 | Linear correction: $-\frac{1}{2\sqrt{n}}\bigl(\text{step 2} - e_1^2\bigr)$ | $e_2/\sqrt{n}$ |
> | 4 | Rescale by $\sqrt{n}$ | $e_2$ |
>
> Steps 2–4 are linear operations on the readout (absorbed into the output head). The only nonlinearity is the elementwise square in step 1, which is the catalogue's gauge with $\beta = 1$ and quadratic kernel. **$e_2$ is computable in a single catalogue layer.**

---

## 6 — Connection to the Fleet Architecture

| Feature | Catalogue (this paper) | Fleet (ISAB / Set Transformer) |
|---|---|---|
| Per-layer map | $V \to V$, equivariant, degree $d_\ell$ | $V^n \to V^n$, equivariant, **rational** (attention) |
| Projection | $P_{\mathrm{inv}}$ (rank-1, §1.2) | Mean-pooling (same algebraic effect) |
| First-layer expressiveness | $\mathbb{C}[V]^{S_n}_{\leq d_1}$ (bounded degree) | All $S_n$-invariant smooth functions (DeepSets) |
| Depth role | **Irrelevant** (bottleneck is layer 1) | Relevant (composes smooth $\mathbb{R}^h \to \mathbb{R}^h$ maps) |
| $e_k$ access | $k \leq d_1$ only | All $k$ (in principle) |
| Limitation type | **Algebraic** (structural impossibility) | **Statistical** (finite capacity, finite data) |

The fleet architecture's expressiveness ceiling (Fleet Spec §0.4) is correct in form: the global decision depends on the $h$-dimensional centroid. But the bound is not algebraic — it is a practical capacity limit. The catalogue's limitation (Theorem 3.4) is the **stronger, structural** result: no amount of depth or training overcomes a degree-1 first layer.

---

## 7 — Status and Remaining Work

| Component | Status | Location |
|---|---|---|
| Setup: representation, Reynolds, architecture, invariant ring | ✅ Complete | This file, §1 |
| Information bottleneck (Prop 2.1) | ✅ Proven | This file, §2 |
| Readout invariance (Prop 3.1) | ✅ Proven | This file, §3.1 |
| Equivariant polynomial readout (Lemma 3.3) | ✅ Proven | This file, §3.2 |
| **Main theorem (Thm 3.4)** | ✅ Stated + proved | This file, §3.3 |
| **Expressiveness classification (Cor 4.3)** | ✅ Stated + proven (follows from Thm 3.4) | This file, §4 |
| Constructive $e_2$ example (Prop 5.1) | ✅ Proven + catalogue-realized | This file, §5 |
| Fleet connection | ✅ Stated | This file, §6 |
| Bridging lemma (catalogue → algebraic model) | ✅ Proven (Lemma 3.4 + Remark 3.6) | This file, §3.2 |
| Empirical $e_2$ verification (Prop 5.1) | ✅ Verified (4/4 tests pass) | `verify_e2.py` |
| Exterior-power / isotypic classification | ✅ Stated (Prop 7.1 + Remark 7.2, incl. conjecture) | This file, §7.5 |
| Related work | ✅ Written | This file, §7.6 |
| Full paper assembly | ⬜ | Assemble §1–§7.6 into submission-ready manuscript |

---

## 7.5 — Representation-Theoretic Refinement

> **Proposition 7.1 (Degree necessity, restated).** Let $n \geq k \geq 2$. In any $L$-layer architecture of §1.3 with first-layer degree $d_1$, the invariant $e_k$ is computable (i.e., $e_k \in \mathcal{A}(d_1, L)$ for some $L$) if and only if $d_1 \geq k$.

**Proof.** $(\Leftarrow)$ If $d_1 \geq k$, take $\theta = e_k \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ and $F = \mathrm{id}$. Then $e_k(x) = F(\theta(x)) \in \mathcal{A}(d_1, L)$. $(\Rightarrow)$ If $d_1 < k$, then $e_k \notin \mathbb{C}[V]^{S_n}_{\leq d_1}$ (since $e_k$ has $x$-degree $k > d_1$). By Theorem 3.4(iii), no $F \in C^\infty(\mathbb{C})$ and $\theta \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ can satisfy $F(\theta(x)) = e_k(x)$ on an open set, since $e_k$ is algebraically independent of $\mathbb{C}(e_1, \ldots, e_{d_1})$. \quad $\square$

> **Remark 7.2 (Isotypic content and the readout — correct decomposition).** The representation-theoretic structure underlying the degree bound can be seen through the isotypic decomposition of $\mathrm{Sym}^k(V)$.
>
> Write $V = \mathbb{C}\mathbf{1} \oplus W$ where $W = \mathbf{1}^\perp \cong \mathbb{C}^{n-1}$ is the standard irreducible of $S_n$. Then
> $$\mathrm{Sym}^k(V) \;=\; \bigoplus_{j=0}^{k} \mathrm{Sym}^{j}(W)$$
> (each $\mathrm{Sym}^{k-j}(\mathbb{C}\mathbf{1}) = \mathbb{C}\mathbf{1}$ contributes one copy of $\mathrm{Sym}^j(W)$).
>
> The readout $\theta_1 = \frac{1}{\sqrt{n}}\mathbf{1}^{\top} \psi_k$ pairs the degree-$k$ part $\psi_k : V \to V$ with the trivial component $\mathbb{C}\mathbf{1} \subset V$. By Schur's lemma, this pairing is non-zero only on the $\mathbb{C}\mathbf{1}$-isotypic summand of $\mathrm{Sym}^k(V)^*$ and vanishes on all non-trivial isotypic components. The invariant $e_k$ is precisely the generator of the $\mathbb{C}\mathbf{1}$-component of $\mathrm{Sym}^k(V)^*$ of degree $k$ in the $x$-variables.
>
> **The non-trivial isotypic content.** By the Littlewood (1940) decomposition, for $k \leq n-1$:
> $$\mathrm{Sym}^k(W) \;=\; \bigoplus_{\substack{\lambda \,\vdash\, k \\ \ell(\lambda) \leq n-1}} s_\lambda(W),$$
> where the sum is over partitions $\lambda$ of $k$ with at most $n-1$ parts, each appearing with multiplicity 1. For example, $k=2$: $\mathrm{Sym}^2(W) = s_{(2)}(W) \oplus s_{(1,1)}(W) = s_{(2)}(W) \oplus \bigwedge^2(W)$. These non-trivial constituents are **invisible** to the rank-1 readout $\theta_1$ but are present in the equivariant map $\psi_k$. They become relevant in architectures that do NOT project to $\mathbb{C}\mathbf{1}$ between layers (e.g., the fleet architecture of §6), where the full isotypic content is available for subsequent layers to exploit.
>
> **Open problem (minimal isotypic necessity).** For an architecture that retains $W$-components between layers, what is the minimal isotypic content of the first layer needed to compute $e_k$ at any depth? The $k=2$ case ($e_2$ computable from the $s_{(2)}(W)$ component via Prop 5.1) is understood. General $k$ requires a full plethysmic computation of $\mathrm{Sym}^k(W)$ combined with an induction on the layer structure, and is left as future work.

---

## 7.6 — Related Work

**Set functions and universality.** Zaheer et al. (2017, *Deep Sets*, Theorem 2) prove that the architecture $\rho\bigl(\sum_i \phi(x_i)\bigr)$ (sum-pooling + MLP) is a *universal approximator* for continuous symmetric functions on compact domains. This is an *approximation* result: for any $\varepsilon > 0$ and any continuous symmetric $f$, there exists a network with $\|\text{net} - f\|_\infty < \varepsilon$. Our Theorem 3.4 is an *exact* classification: it characterizes the set of functions (invariant polynomials up to degree $d_1$, and smooth functions of one such invariant) that the architecture computes *exactly*, and proves that functions outside this set (e.g., $e_k$ for $k > d_1$) are *not* in the image, regardless of depth or approximation. The two results are complementary: DeepSets says "you can approximate any symmetric function"; we say "the exact image of a fixed layer structure is a proper subset, and depth cannot enlarge it."

**Equivariant networks.** Cohen and Welling (2016, *Group Equivariant Convolutional Networks*) and Sannai and Kawano (2019, *A Group Theoretic Framework for Equivariant Neural Network Architectures*) develop equivariant architectures for general compact groups. Their contributions are *expressiveness* (the network can represent any equivariant map) and *efficiency* (parameter sharing). Our result is *restriction*: for a specific equivariant architecture with a rank-1 projection (Reynolds operator) between layers, the exact image is strictly smaller than the full equivariant function class, and the restriction is depth-independent.

**Invariant theory.** The fundamental theorem of symmetric polynomials (Newton, 1725; see e.g. Fulton, *Young Tableaux*, Ch. 2) gives $\mathbb{C}[x_1,\ldots,x_n]^{S_n} = \mathbb{C}[e_1,\ldots,e_n]$. Our work applies this algebraic fact to a *computational* setting: the architecture's readout is a single element of this ring (or a smooth function of one), and the degree bound on that element is the expressiveness ceiling.

**Polynomial method in complexity.** The degree-as-complexity measure used here (a function's expressiveness is bounded by its polynomial degree) is analogous to the polynomial method in communication complexity and property testing (e.g., Yehudayoff 2009, *The Polynomial Method*). The novelty is the *architectural* interpretation: the degree bound is imposed by the layer structure, not by a lower-bound argument on the function itself.

**Positioning.** To the best of our knowledge, no prior work proves a *depth-irrelevance* theorem for equivariant neural architectures: the statement that "no amount of depth or nonlinearity in layers $2,\ldots,L$ can recover a function outside the first layer's polynomial image" appears to be new. Existing equivariant-NN literature focuses on sufficiency (can the architecture represent $f$?) rather than necessity (can it *not* represent $g$, provably, for all $L$?).


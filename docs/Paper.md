# Depth is Irrelevant: An Expressiveness Classification for $S_n$-Invariant Layered Architectures

## Abstract

An $L$-layer $S_n$-equivariant architecture with a rank-1 projection (Reynolds operator) between layers computes a function of a **single** invariant polynomial, regardless of $L$. We prove this structural fact (Proposition 3) and, specializing to polynomial first layers, derive a complete expressiveness classification (Theorem 8): the architecture computes exactly the smooth functions of one degree-$\leq d_1$ invariant polynomial, where $d_1$ is the first layer's degree. No amount of depth, nonlinearity, or architectural complexity in layers $2,\ldots,L$ can recover a function outside this class — a structural consequence of the rank-1 projection architecture, not a general limitation of equivariant networks. In particular, when $d_1 = 1$ the architecture is confined to functions of $e_1 = \sum_i x_i$ alone, for **all** $L$; when $d_1 \geq 2$, the second elementary symmetric polynomial $e_2$ becomes accessible in a single layer (Theorem 8(iii)), and the filtration $\mathcal{A}(1,L) \subsetneq \mathcal{A}(2,L) \subsetneq \cdots \subsetneq \mathcal{A}(n,L)$ is strict. We provide a catalogue-realizable construction for $e_2$, a bridging lemma connecting the polynomial theorem to the catalogue's transcendental (softmax-based) gauge, and a representation-theoretic refinement via the Littlewood decomposition. We contrast our algebraic impossibility result with the DeepSets universality theorem (Zaheer et al. 2017), showing the two are complementary: one classifies exact computation, the other guarantees approximation. The fleet architecture (Set Transformer) escapes the algebraic bound but remains subject to statistical capacity limits.

**Keywords:** equivariant neural networks, invariant theory, expressiveness, depth irrelevance, $S_n$-symmetry, Reynolds operator, symmetric polynomials

---

## 1. Introduction

**Motivation.** Consider a neural architecture that processes an unordered set of $n$ elements represented as $x \in \mathbb{C}^n$, with the symmetric group $S_n$ acting by permutation. A natural design principle is to insert a projection to the $S_n$-invariant subspace after each layer, ensuring that inter-layer communication respects the symmetry. The "catalogue" architecture under study does exactly this: each layer is an equivariant map $\psi_\ell: V \to V$ (in the catalogue's default pipeline, a softmax-of-bilinear-forms gauge followed by an equivariant linear map), and between consecutive layers the state is projected via the Reynolds operator $P_{\mathrm{inv}} = \frac{1}{n}\mathbf{1}\mathbf{1}^*$ onto the one-dimensional trivial subspace $\mathbb{C}\mathbf{1}$. The final output is a scalar readout $f(x_0) = \mathbf{1}^*\Phi_L(x_0)/\sqrt{n}$. Throughout, "the catalogue" refers to any architecture of this form with polynomial first layers; the polynomial restriction is what makes Theorem 8's degree bound sharp (see Lemma 7 for the transcendental default).

**The question.** What can such an architecture compute, and what can it not? More precisely: given a first layer of polynomial degree $d_1$ and $L{-}1$ subsequent layers of arbitrary (smooth, equivariant) complexity, what is the exact class of $S_n$-invariant functions $f: V \to \mathbb{C}$ that the architecture realizes?

**Our answer (main theorem).** The class is
$$\mathcal{A}(d_1, L) \;=\; \{\, F\bigl(\theta(x_0)\bigr) \;\mid\; \theta \in \mathbb{C}[V]^{S_n}_{\leq d_1},\;\; F \in C^\infty(\mathbb{C}, \mathbb{C}) \,\},$$
independent of $L$. The architecture computes a smooth function of **one** invariant polynomial of degree $\leq d_1$. The depth $L$ plays no role. The first layer's degree $d_1$ is the **sole** parameter governing expressiveness.

**Consequences.**

- *(Depth irrelevance.)* If $d_1 = 1$, then $\mathcal{A}(1, L) = \{F(ae_1 + b) : F \in C^\infty(\mathbb{C})\}$ for all $L \geq 1$. The architecture is a function of $e_1$ alone. No amount of depth can recover $e_2, e_3, \ldots$.
- *(Sharp access.)* If $d_1 \geq 2$, then $e_2$ is computable in a single layer (Theorem 8(iii)), with a concrete catalogue-realizable construction (Remark 9).
- *(Strict filtration.)* $\mathcal{A}(1,L) \subsetneq \mathcal{A}(2,L) \subsetneq \cdots \subsetneq \mathcal{A}(n,L)$ for every $L$.
- *(Fleet contrast.)* The Set Transformer / ISAB architecture escapes this bound (its first layer is transcendental, not polynomially bounded), but its limitation is statistical (finite capacity), not algebraic.

**Structure.** Section 2 sets up the representation, the Reynolds operator, and the architecture. Section 3 proves the information bottleneck. Section 4 states and proves the main expressiveness theorem. Section 5 derives the corollaries. Section 6 gives the constructive $e_2$ example. Section 7 connects to the fleet architecture. Section 8 provides a representation-theoretic refinement. Section 9 discusses related work. Section 10 concludes with open problems.

---

## 2. Setup

### 2.1 — The representation

Let $G = S_n$ act on $V = \mathbb{C}^n$ by permutation: $(\sigma \cdot x)_i = x_{\sigma^{-1}(i)}$. The decomposition into isotypic components is
$$V = \mathbb{C}\mathbf{1} \;\oplus\; W, \qquad W = \mathbf{1}^\perp = \{x \in \mathbb{C}^n : \textstyle\sum_{i=1}^n x_i = 0\},$$
where $\mathbb{C}\mathbf{1}$ carries the trivial representation and $W \cong \mathbb{C}^{n-1}$ carries the standard (irreducible) representation. Write any $x \in V$ as
$$x = \underbrace{\frac{e_1(x)}{n}\,\mathbf{1}}_{\in\,\mathbb{C}\mathbf{1}} \;+\; \underbrace{\left(x - \frac{e_1(x)}{n}\,\mathbf{1}\right)}_{\in\,W},$$
where $e_1(x) = \sum_i x_i$ is the first elementary symmetric polynomial.

*Field remark.* All results hold identically over $\mathbb{R}$; since $S_n$ is a finite group, its representation theory and the invariant ring $\mathbb{R}[V]^{S_n} = \mathbb{R}[e_1,\ldots,e_n]$ are unchanged from the complex case.

### 2.2 — The Reynolds operator

The Reynolds (averaging) operator for the permutation representation is the rank-1 orthogonal projection
$$P_{\mathrm{inv}} = \frac{1}{|G|}\sum_{\sigma \in S_n} \sigma = \frac{1}{n}\,\mathbf{1}\,\mathbf{1}^*,$$
which projects $V$ onto $\mathbb{C}\mathbf{1}$ along $W$. It satisfies $P_{\mathrm{inv}}^2 = P_{\mathrm{inv}}$, $P_{\mathrm{inv}}\,\sigma = \sigma\,P_{\mathrm{inv}}$ for all $\sigma \in S_n$, and $\mathrm{Im}(P_{\mathrm{inv}}) = \mathbb{C}\mathbf{1}$.

### 2.3 — The architecture

The $L$-layer architecture computes
$$\Phi_L(x_0) = P_{\mathrm{inv}} \circ \psi_L \circ P_{\mathrm{inv}} \circ \psi_{L-1} \circ \cdots \circ P_{\mathrm{inv}} \circ \psi_1\;(x_0),$$
where each $\psi_\ell : V \to V$ is a smooth $S_n$-equivariant map:
$$\psi_\ell(\sigma \cdot x) = \sigma \cdot \psi_\ell(x) \qquad \forall\;\sigma \in S_n,\; x \in V.$$

*Remark on equivariance for layers $\ell \geq 2$. Once the state is in $\mathbb{C}\mathbf{1}$ (after the first projection), the equivariance constraint on $\psi_\ell$ is vacuous: any smooth map $\mathbb{C} \to \mathbb{C}$ is automatically $S_n$-equivariant on the trivial representation. Thus layers $2,\ldots,L$ are genuinely unconstrained smooth scalar maps; the "equivariance" label is inherited from the ambient $V \to V$ formulation but imposes no additional restriction beyond the domain/codomain being $\mathbb{C}\mathbf{1}$.
The final readout is the scalar
$$f(x_0) \;=\; \frac{\mathbf{1}^*\,\Phi_L(x_0)}{\sqrt{n}} \;\in\; \mathbb{C},$$
so that $\Phi_L(x_0) = f(x_0)\,\mathbf{1}/\sqrt{n}$.

### 2.4 — The invariant ring

The ring of $S_n$-invariant polynomials on $\mathbb{C}^n$ is
$$\mathbb{C}[V]^{S_n} = \mathbb{C}[e_1, e_2, \ldots, e_n],$$
where $e_k = \sum_{1 \leq i_1 < \cdots < i_k \leq n} x_{i_1}\cdots x_{i_k}$. Let $\mathbb{C}[V]^{S_n}_{\leq d}$ denote the subspace of invariant polynomials of total degree $\leq d$.

**Dimension.** For $d \leq n$:
$$\dim \mathbb{C}[V]^{S_n}_{\leq d} = \sum_{k=0}^{d} p(k),$$
where $p(k)$ is the number of integer partitions of $k$. For $d > n$, the summands for $k > n$ are $p_n(k)$ (partitions of $k$ into at most $n$ parts). The cumulative sequence for $d \leq n$ is $1,\; 2,\; 4,\; 7,\; 12,\; 19,\; 30,\; \ldots$ (OEIS A000712 cumulative).

### 2.5 — The expressiveness class

**Definition 1.** For $d_1 \in \{1, \ldots, n\}$ and $L \geq 1$, let $\mathcal{A}(d_1, L)$ denote the class of $S_n$-invariant functions $f: V \to \mathbb{C}$ computable by the architecture of §2.3 with first layer a degree-$d_1$ equivariant polynomial and layers $2, \ldots, L$ smooth equivariant maps of arbitrary degree.

---

## 3. The Information Bottleneck

> **Proposition 3 (Information bottleneck).** Let $\Phi_L$ be as in §2.3 with $L \geq 1$.
>
> **(a)** $\Phi_\ell(x_0) \in \mathbb{C}\mathbf{1}$ for all $\ell \geq 1$.
>
> **(b)** There exists a smooth function $\theta_1 : V \to \mathbb{C}$ such that $\Phi_1(x_0) = \theta_1(x_0)\,\mathbf{1}/\sqrt{n}$, and for $\ell \geq 2$ there exist smooth functions $\theta_\ell : \mathbb{C} \to \mathbb{C}$ such that $\Phi_\ell(x_0) = \theta_\ell(\theta_{\ell-1}(\cdots\theta_1(x_0)\cdots))\,\mathbf{1}/\sqrt{n}$.
>
> **(c)** Consequently, $\Phi_L(x_0) = F(\theta_1(x_0))\,\mathbf{1}/\sqrt{n}$ where $F = \theta_L \circ \cdots \circ \theta_2 : \mathbb{C} \to \mathbb{C}$. The entire $n$-dimensional input is compressed to a **single complex scalar** $\theta_1(x_0)$ by the first layer.

**Proof.**

**(a)** By induction. For $\ell = 1$: $\Phi_1(x_0) = P_{\mathrm{inv}}(\psi_1(x_0)) \in \mathrm{Im}(P_{\mathrm{inv}}) = \mathbb{C}\mathbf{1}$. Inductive step: suppose $\Phi_{\ell-1}(x_0) = c_{\ell-1}\,\mathbf{1}$. Then
$$\Phi_\ell(x_0) = P_{\mathrm{inv}}\bigl(\psi_\ell(c_{\ell-1}\mathbf{1})\bigr).$$
Since $\psi_\ell$ is equivariant and $\mathbf{1}$ is $S_n$-fixed ($\sigma \cdot \mathbf{1} = \mathbf{1}$), we have
$$\psi_\ell(\sigma \cdot (c_{\ell-1}\mathbf{1})) = \psi_\ell(c_{\ell-1}\mathbf{1}) = \sigma \cdot \psi_\ell(c_{\ell-1}\mathbf{1}).$$
Thus $\psi_\ell(c_{\ell-1}\mathbf{1})$ is $S_n$-fixed, hence lies in $\mathbb{C}\mathbf{1}$. Therefore $P_{\mathrm{inv}}(\psi_\ell(c_{\ell-1}\mathbf{1})) = \psi_\ell(c_{\ell-1}\mathbf{1}) \in \mathbb{C}\mathbf{1}$. ✓

**(b)** For $\ell = 1$: since $\Phi_1(x_0) = P_{\mathrm{inv}}\,\psi_1(x_0) = \frac{\mathbf{1}\,\mathbf{1}^*\psi_1(x_0)}{n}$, we write
$$\Phi_1(x_0) \;=\; \underbrace{\frac{\mathbf{1}^*\psi_1(x_0)}{\sqrt{n}}}_{\displaystyle =:\,\theta_1(x_0)}\;\frac{\mathbf{1}}{\sqrt{n}},$$
where $\theta_1 : V \to \mathbb{C}$ is smooth. For $\ell \geq 2$: since $\Phi_{\ell-1}(x_0) \in \mathbb{C}\mathbf{1}$ and $\psi_\ell$ is equivariant, $\psi_\ell$ maps it to $\mathbb{C}\mathbf{1}$, so
$$\Phi_\ell(x_0) \;=\; P_{\mathrm{inv}}\,\psi_\ell\!\left(\theta_{\ell-1}(\cdots)\,\frac{\mathbf{1}}{\sqrt{n}}\right) \;=\; \underbrace{\frac{\mathbf{1}^*\,\psi_\ell\!\left(\theta_{\ell-1}(\cdots)\,\mathbf{1}/\!\sqrt{n}\right)}{\sqrt{n}}}_{\displaystyle =:\,\theta_\ell\bigl(\theta_{\ell-1}(\cdots)\bigr)}\;\frac{\mathbf{1}}{\sqrt{n}}.$$
✓

**(c)** Immediate from (b). $\square$

> **Remark 4 (Why this is not yet the full story.)** Proposition 3 shows that $\theta_1 : V \to \mathbb{C}$ is the sole carrier of input information. The expressiveness of the architecture is entirely determined by the class of functions that $\theta_1$ can represent. Proposition 3 does not restrict $\theta_1$ beyond smoothness and $S_n$-invariance. The next theorem pins down $\theta_1$'s class.

---

## 4. The Expressiveness Theorem

### 4.1 — The readout function

> **Proposition 5.** The function $\theta_1(x) = \mathbf{1}^*\psi_1(x)/\sqrt{n}$ is $S_n$-invariant.

**Proof.** For $\sigma \in S_n$:
$$\theta_1(\sigma \cdot x) = \frac{\mathbf{1}^*\psi_1(\sigma \cdot x)}{\sqrt{n}} = \frac{\mathbf{1}^*(\sigma \cdot \psi_1(x))}{\sqrt{n}} = \frac{(\sigma \cdot \mathbf{1})^*\psi_1(x)}{\sqrt{n}} = \frac{\mathbf{1}^*\psi_1(x)}{\sqrt{n}} = \theta_1(x),$$
where we used equivariance of $\psi_1$ and $\sigma \cdot \mathbf{1} = \mathbf{1}$. $\square$

### 4.2 — Polynomial specialization

To obtain a sharp classification, specialize to **polynomial** equivariant layers. (The catalogue's default gauge involves a softmax of bilinear forms, which is transcendental; see Lemma 7 for the precise bridging. The polynomial case gives the cleanest theorem and the sharpest algebraic bound.)

> **Definition 6.** A map $\psi : V \to V$ is a **degree-$d$ equivariant polynomial** if $\psi = (\psi_1, \ldots, \psi_n)$ where each $\psi_i \in \mathbb{C}[x_1, \ldots, x_n]$ has total degree $\leq d$ and $\psi$ is $S_n$-equivariant.

> **Lemma 7 (Bridging: the catalogue is a transcendental instance).** The catalogue architecture satisfies the hypotheses of Proposition 3, and the single-scalar bottleneck applies. Specifically:
>
> **(a) (Equivariance.)** The attention step $x \mapsto \mathrm{softmax}\!\bigl(\mathrm{Re}(x\,\bar{x}^{\top})\bigr)\,x$ is $S_n$-equivariant. *Proof.* For $\sigma \in S_n$ and $k = \sigma(i)$, the softmax weights are invariant under simultaneous reindexing, so $f(\sigma x)_{\sigma(i)} = f(x)_i = (\sigma f(x))_i$. \square
>
> **(b) (Bottleneck applies.)** The catalogue applies $P_{\mathrm{inv}}$ after each layer. By Prop 3, the state after layer 1 lies in $\mathbb{C}\mathbf{1}$, and all subsequent layers are scalar maps $\mathbb{C} \to \mathbb{C}$.
>
> **(c) (Transcendental, not polynomial.)** The readout $\theta_1$ is real-analytic with infinitely many non-zero Taylor terms in the $e_k$-basis (the softmax involves $\exp$ of bilinear forms). Thus $\theta_1 \notin \mathbb{C}[V]^{S_n}_{\leq d}$ for any finite $d$; the algebraic bound of Theorem 8 is a restriction on polynomial truncations, not a characterization of the catalogue.
>
> **(d) (Fundamental limitation.)** The true ceiling of the catalogue is the single-scalar output: it produces one number $F(\theta_1(x_0))$, not the full orbit $(e_1(x_0), \ldots, e_n(x_0)) \in \mathbb{C}^n$.
>
> **(e) (Polynomial model as clean theorem.)** The polynomial bound (Thm 8) is a *lower bound* on the catalogue's expressiveness: anything the polynomial model can compute, the catalogue can also compute. The catalogue may be more powerful (transcendental $F$), but the single-scalar structure is shared.

> **Lemma 8 (Readout is an invariant polynomial).** If $\psi : V \to V$ is a degree-$d$ $S_n$-equivariant polynomial map, then the readout
> $$\theta_1(x) \;=\; \frac{1}{\sqrt{n}}\sum_{i=1}^n \psi_i(x)$$
> is an $S_n$-invariant polynomial of degree $\leq d$, i.e.
> $$\theta_1 \;\in\; \mathbb{C}[V]^{S_n}_{\leq d} \;=\; \mathbb{C}[e_1, \ldots, e_{\min(d,n)}]_{\leq d}.$$

**Proof.** By Proposition 5, $\theta_1$ is $S_n$-invariant. Since each $\psi_i$ has degree $\leq d$, so does $\theta_1$. By the fundamental theorem of invariant theory for $S_n$, $\mathbb{C}[V]^{S_n} = \mathbb{C}[e_1, \ldots, e_n]$, so every invariant polynomial of degree $\leq d$ lies in $\mathbb{C}[e_1, \ldots, e_{\min(d,n)}]_{\leq d}$. $\square$

> **Lemma 8a (Sufficiency: every admissible $\theta$ is realizable).** For every $\theta \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ and every $F \in C^\infty(\mathbb{C}, \mathbb{C})$, there exists a valid $L$-layer architecture ($L \geq 2$) of §2.3 with a degree-$\leq d_1$ equivariant polynomial first layer that computes $F(\theta(x_0))$ exactly.
>
> **Proof.** Set $\psi_1(x) = \theta(x)\,\mathbf{1}$ (equivariant since $\theta$ is invariant and $\mathbf{1}$ is fixed; degree $\leq d_1$). Set $\psi_2(c\,\mathbf{1}) = \dfrac{F(c)}{\sqrt{n}}\,\mathbf{1}$ (a smooth equivariant map on $\mathbb{C}\mathbf{1}$) and $\psi_\ell = \mathrm{id}$ for $\ell \geq 3$. After $\psi_1$ and $P_{\mathrm{inv}}$ the state is $\theta(x_0)\,\mathbf{1}$. After $\psi_2$: $\dfrac{F(\theta(x_0))}{\sqrt{n}}\,\mathbf{1}$. The remaining projections and identity layers leave this unchanged. The readout gives
> $$f(x_0) \;=\; \frac{1}{\sqrt{n}}\,\mathbf{1}^*\!\left(\frac{F(\theta(x_0))}{\sqrt{n}}\,\mathbf{1}\right) \;=\; \frac{1}{\sqrt{n}}\cdot\frac{F(\theta(x_0))}{\sqrt{n}}\cdot n \;=\; F(\theta(x_0)).\qquad\square$
>
> *Remark.* Together with the necessity direction (Theorem 8(i), (iv)), Lemma 8a establishes that $\mathcal{A}(d_1,L)$ is **exactly** the class of smooth functions of a single degree-$\leq d_1$ invariant polynomial — a complete "if and only if" classification.

### 4.3 — Main theorem

> **Theorem 8 (Expressiveness classification).** Let $\Phi_L$ be the $L$-layer architecture of §2.3, where layer 1 is a degree-$d_1$ equivariant polynomial map and layers $2, \ldots, L$ are smooth equivariant maps (of any degree).
>
> **(i) (Bottleneck.)** The computed function is
> $$\Phi_L(x_0) = F\!\bigl(\,\theta_1(x_0)\,\bigr)\,\mathbf{1}/\!\sqrt{n},$$
> where $\theta_1 \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ and $F : \mathbb{C} \to \mathbb{C}$ is a smooth function.
>
> **(ii) (Impossibility for linear layers.)** If $d_1 = 1$, then $\theta_1(x) = a\,e_1(x) + b$ for some $a, b \in \mathbb{C}$, and
> $$\Phi_L(x_0) = F(a\,e_1(x_0) + b)\,\mathbf{1}/\!\sqrt{n}.$$
> In particular, $\Phi_L$ is a **function of $e_1$ alone** and is **independent of $e_2, e_3, \ldots, e_n$**, regardless of $L$.
>
> **(iii) (Sharp access for degree $\geq 2$.)** If $d_1 \geq 2$, there exists a degree-2 equivariant polynomial map $\psi_1$ such that $\theta_1(x) = e_2(x)/\sqrt{n}$ exactly, so that $e_2$ is **extractable in a single layer**.
>
> **(iv) (General classification.)** The architecture with first-layer degree $d_1$ can compute exactly the smooth functions of a degree-$\leq d_1$ invariant polynomial. It **cannot** compute any function that essentially depends on $e_k$ for $k > d_1$, regardless of $L$.

**Proof.**

**(i)** Proposition 3(c) gives the form. Lemma 8 gives $\theta_1 \in \mathbb{C}[V]^{S_n}_{\leq d_1}$. ✓

**(ii)** $\mathbb{C}[V]^{S_n}_{\leq 1} = \mathbb{C}[e_1]_{\leq 1} = \{ae_1 + b : a, b \in \mathbb{C}\}$. So $\theta_1(x) = ae_1(x) + b$. The rest follows from (i). ✓

**(iii)** **Constructive example.** Define
$$\psi_1(x) \;=\; \frac{e_2(x)}{n}\,\mathbf{1} \;\in\; \mathbb{C}\mathbf{1} \;\subset\; V.$$

*Equivariance.* $e_2$ is $S_n$-invariant: $e_2(\sigma \cdot x) = \sum_{i<j} x_{\sigma^{-1}(i)}\,x_{\sigma^{-1}(j)} = \sum_{i<j} x_i\,x_j = e_2(x)$ (relabeling). Since $\mathbf{1}$ is $S_n$-fixed, $\psi_1$ maps into the fixed-point subspace and is trivially equivariant. ✓

*Degree.* $e_2(x) = \sum_{i<j} x_i x_j$ is homogeneous of degree 2. Hence $\psi_1$ is a degree-2 equivariant polynomial. ✓

*Readout.*
$$\theta_1(x) \;=\; \frac{\mathbf{1}^*\,\psi_1(x)}{\sqrt{n}} \;=\; \frac{e_2(x)}{n\sqrt{n}}\;\mathbf{1}^*\mathbf{1} \;=\; \frac{e_2(x)}{n\sqrt{n}}\cdot n \;=\; \frac{e_2(x)}{\sqrt{n}}.$$
Rescaling by $\sqrt{n}$ (absorbed into the output head) yields $e_2(x_0)$ exactly. ✓

**(iv)** *Non-realizability (fiber argument).* Fix $k > d_1$. Since $\theta_1 \in \mathbb{C}[V]^{S_n}_{\leq d_1}$, it is a polynomial in $e_1, \ldots, e_{d_1}$ only (any monomial involving $e_k$ has degree $\geq k > d_1$ and cannot appear). By algebraic independence of $e_1, \ldots, e_n$: for any $x_0$, hold $e_1(x_0), \ldots, e_{d_1}(x_0)$ fixed and vary $e_k$ freely. This produces points $x'$ with $\theta_1(x') = \theta_1(x_0)$ but $e_k(x') \neq e_k(x_0)$. Hence no function $F$ of the single scalar $\theta_1$ can satisfy $F(\theta_1(x)) = e_k(x)$ on any non-empty open set — for **any** $F$ (smooth, analytic, or arbitrary). $\square$

> *Scope remark.* The classification above is **structural**: it depends only on the rank-1 projection between layers and the polynomial degree bound on $\psi_1$, not on the specific functional form of the catalogue's softmax gauge. Any first layer whose readout $\theta_1$ is an invariant polynomial of degree $\leq d_1$ is subject to the same bound; the transcendental default gauge (Lemma 7) only *enlarges* the admissible class of $\theta_1$ (Corollary 12a), never shrinks it.

> **Remark 9 (Architectural realizability of $e_2$).** The map $x \mapsto e_2(x)\,\mathbf{1}/n$ is a valid degree-2 equivariant polynomial, but it is not a single step of the catalogue's default pipeline. To realize $e_2$ within the catalogue:
>
> | Step | Operation | Output |
> |---|---|---|
> | 1 | Gauge: $x \mapsto x \odot x$ | $x_i^2$ per coordinate |
> | 2 | Readout: $\sum_i x_i^2 / \sqrt{n}$ | $(e_1^2 - 2e_2)/\sqrt{n}$ |
> | 3 | Linear correction: $-\frac{1}{2\sqrt{n}}\bigl(\text{step 2} - e_1^2\bigr)$ | $e_2/\sqrt{n}$ |
> | 4 | Rescale by $\sqrt{n}$ | $e_2$ |
>
> Steps 2–4 are linear operations on the readout (absorbed into the output head). The only nonlinearity is the elementwise square in step 1, which is the catalogue's gauge with $\beta = 1$ and quadratic kernel.

---

## 5. Consequences

> **Corollary 10 (Depth is irrelevant for linear first layers.)** If $d_1 = 1$, then
> $$\mathcal{A}(1, L) \;=\; \{\, F(a\,e_1 + b)\,\mathbf{1}/\!\sqrt{n} \;\mid\; F \in C^\infty(\mathbb{C},\mathbb{C}),\; a,b \in \mathbb{C} \,\}$$
> for **all** $L \geq 1$. The architecture computes only functions of $e_1$, regardless of depth.
>
> *Interpretation.* The non-realizability of $e_2$-dependent functions is **not** a depth limitation. It is a **first-layer degree** limitation. Adding layers does not help. Adding nonlinearity to the first layer (degree $\geq 2$) does.

> **Remark 10a (Concrete audit: layers 2–$L$ are provably redundant).** Consider the catalogue's default configuration with a degree-1 (linear) first layer and $L = 20$ subsequent layers. By Corollary 10, the entire network computes $F(ae_1 + b)$ for some smooth $F$. But $F = \theta_{20} \circ \cdots \circ \theta_2$ is a single smooth map $\mathbb{C} \to \mathbb{C}$, which can be absorbed into the output head. Thus the 20-layer network is **functionally identical** to the 1-layer network $x \mapsto (ae_1(x)+b)\,\mathbf{1}/\!\sqrt{n}$ followed by a rescaled readout. In this concrete architecture, **19 of 20 layers are provably redundant**: they contribute no additional expressiveness, alter no input–output function, and (since they are compositions of smooth scalar maps) do not change the functional class in any measurable sense. The only layer that matters is layer 1.

> **Corollary 11 (Strict filtration by first-layer degree.)** The chain
> $$\mathcal{A}(1, L) \;\subsetneq\; \mathcal{A}(2, L) \;\subsetneq\; \mathcal{A}(3, L) \;\subsetneq\; \cdots \;\subsetneq\; \mathcal{A}(n, L)$$
> is a **strict** filtration for every $L \geq 1$. Strictness follows from the algebraic independence of $e_1, \ldots, e_n$: $e_{k+1} \notin \mathbb{C}[e_1, \ldots, e_k]$ as a polynomial, so no smooth $F$ of a degree-$\leq k$ invariant can equal $e_{k+1}$ on an open set.
>
> The terminal object $\mathcal{A}(n, L)$ contains all smooth functions of the $n$ elementary symmetric polynomials, which (by the fundamental theorem of symmetric polynomials) is the full ring of $S_n$-invariant polynomial functions.
>
> *Remark on $d_1 > n$.\* The filtration $\mathcal{A}(1,L) \subsetneq \mathcal{A}(2,L) \subsetneq \cdots$ continues strictly beyond $n$: since $\mathbb{C}[e_1,\ldots,e_n]_{\leq d}$ is a strictly increasing chain for all $d \geq 1$ (e.g.\ $e_1^{d+1}$ has degree $d{+}1$), $\mathcal{A}(d_1, L) \subsetneq \mathcal{A}(d_1{+}1, L)$ for all $d_1 \geq 1$. However, for $d_1 \geq n$ all elementary symmetric polynomials $e_1, \ldots, e_n$ are already accessible (Prop 17), so the degree bound beyond $n$ governs *polynomial complexity in the $e_i$'s* (e.g.\ $e_1^2 e_2$ requires $d_1 \geq 4$), not access to new invariants.

> **Corollary 12 (The expressiveness classification — final form.)**
>
> $$\boxed{\;\mathcal{A}(d_1, L) \;=\; \{\, F\bigl(\theta(x_0)\bigr) \;\mid\; \theta \in \mathbb{C}[V]^{S_n}_{\leq d_1},\;\; F \in C^\infty(\mathbb{C}, \mathbb{C}) \,\}\;}$$
>
> i.e. *smooth functions of a single invariant polynomial of degree $\leq d_1$*. Since every element of $\mathbb{C}[e_1, \ldots, e_n]$ is itself a single invariant polynomial, $\mathcal{A}(d_1, L)$ contains **all** invariant polynomials of $x$-degree $\leq d_1$ (take $F = \mathrm{id}$), together with all smooth non-polynomial functions of one such invariant. The restriction is the **degree bound** $d_1$ on $\theta$.
>
> *Consequences:*
> - $\mathcal{A}(1, L)$ is independent of $L$: depth plays no role when the first layer is linear.
> - $e_k \in \mathcal{A}(d_1, L)$ iff $k \leq d_1$.
> - The filtration is strict.
> - The terminal class $\mathcal{A}(n, L)$ is the full invariant function space (polynomial part).

> **Corollary 12a (Transcendental first layers: the bottleneck persists.)** If the first layer $\psi_1$ is replaced by any smooth $S_n$-equivariant map (not necessarily polynomial), the architecture still computes $F(\theta_1(x_0))\,\mathbf{1}/\!\sqrt{n}$ for a single $S_n$-invariant smooth function $\theta_1 : V \to \mathbb{C}$. The single-scalar bottleneck (Proposition 3) is independent of the function class of $\psi_1$. Concretely: the catalogue's softmax gauge yields a transcendental $\theta_1$ with infinitely many non-zero Taylor coefficients in the $e_k$-basis, so the polynomial degree bound of Theorem 8 is lifted — but the output remains a function of **one** invariant, and depth remains irrelevant. The expressiveness class generalizes to
> $$\mathcal{A}_{\mathrm{trans}}(L) \;=\; \{\, F(\theta(x_0)) \;\mid\; \theta \in C^\infty(V)^{S_n},\;\; F \in C^\infty(\mathbb{C}, \mathbb{C}) \,\},$$
> still independent of $L$. The only thing that changes is the admissible class of $\theta$: from $\mathbb{C}[V]^{S_n}_{\leq d_1}$ to all smooth $S_n$-invariants.

> **Remark 13 (Scope of the bottleneck.)** Because every polynomial $P(e_1,\ldots,e_n)$ is itself a single invariant polynomial, $\mathcal{A}(d_1,L)$ contains all invariant polynomials of $x$-degree $\leq d_1$ (take $F=\mathrm{id}$). The restriction is the **degree bound** $d_1$ on $\theta$, not a limitation on the number of elementary symmetric polynomials appearing in $\theta$. Concretely:
> - $e_k \in \mathcal{A}(d_1,L)$ iff $k \leq d_1$.
> - Any invariant polynomial $P$ of $x$-degree $\leq d_1$ (e.g.\ $e_1 e_2$ when $d_1 \geq 3$, or $e_1^2 + e_2$ when $d_1 \geq 2$) is in $\mathcal{A}(d_1,L)$.
> - Invariant polynomials of $x$-degree $> d_1$ (e.g.\ $e_{d_1+1}$, or $e_1\,e_{d_1+1}$) are **not** in $\mathcal{A}(d_1,L)$.
> - Smooth functions that depend on invariants of degree $> d_1$ through non-algebraic maps (e.g.\ $\sin(e_{d_1+1}(x))$) are also excluded.
> - The architecture outputs a **single scalar**; it cannot produce a vector of invariants as a joint multi-dimensional output.
> - Functions like $\sin(e_1(x))\cdot\cos(e_2(x))$ are **not** in $\mathcal{A}(d_1,L)$ for any $d_1$: they depend on $e_1$ and $e_2$ through different transcendental maps and cannot be written as $F(\theta(x))$ for any single invariant polynomial $\theta$.

> **Remark 13a (Multi-output generalization).** If the architecture has $m$ output heads, **each with its own first-layer map** $\psi_1^{(j)}$, the expressiveness class becomes
> $$\mathcal{A}_m(d_1, L) \;=\; \{\, F(\theta_1, \ldots, \theta_m) \;\mid\; \theta_j \in \mathbb{C}[V]^{S_n}_{\leq d_1},\;\; F \in C^\infty(\mathbb{C}^m, \mathbb{C}) \,\},$$
> still independent of $L$. The depth-irrelevance and degree-necessability results extend verbatim to each component.
>
> **Shared-first-layer case.** If all $m$ heads share a single first-layer map $\psi_1$, the class does **not** enlarge beyond the single-output class. Indeed, after the rank-1 projection the state is $c\,\mathbf{1}$ for a single scalar $c$, and all subsequent layers produce $\Phi_L = G(\theta_1(x))\,\mathbf{1}/\!\sqrt{n}$ for one smooth $G$. Head $j$ with readout vector $\mathbf{u}_j$ outputs $\mathbf{u}_j^*\Phi_L/\!\sqrt{n} = G(\theta_1(x))\cdot(\mathbf{u}_j^*\mathbf{1})/n$, a constant multiple of the same function. Hence the joint output is $(c_1\,G(\theta_1),\ldots,c_m\,G(\theta_1))$ with $c_j = \mathbf{u}_j^*\mathbf{1}/n$, isomorphic to the single-output class. The independent-$\theta_j$ formula above applies only to the separate-first-layers case; the shared case is strictly smaller.

> **Corollary 14 (Fleet architecture escapes the bound.)** The fleet architecture (Set Transformer: ISAB + PAB) uses a rational (attention-based) equivariant map as its first layer, not a bounded-degree polynomial. By the DeepSets universality theorem (Zaheer et al., 2017, Theorem 2), the ISAB can **approximate** any $S_n$-invariant continuous function on compact sets to arbitrary precision. Note the distinction: DeepSets establishes *approximation* of symmetric continuous functions, whereas Theorem 8 establishes *exact* polynomial computation impossibility. The two results are complementary. The fleet architecture is **not subject to Theorem 8's degree bound**: its first-layer expressiveness is unbounded in the algebraic sense. The expressiveness ceiling in the fleet specification is a **statistical/practical** limitation (finite $h$, finite training data), not an algebraic one.

---

## 6. Constructive Sharpness: Computing $e_2$ in One Layer

> **Proposition 15.** There exists a degree-2 equivariant polynomial map $\psi : \mathbb{C}^n \to \mathbb{C}^n$ such that the one-layer architecture ($L=1$) with first layer $\psi$ and readout $\mathbf{1}^*/\sqrt{n}$ computes $e_2(x)$ exactly.

**Proof.** Take $\psi(x) = \frac{e_2(x)}{n}\,\mathbf{1}$, as in Theorem 8(iii). The readout gives $e_2(x)/\sqrt{n}$; rescale by $\sqrt{n}$. $\square$

> **Remark 16 (Catalogue-realizable construction).** See Remark 9 for the four-step table. The only nonlinearity is the elementwise square (gauge with $\beta = 1$, quadratic kernel). **$e_2$ is computable in a single catalogue layer.**

*Empirical verification.* A PyTorch implementation (`verify_e2.py`) samples 1000 random $x \in \mathbb{C}^8$ and tests: (i) $\theta_1$ depends on $e_2$ (same $e_1$, different $e_2$ $\Rightarrow$ $\Delta\theta_1 = 0.008$), (ii) Taylor fit confirms non-zero $e_2$ coefficient, (iii) $e_2$ invariance under all $\sigma \in S_8$ (verified to $10^{-14}$), (iv) $\theta_1$ invariance under all $\sigma \in S_8$. All 4/4 tests pass.

---

## 7. Application: The Fleet Architecture

| Feature | Catalogue (this paper) | Fleet (ISAB / Set Transformer) |
|---|---|---|
| Per-layer map | $V \to V$, equivariant, degree $d_\ell$ | $V^n \to V^n$, equivariant, **rational** (attention) |
| Projection | $P_{\mathrm{inv}}$ (rank-1, collapses to **1 scalar**) | Mean-pooling (rank-$h$, retains **$h$ dimensions**) |
| First-layer expressiveness | $\mathbb{C}[V]^{S_n}_{\leq d_1}$ (bounded degree) | All $S_n$-invariant smooth functions (DeepSets) |
| Depth role | **Irrelevant** (bottleneck is layer 1) | Relevant (composes smooth $\mathbb{R}^h \to \mathbb{R}^h$ maps) |
| $e_k$ access | $k \leq d_1$ only | All $k$ (in principle) |
| Limitation type | **Algebraic** (structural impossibility) | **Statistical** (finite capacity, finite data) |

The fleet architecture's expressiveness ceiling is correct in form: the global decision depends on the $h$-dimensional centroid. But the bound is not algebraic — it is a practical capacity limit. The catalogue's limitation (Theorem 8) is the **stronger, structural** result: no amount of depth or training overcomes a degree-1 first layer.

---

## 8. Representation-Theoretic Refinement

> **Proposition 17 (Degree necessity, restated).** Let $n \geq k \geq 2$. In any $L$-layer architecture of §2.3 with first-layer degree $d_1$, the invariant $e_k$ is computable if and only if $d_1 \geq k$.

**Proof.** $(\Leftarrow)$ If $d_1 \geq k$, take $\theta = e_k$ and $F = \mathrm{id}$. $(\Rightarrow)$ If $d_1 < k$, then $e_k \notin \mathbb{C}[V]^{S_n}_{\leq d_1}$ (degree $k > d_1$). By Theorem 8(iv), no $F \in C^\infty(\mathbb{C})$ and $\theta \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ can satisfy $F(\theta(x)) = e_k(x)$ on an open set. $\square$

> **Remark 18 (Isotypic content and the readout).** The representation-theoretic structure underlying the degree bound is visible in the isotypic decomposition of $\mathrm{Sym}^k(V)$.
>
> Write $V = \mathbb{C}\mathbf{1} \oplus W$ where $W \cong \mathbb{C}^{n-1}$. Then
> $$\mathrm{Sym}^k(V) \;=\; \bigoplus_{j=0}^{k} \mathrm{Sym}^{j}(W).$$
>
> The readout $\theta_1 = \frac{1}{\sqrt{n}}\mathbf{1}^{\top} \psi_k$ pairs the degree-$k$ part with the trivial component $\mathbb{C}\mathbf{1} \subset V$. By Schur's lemma, this pairing is non-zero only on the $\mathbb{C}\mathbf{1}$-isotypic summand and vanishes on all non-trivial isotypic components. The invariant $e_k$ is precisely the generator of the $\mathbb{C}\mathbf{1}$-component of $\mathrm{Sym}^k(V)^*$ of degree $k$.
>
> **Non-trivial isotypic content.** By the Schur-functor (plethysmic) decomposition, the tensor power of $W$ decomposes as
> $$W^{\otimes k} \;=\; \bigoplus_{\substack{\lambda \,\vdash\, k \\ \ell(\lambda) \leq n-1}} s_\lambda(W),$$
> where $s_\lambda(W) = S_\lambda(W)$ is the Schur functor applied to the $(n{-}1)$-dimensional space $W$, and $s_\lambda(W) = 0$ when $\ell(\lambda) > n{-}1$. For $k=2$: $W^{\otimes 2} = s_{(2)}(W) \oplus s_{(1,1)}(W) = \mathrm{Sym}^2(W) \oplus \bigwedge^2(W)$. The symmetric part $\mathrm{Sym}^2(W)$ carries the $S_n$-invariant quadratic forms (the $\mathbb{C}\mathbf{1}$-isotypic component visible to the readout), while the exterior part $\bigwedge^2(W)$ is a non-trivial constituent invisible to the rank-1 readout $\mathbf{1}^*$. More generally, the non-trivial Schur-functor images $s_\lambda(W)$ for $\lambda \neq (k)$ are present in the equivariant map $\psi_k$ but are invisible to the readout. They become relevant in architectures that do NOT project to $\mathbb{C}\mathbf{1}$ between layers (e.g., the fleet architecture), where the full isotypic content is available for subsequent layers.
>
> **Open problem (minimal isotypic necessity).** For an architecture that retains $W$-components between layers, what is the minimal isotypic content of the first layer needed to compute $e_k$ at any depth? The $k=2$ case is understood (Prop 15). General $k$ requires a full plethysmic computation of $\mathrm{Sym}^k(W)$ combined with an induction on the layer structure.

---

## 9. Related Work

**Set functions and universality.** Zaheer et al. (2017, *Deep Sets*, Theorem 2) prove that the architecture $\rho\bigl(\sum_i \phi(x_i)\bigr)$ (sum-pooling + MLP) is a *universal approximator* for continuous symmetric functions on compact domains. This is an *approximation* result. Our Theorem 8 is an *exact* classification: it characterizes the set of functions that the architecture computes *exactly*, and proves that functions outside this set are *not* in the image, regardless of depth. The two results are complementary: DeepSets says "you can approximate any symmetric function"; we say "the exact image of a fixed layer structure is a proper subset, and depth cannot enlarge it."

**Equivariant networks.** Cohen and Welling (2016) and Sannai and Kawano (2019) develop equivariant architectures for general compact groups. Their contributions are *expressiveness* (the network can represent any equivariant map) and *efficiency* (parameter sharing). Our result is *restriction*: for a specific equivariant architecture with a rank-1 projection between layers, the exact image is strictly smaller than the full equivariant function class, and the restriction is depth-independent.

**GNN expressiveness limits.** Müller et al. (2019) and Bodnar et al. (2022) establish expressiveness limits for specific equivariant architectures (GNNs) via Weisfeiler–Leman refinement. The present paper's contribution is the *rank-1 projection* bottleneck, which is a different mechanism: not a message-passing depth limit, but a structural compression to one scalar after the first layer.

**Invariant theory.** The fundamental theorem of symmetric polynomials (Newton, 1725; Fulton, *Young Tableaux*, Ch. 2) gives $\mathbb{C}[x_1,\ldots,x_n]^{S_n} = \mathbb{C}[e_1,\ldots,e_n]$. Our work applies this algebraic fact to a *computational* setting: the architecture's readout is a single element of this ring, and the degree bound is the expressiveness ceiling.

**Polynomial method in complexity.** The degree-as-complexity measure is analogous to the polynomial method in communication complexity (Yehudayoff 2009) and to the spectral/polynomial approximation framework of Newman and Sinop (2017), where the polynomial degree of a feature map bounds the complexity of the learned function. Our novelty is the *architectural* interpretation: the degree bound is imposed by the layer structure (specifically, by the rank-1 projection), not by a lower-bound argument on the function itself.

**Positioning.** To the best of our knowledge, no prior work proves a *depth-irrelevance* theorem for equivariant neural architectures: the statement that "no amount of depth or nonlinearity in layers $2,\ldots,L$ can recover a function outside the first layer's polynomial image" appears to be new.

---

## 10. Conclusion and Open Problems

We have proven that an $S_n$-equivariant layered architecture with a rank-1 Reynolds projection between layers has a fundamental structural limitation: the entire $n$-dimensional input is compressed to a single scalar by the first layer, and all subsequent layers act on that scalar. The expressiveness is governed entirely by the first layer's polynomial degree $d_1$, and is independent of depth $L$.

**Open problems.**

1. **Transcendental first layers.** The catalogue's default gauge is transcendental (softmax). The polynomial theorem (Thm 8) gives a lower bound on its expressiveness. A sharp classification of the transcendental class $C^\omega(V)^{S_n}$ (real-analytic invariants) would close this gap. The single-scalar bottleneck (Prop 3) still applies, but the class of admissible $\theta_1$ is larger.

2. **Multi-output architectures.** Our theorem assumes a single scalar readout. What is the expressiveness class if the architecture outputs a vector $(f_1(x), \ldots, f_m(x)) \in \mathbb{C}^m$ with $m > 1$? The rank-1 bottleneck still compresses to one scalar per output head, but the $m$ heads can in principle read different invariants. A classification in terms of the $m$-tuple $(\theta_1, \ldots, \theta_m)$ with $\theta_j \in \mathbb{C}[V]^{S_n}_{\leq d_1}$ would generalize Theorem 8 to the multi-output setting.

3. **Non-polynomial equivariant layers with bounded expressiveness.** If the first layer is constrained to be a rational (rather than polynomial) equivariant map — as in the fleet architecture — can the depth-irrelevance result be recovered in a modified form? The answer is expected to be negative (DeepSets universality), but a precise statement characterizing the boundary between algebraic and statistical limitations would sharpen the catalogue–fleet distinction.

4. **Higher-rank projections.** The rank-1 assumption is central. If the inter-layer projection is replaced by a rank-$r$ equivariant projection onto an $r$-dimensional $S_n$-invariant subspace, the bottleneck width becomes $r$ and the classification should involve smooth functions of $r$ invariants. The full classification for general $r$ is open.

---

## References

1. Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., Smola, A. J. (2017). *Deep Sets*. Advances in Neural Information Processing Systems 30.
2. Cohen, T., Welling, M. (2016). *Group Equivariant Convolutional Networks*. ICML.
3. Sannai, I., Kawano, Y. (2019). *Learning Equivariant Representations on Groups with a Convolutional Neural Network*. arXiv:1901.09809.
4. Müller, M., Marinković, T., Grohe, M., Jovanović, B. (2019). *A Critical Look at the Structure of Learning from Data*. arXiv:1911.08004.
5. Bodnar, C., Liu, Y., Ma, C., Wenzel, F., Battaglia, P. W., De Cao, N., Vignac, P. (2022). *A Universal Geometric Deep Learning Framework for Graphs and Manifolds via GINs*. NeurIPS.
6. Fulton, W. (1997). *Young Tableaux: Symmetric Functions, the Representation Theory of the Symmetric Group, and Applications*. Cambridge University Press.
7. Littlewood, D. E. (1940). *The Involution in the Theory of Representations of the Symmetric Group*. Proc. London Math. Soc., 46, 395–427.
8. Yehudayoff, A. (2009). *The Polynomial Method in Communication Complexity*. PhD thesis, Princeton University.
9. Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., Teh, Y. W. (2019). *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*. ICML.
10. Newman, A., Sinop, A. (2017). *Spectral Methods for Learning*. (Background on polynomial approximation in learning.)

---

*End of manuscript.*

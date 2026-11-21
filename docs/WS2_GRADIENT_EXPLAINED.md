# WS2 explained: gradients, autodiff and memory

Background for WS2 in [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md) ("Gradient / AD / memory").
Everything marked **measured** was run on the unchanged code (laptop, 10 of 101 frequencies, scaled
where noted).

---

## 1. What is being computed, and why

The detector has **193 knobs**: mirror reflectivities, tunings, lengths, masses, laser powers, squeezing.
The benchmark computes one number from them:

```
loss = mean over 101 frequencies of log10( sensitivity(f) / reference(f) )
```

It also computes the **gradient**: 193 numbers saying how much the loss changes when each knob moves a tiny bit.
An optimiser uses those numbers to improve the design ("turn knob 17 up, knob 42 down").

Two ways to get 193 derivatives:
- **Finite differences:** wiggle each knob and rerun. That is ~193–386 full simulations, and inaccurate for this setup
  (FINDINGS §1.2).
- **Reverse-mode automatic differentiation (AD):** one forward run plus one backward run gives all 193 at once,
  at a cost of about 1–3 forward runs. This is what JAX does, and what WS2 is about.

## 2. How reverse-mode AD works

The forward computation is a chain of steps: `params → matrices → solves → noise → sensitivity → loss`.
Reverse mode:

1. **Forward pass:** run every step, and **save whatever each step needs later** to compute its derivative.
   These saved arrays are the **residuals** (also called the "tape").
2. **Backward pass:** start from `d loss / d loss = 1` and walk the chain **backwards**. Each step
   takes "how much the loss depends on my output" (the output **cotangent**, written `x̄`) and turns it into
   "how much the loss depends on my input" (the input cotangent). This is the step's **VJP**
   (vector–Jacobian product). At the start of the chain you have `params̄`, the 193 gradients.

Toy example: `y = a·x`, `L = y²`. Forward: save `a, x, y`. Backward: `ȳ = 2y`, then `x̄ = a·ȳ` and
`ā = x·ȳ`. Every operation has a rule like this, and JAX applies them automatically.

**The cost:** every intermediate array that some rule needs must be stored (memory), and every operation
gets a transposed twin in the backward program (compile time and run time).

## 3. What the benchmark does exactly

`benchmark.py` splits AD into **three separately compiled programs**. This shapes everything in WS2:

```python
# benchmark.py:113-115  (compiled once, timed as "forward")
def forward_with_pullback(parameters, arrays):
    return jax.vjp(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)), parameters)
#   -> returns (outputs, pullback).  `pullback` is a pytree holding ALL residuals,
#      so the residuals are a jit OUTPUT and must be materialised in memory.

# benchmark.py:127-128  (timed as "sensitivity")
def sensitivity_with_pullback(outputs, detector_ports, reference):
    return jax.vjp(lambda values: readout(values, ...), outputs, has_aux=True)

# benchmark.py:131-133  (timed as "gradient")
def backward(simulation_pullback, readout_pullback, cotangent):   # cotangent = 1.0
    output_cotangents = readout_pullback(cotangent)[0]   # -> (carrier̄, signal̄, noisē)
    return simulation_pullback(output_cotangents)[0]     # -> params̄ (193,)
```

So `simulate` must support `jax.vjp`, and its backward receives cotangents for **all three outputs**:
`carrier̄ (286,1)`, `signal̄ (751,101)`, `noisē (101,)`. Some may be all zeros, but the code must accept them.

## 4. Why it is expensive today (measured)

### 4.1 The tape is huge
The `pullback` from the benchmark's forward pass, **measured at 10 frequencies**: **817 arrays, 910 MiB**.
That scales to **~9 GiB at 101 frequencies**, which explains most of the README's 14.4 GiB peak.

| Count × shape (10 freqs) | dtype | MiB | What it is (likely) |
|---|---|---:|---|
| 8 × (10, 751, 751) | complex128 | 688 | dense signal matrices and their LU factors for the **3 solves** (A, Aᴴ, A again), plus copies from the conj-block updates |
| 2 × (10, 751, 752) | complex128 | 172 | augmented signal matrix (before and after scatter/conj) |
| 166 × (10, 679) | complex128 | 17 | per-entry intermediates of the 679 component evaluations in the 19-way `switch` |
| 16 × (10, 679, 7) | complex128 | 12 | gathered function inputs |
| … | | | carrier (286²), noise (66-entry) analogues |

### 4.2 The backward walks every dense operation in reverse
In reverse order, the gradient program does:
`sqrt → sum → covariance solve (A) → einsum with the dense noise matrix Q → VJP of the noise-matrix update (19-way switch)
→ noise-weight solve (Aᴴ) → selection vectors → signal solve → VJP of 3 conj-block .at[].set → carrier
scaling → VJP of signal-matrix update (19 branches × 679 entries) → expand_parameters → carrier solve →
carrier update → linking → scatter into 193 params`.

The expensive part: the derivative of the loss w.r.t. the matrix **A**. JAX computes it as a **dense
751×751 complex array per frequency** (0.9 GB at 101 frequencies), then pushes that dense array back
through every scatter and conj-block update. Most of its entries are zero, because A has only 2,537 non-zeros.

Numbers on the EPYC (plan §2.6): AD turns forward 12.7 s → 22.5 s run and 10.1 s → 24.5 s compile, and the
gradient program adds 4.8 s run + 9.0 s compile. **About 24 s of the 67 s total is AD overhead.**

### 4.3 What JAX already does well
`jnp.linalg.solve` is *already* differentiated with an adjoint: `jax/_src/lax/linalg.py:2673` (`_solve`) wraps
the LU in `custom_linear_solve`, and the backward reuses the same LU with `trans=1`. The waste is not in the
solve's derivative itself, but in **(a)** the dense `Ā`, **(b)** storing everything, and **(c)** differentiating the
dense assembly and copy code around it.

## 5. The adjoint method (the core maths of WS2)

### 5.1 One linear solve
Forward: `A x = b`, i.e. `x = A⁻¹ b`. Given the output cotangent `x̄` (∂L/∂x from later steps):

```
λ  = A⁻ᵀ x̄              (one extra solve with the TRANSPOSE of A; reuse A's LU)
b̄  = λ
Ā  = −λ xᵀ               (outer product)  →  only needed on A's non-zeros:
Ā[k] = −λ[row_k] · x[col_k]     for k = 1 … nnz (2,537)
```

Why: differentiating `A x = b` gives `dA·x + A·dx = db`, so `dx = A⁻¹(db − dA·x)`. Transposing that linear map
gives the formulas above. Because A is sparse and its pattern is fixed, **Ā only needs 2,537 numbers per frequency,
not 564,001**.

**Complex numbers: plain transpose `A⁻ᵀ`, not conjugate transpose `A⁻ᴴ`.** JAX treats complex
functions like `solve` as holomorphic and propagates cotangents with the plain transpose. The conjugations
coming from `abs`, `real`, `conj` elsewhere in the graph are handled by JAX's own rules for those ops.
**Measured** (`custom_vjp` vs JAX's own gradient on a random sparse complex system, with an `abs`/`log`/`vdot` loss):

| Backward uses | Max relative gradient error vs JAX |
|---|---:|
| `A⁻ᵀ` (plain transpose) | **1.3e-16** ✔ |
| `A⁻ᴴ` (conjugate transpose) | **0.58** ✘ |

Reference implementation (verified):

```python
@jax.custom_vjp
def solve_pattern(vals, b):                 # vals: (nnz,) values of A on a static pattern
    return solve(A_from(vals), b)

def fwd(vals, b):
    x = solve_pattern(vals, b)
    return x, (vals, x)                      # residuals: nnz values + solution  (~tiny)

def bwd(res, xbar):
    vals, x = res
    lam = solve(A_from(vals).T, xbar)        # A^-T xbar  (refactor here, or reuse saved LU)
    return (-lam[rows] * x[cols], lam)       # (vals̄ on the pattern, b̄)

solve_pattern.defvjp(fwd, bwd)
```

Batching over 101 frequencies: write it for one frequency and `vmap` it, or write the batched version directly.
`rows`/`cols` must be **static** (closure or `nondiff_argnums`), not differentiated.

### 5.2 The noise solve reuses the same primitive
With the plan's identity (WS1): `w = A⁻ᴴ c`, `S = Re(wᴴ Q w)`, `noise = sqrt(K·S)`.
`A⁻ᴴ c = conj( A⁻ᵀ conj(c) )`, and `A⁻ᵀ` is just `solve_pattern` on the **transposed pattern** (swap
`rows`/`cols`). So one primitive covers both solves; JAX differentiates the `conj`, the quadratic form and
the `sqrt` automatically (these are cheap, O(nnz)).

### 5.3 Everything else stays as ordinary JAX autodiff
- **Assembly** (component formulas → `vals`): small elementwise functions on ~2,537 entries. Let `jax.vjp` handle them,
  ideally on WS3's per-type assembly rather than the 19-way switch.
- **Carrier → signal coupling:** the signal matrix's connector entries are scaled by the carrier solution
  (`simulate.py:~313-319`). Chain rule through `solve_pattern` for the carrier (286 system) handles it.
- **Outputs:** `carrier`, `signal` = x for all frequencies, and `noise`, each receiving a cotangent from the readout.

### 5.4 Save the LU, or factorise again in the backward?
| Option | Residual memory (101 freqs) | Extra backward time |
|---|---|---|
| Save dense LU | 101 × 751² × 16 B ≈ **0.9 GiB** | none |
| Refactor dense in backward | ~0 | ~36 ms × 101 ≈ **3.6 s** (EPYC) |
| Refactor **sparse** (WS1's solver) | ~0 (2,537 × 101 values) | ~0.6 ms × 101 ≈ **0.06 s** |

The plan's "about 1 GiB / under 1 s" assumes WS1's sparse solver. **Before WS1 lands, prototype with saved dense LU
factors** (`jax.scipy.linalg.lu_factor` in fwd, `lu_solve(..., trans=1)` in bwd). That already removes the dense
`Ā` and most of the tape. Swap in the sparse solver later behind the same interface.

## 6. Code map (what's in effect)

| Where | What | WS2 relevance |
|---|---|---|
| `benchmark.py:113-133` | `forward_with_pullback`, `readout`, `backward` | fixed contract: `jax.vjp(simulate)`, residuals are jit outputs |
| `differometor/simulate.py:35-62` `solve` | `jnp.linalg.solve` after a cast to complex128 | replace with `solve_pattern` (custom_vjp) |
| `simulate.py:64-110` `update` | `vmap(lax.switch)` assembly, dense scatter | WS3 rewrites; WS2 needs its output as `vals (F, nnz)` |
| `simulate.py:268-412` `simulate` | linking → carrier solve → signal assembly → conj blocks → signal solve → noise (2 more solves) | **all of it is differentiated today**; WS2 owns the tail (solves + noise) |
| `simulate.py:469-479` `run_build_step` | zips `simulate`'s argument names with build outputs | change `simulate`'s signature here too (plan §3) |
| `jax/_src/lax/linalg.py:2673` `_solve` | JAX's built-in solve adjoint (`custom_linear_solve`, `trans=1`) | the reference behaviour your `custom_vjp` must match |

## 7. Other open-source projects to learn from

| Project | Why it's relevant | Look at |
|---|---|---|
| **Ceviche** (`fancompute/ceviche`) | Frequency-domain EM solver: sparse `A(ε) x = b`, hand-written adjoint for the **sparse solve, with gradient only on the non-zeros** (`-adj[rows]*x[cols]`). The closest match to WS2 | `ceviche/primitives.py` (sparse solve / sparse multiply VJPs) |
| **klujax** (`flaport/klujax`) + **SAX** (`flaport/sax`) | SAX simulates photonic *circuits* in JAX (frequency sweeps, like us). klujax wraps the **KLU sparse solver** as a JAX primitive with autodiff rules, complex128 and batching | klujax's VJP/transpose rules; SAX's "klu" backend. Not in our pinned env: use as reference, adding a dependency is risky for the scored run |
| **Finesse 3** (`gwoptics/finesse`, GitLab/GitHub) | The simulator Differometor was ported from (code comments cite its routines). Solves the same systems with **sparse KLU**, including `solve_noises`. No AD, but shows the sparse noise algorithm | `finesse/simulations/sparse/` |
| **Lineax** (`patrick-kidger/lineax`) | JAX linear solves with clean implicit-differentiation (adjoint) rules for any solver | how transpose / adjoint solves are wired |
| **JAX** itself | `custom_linear_solve`, `jax.experimental.sparse.linalg.spsolve` (sparse solve with a custom linear solve) | `jax/_src/lax/control_flow/solves.py`, `jax/_src/lax/linalg.py:_solve` |
| Adjoint method background | Photonic inverse design (Meep adjoint, Tidy3D adjoint) uses the same "one extra transpose solve" idea | concepts only |

## 8. Suggested path for WS2

1. **Understand by measuring:** `python scripts/bench.py parts --parts pipeline --freqs 4`. Watch
   `forward.compile`, `forward.run`, `gradient.compile`, `gradient.run` and peak RSS.
2. **Prototype `differometor/adjoint.py`:** `solve_pattern` (custom_vjp) with **saved dense LU**, used for the
   signal solve and for `w` (transposed pattern). It needs the pattern (`rows`, `cols`) and `vals (F, nnz)`. Until WS3 delivers
   `vals`, extract them from the dense matrix by a static gather `A[:, rows, cols]` (the pattern is identical at
   every frequency).
3. **Correctness:** `python scripts/bench.py check --fd` must PASS (all 193 gradients vs baseline).
4. **Then** swap the saved LU for refactoring (dense, then WS1's sparse), carrier solve included, and measure
   residual size: `jax.tree_util.tree_leaves(pullback)` total bytes should drop from ~9 GiB to ~MBs.

**Pitfalls:**
- Duplicate `(row, col)` in the pattern: the current `.at[].set` is last-write-wins, but a gather-based `Ā` would
  double-count. Check uniqueness.
- Pass `rows`/`cols` as static (closure or `nondiff_argnums`), never as differentiated inputs.
- `custom_vjp` returns one cotangent per input. Return zeros of the right shape and dtype, not `None`, if unsure.
- The baseline is already 5.4e-6 from the 1e-5 limit. Don't change the forward numerics in WS2.

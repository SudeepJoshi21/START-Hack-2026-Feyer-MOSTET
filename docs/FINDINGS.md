# Findings addendum + results log

**The plan is [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md)** (workstreams WS1–WS4, interfaces,
process). This file only adds what that plan doesn't cover, corrects one point, and holds the shared
**results log** (§4). How to measure on a laptop: [`docs/BENCHMARKING.md`](BENCHMARKING.md).

Laptop numbers below: AMD Ryzen AI 7 350, WSL2, 7 GB RAM, `--cpus 1`, usually 10 of 101 frequencies.
Compare only ratios measured on one machine.

---

## 1. Additions to OPTIMIZATION_PLAN.md

### 1.1 Index *values* can be compile-time constants (WS3/WS4) — measured ~6 s of compile
OPTIMIZATION_PLAN §3 says index values are runtime data because `arrays` is a jit argument. They don't
have to be: jit treats a **registered pytree node's `aux_data` as static**. If `run_build_step` returns the
structure (pattern `rows/cols`, per-type groups, RCM permutation, scatter indices) inside a node like

```python
@jax.tree_util.register_pytree_node_class
class Static:
    def __init__(self, **arrays): self.d = {k: np.asarray(v) for k, v in arrays.items()}
    def tree_flatten(self): return (), self            # no leaves -> nothing traced
    @classmethod
    def tree_unflatten(cls, aux, children): return aux
    def __hash__(self): return hash(tuple((k, v.dtype.str, v.shape, v.tobytes()) for k, v in sorted(self.d.items())))
    def __eq__(self, o): return isinstance(o, Static) and self.__hash__() == o.__hash__()
```

then `dict(arrays, optimized_parameters=p)` still passes it into `simulate(**...)`, but XLA sees
constant gathers/scatters. Measured on the unchanged code: forward+vjp compile **20.4 s with indices as
jit args vs 14.2 s as constants** (10 freqs, laptop). Keep *numeric* inputs (base parameters,
frequencies) as leaves so nothing physical is precomputed outside the timer.

### 1.2 Finite differences are unreliable for most tunings (WS2)
At the benchmark's steps (1e-3, 5e-4), FD for `ml11`, `mt23`, `m43` tunings does not converge (the two
steps disagree by 30–100%); only `mb11` is roughly stable, which is presumably why the benchmark checks
only that one. **Validate gradients against the baseline's AD gradient on all 193 entries**
(`scripts/bench.py check` does this), not against FD.

### 1.3 Timing noise is ~±20% per stage
Same code, two fresh runs: forward compile 43.2 s vs 36.1 s, first-run total 70.0 vs 64.1 s. Use
`bench.py parts --trials 3` and don't trust single-stage gains < ~1.2× without repeating.

### 1.4 Laptop vs EPYC vs Xeon stage split (unchanged code)

| Stage | Xeon 8468, 101 f (README) | EPYC 7763, 101 f (plan) | Ryzen laptop, 10 f |
|---|---:|---:|---:|
| imports + build | 3.6 | 1.9 | 3.2–3.8 |
| forward lower+compile | 17.0 | 27.1 | 38–49 |
| forward run | 13.3 | 22.5 | 5.2–5.6 |
| gradient lower+compile | 5.6 | 9.7 | 12–14 |
| gradient run | 4.3 | 4.8 | 1.3–1.7 |
| **total** | **44.2** | **66.8** | **64–77** |

The laptop compiles 2–3× slower than the Xeon; the jury's machine is relatively *run*-heavy, so WS1/WS2
run-time wins matter at least as much as compile wins there.

### 1.5 Assembly-only compile, isolated (supports WS3)
Signal fill `vmap(update)` alone: compile 4.1–4.9 s, its VJP 8.4–10.5 s; dense `vmap(solve)` + VJP:
~1.3 s total. Benchmark-style forward (returning residuals) vs `simulate` under `vjp` with residuals
dropped: 36–49 s vs 20 s compile, so the **residual outputs themselves** roughly double forward cost
(supports WS2's custom_vjp keeping residuals small).

### 1.6 Comparison with the original repo (artificial-scientist-lab/Differometor)
- **Same code.** Upstream `main` *is* the challenge's pinned commit `b0dc7c7`. `differometor/` and `examples/` are
  byte-identical, and there are no newer commits or branches with speedups to borrow.
- **Designed for GPU.** Upstream's speed claims (README) are for **GPU** and for vmapped *batches* of parameter sets
  (`run_with_parameter_sets`). Dense batched solves are the right design there, and the wrong one for one CPU core.
- **Finesse used sparse KLU.** The code is ported from Finesse (comments cite the matching `finesse > … .pyx` routine), and an older
  `simulate.py` (before `7d246c5`) cites `finesse > simulations > sparse > KLU.pyx > solve_noises()`, so the
  reference implementation solves these systems with **sparse KLU**. That supports WS1. KLU first permutes to
  **block triangular form (BTF)**. Measured on our matrices (strongly connected components):
  - signal 751×751 → **one coupled 612×612 block + 139 trivial 1×1 blocks** (12 weakly connected pieces)
  - carrier 286×286 → one 240×240 block + 46 1×1 blocks
  A BTF permutation (static, from the pattern) cuts the block that needs real factorisation from 751 to
  612 (≈0.54× the dense flops). The 1×1 blocks are just back-substitution. It combines with any ordering inside the
  612 block.
- **Author's TODO in old `simulate.py`:** *"ideally we would only do this [noise-matrix] update if it is necessary
  … Otherwise we could just use jnp.tile"*. The original authors also saw the noise matrix as frequency-invariant work (WS3).
- **History hints the numerics are delicate:** an old comment on `solve` refers to a `linalg_solve_debug.ipynb` that
  investigated precision errors (matches the plan's cond(A) ≈ 1e30).

### 1.7 XLA compile-effort settings (measured; needs organiser approval before use)
Same code (`ws2-adjoint`), laptop, 1 CPU, 4 freqs, `scripts/profile_stages.py`, 2 runs each:

| Setting | forward compile | gradient compile | forward run | first-run total |
|---|---:|---:|---:|---:|
| default | 40.4 / 39.4 s | 11.7 / 11.6 s | 0.95 / 1.44 s | 61.8 / 61.1 s |
| `XLA_FLAGS=--xla_backend_optimization_level=0` | 21.6 / 22.3 s | 7.1 / 7.1 s | 1.20 / 1.00 s | **39.2 / 39.2 s (1.56×)** |
| `JAX_EXEC_TIME_OPTIMIZATION_EFFORT=-1` (1 run) | 38.3 s | 12.0 s | 0.98 s | 60.5 s |

- `max |Δln S|` unchanged (2.94e-6), so the numerics are not affected here.
- Backend optimisation level 0 roughly **halves compile** (LLVM does less work). The LU itself is a LAPACK call and is
  unaffected, but elementwise code may run slower. **Run-time cost at 101 freqs is not measured yet.**
- It would have to be set from inside `differometor` before the CPU backend initialises (the benchmark sets
  no flags, and its report records `XLA_FLAGS`). That is the plan's open question: **ask Feyer before relying on it.**
- `exec_time_optimization_effort` alone does nothing measurable.

## 2. Correction to my earlier notes
- Noise matrix `Q`: **66 component-function calls producing 272 non-zeros** (the plan's number is right;
  I had written "66 non-zeros").
- Plotly / `import differometor` is negligible (~15–80 ms after jax); lazy-importing `plot` is not worth
  much on its own.

## 3. Baseline correctness margin
Unmodified code: max `|Δln S|` = 5.37e-6 (Xeon and EPYC, 101 f), 3.8e-6 (laptop, 10 f), against the 1e-5
limit. `bench.py check` fails a candidate whose sensitivity drifts > 5e-6 (in ln) from the *baseline*.

## 4. Results log (append one row per measured change)

| Date | Branch / commit | Change | Machine, freqs | First-run total | Compile | Warm median | Check | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| 2026-11-16 | upstream | baseline (organisers) | Xeon 8468, 101 | 44.20 | 22.92 | 17.68 | PASS | `benchmarks/cpu-baseline.json` |
| 2026-11-21 | upstream | baseline (plan §1) | EPYC 7763, 101 | 66.79 | 37.61 | ~24.97 | PASS | `OPTIMIZATION_PLAN.md` |
| 2026-11-21 | upstream vs itself | noise calibration (`bench.py parts --ref 02ec741`) | Ryzen laptop, 10 | 70.0 / 64.1 | 60.3 / 53.6 | 4.58 / 4.46 | PASS (0 diff) | same code, ~9% apart |
| 2026-11-21 | `ws1-solver` | WS1(a): one LU of A per frequency reused for the A and Aᴴ solves; noise via Re(wᴴQw), third solve removed | EPYC 7763, 101 | 60.90 | 41.23 | 16.61 | PASS (warn: noise 1.6e-8, grad 2.7e-6 rel) | forward run 23.5 → 11.8 s; same-session baseline first run 67.50; compile within ±20% noise |
| 2026-11-21 | `ws1-solver` | WS1(b): signal solves on the 2597-entry sparsity pattern with SciPy SuperLU (`pure_callback` inside `custom_linear_solve`); noise matrix built once instead of per frequency | EPYC 7763, 101 | 44.90 | 34.05 | 9.07 | PASS (warn: signal 6.3e-8, grad 1.6e-6 rel) | forward run 11.8 → 5.0 s; gradient run unchanged (~4.1 s, dense cotangents → WS2); aLIGO sparse vs original ≤ 1.4e-9 rel, incl. a squeezer-db sweep |
| 2026-11-21 | `ws1` @ `095df6e` | WS1(a)+(b) re-validated: `bench.py check --freqs 101` + `bench.py parts --parts pipeline --ref 02ec741 --trials 3 --repeats 1 --freqs 101` | EPYC 7763, 101 | 44.68 vs 64.90 (**1.45×**) | 33.81 vs 38.83 (1.15×) | 8.96 vs 25.03 (**2.79×**) | PASS (warn: signal 6.3e-8, noise 1.2e-8, grad 1.6e-6 rel, cosine 1.0) | forward run 19.49 → 4.95 s (3.94×); gradient run 4.60 → 4.09 s; peak RSS 14.7 → 9.5 GB; trials 42.0/44.7/45.7 vs 64.9/65.9/64.2. Merge plan with WS2: `docs/WS1_WS2_INTEGRATION.md` |

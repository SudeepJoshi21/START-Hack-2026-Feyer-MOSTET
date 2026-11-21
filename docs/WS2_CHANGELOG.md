# WS2 change log: gradient / AD / memory

Branch `ws2-adjoint`. Every change to the code is listed here with what, why, where, and how it was verified.
Background: [`docs/WS2_GRADIENT_EXPLAINED.md`](WS2_GRADIENT_EXPLAINED.md). Plan: [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md) WS2.

---

## Step 1 (2026-11-21): adjoint solve with one LU, sparse noise, custom gradient

### Result (laptop, Ryzen AI 7 350, 1 CPU, 10 of 101 frequencies, 2 trials, vs unmodified `02ec741`)

| Metric | Baseline | WS2 step 1 | Change |
|---|---:|---:|---:|
| forward run (first) | 4.57 s | 2.20 s | **2.08× faster** |
| gradient run | 1.58 s | 1.26 s | 1.25× faster |
| warm forward+sensitivity+gradient | 4.41 s | 2.33 s | **1.89× faster** |
| compile total | 57.4 s | 55.8 s | ~same (within noise; compile cost is assembly = WS3) |
| scored-equivalent first-run total | 67.0 s | 62.7 s | 1.07× |
| peak RSS | 2.58 GB | 1.98 GB | 1.31× less |
| AD tape (pullback) | 817 arrays, 910 MiB | 814 arrays, 316 MiB | **2.9× smaller** (≈9.0 → ≈3.1 GiB at 101 freqs) |

At 101 frequencies, run time is a larger share of the score (Xeon: forward run 13.3 s + gradient run 4.3 s),
so the run-time gain should matter more on the full benchmark. **Not yet measured at 101 frequencies or
on the grid.**

Correctness (`python scripts/bench.py check --fd`, 10 freqs, vs unmodified code): **PASS**
- signal, carrier, powers: **bit-identical** (0 difference)
- noise: 1.5e-8 relative (expected: the third solve is replaced by the `Re(wᴴQw)` identity; plan measured 2e-8)
- max `|Δln S|` vs public reference: 3.762e-6 (baseline 3.759e-6; limit 1e-5)
- all 193 gradients: max relative error 9.7e-6 (worst `m01.reflectivity`), cosine similarity 1.000000000000
- finite-difference check on `mb11.tuning` (the benchmark's own check) agrees
- aLIGO (a different setup, `examples/data/aligo_test_data.json`) through the new path: max `|Δln S|` 3.8e-11 (original code 7.7e-11)
- `tests/test_adjoint.py`: 4/4 pass (forward and gradients equal JAX autodiff of the original 3-solve formulation to < 1e-10)

### What changed, file by file

**1. New file `differometor/adjoint.py`**
- `signal_pattern(signal_matrix, signal_matrix_indices)`: NumPy, at build time. The positions that can be non-zero in
  the signal matrix A: base-matrix non-zeros ∪ positions written by `update`. `update` writes the right-hand side
  through column index **−1**; those indices are normalised (`mod (n+1)`) and excluded from A. The result is sorted and unique,
  so a scatter-set equals a scatter-add. For this fixture: **2,597** positions (same set as WS1's `_structure_arrays`), and all 2,537 actual non-zeros lie inside. *(Corrected: an earlier version of this log said 2,668. That count came from a one-off script that did not normalise the −1 right-hand-side column, not from the code; WS1 spotted it.)*
- `pattern_arrays(simulation_arrays)`: returns `{"signal_pattern_rows", "signal_pattern_cols"}`, or `{}` (which
  keeps the original path) if a precondition fails: the noise matrix is not N×N, the noise update indices are negative, or the noise update positions are
  not unique.
- `solve_signal_and_noise_weights(vals, rows, cols, b, C)`: `jax.custom_vjp`, one frequency (vmapped by the caller).
  - forward: densify A from `vals`, **one** `lu_factor`, then `x = A⁻¹b` (`trans=0`) and `W = A⁻ᴴC` (`trans=2`).
  - tape: LU factors, pivots, `x`, `W`, pattern.
  - backward: `λx = A⁻ᵀx̄` (`trans=1`), `λu = A⁻¹ conj(W̄)` (`trans=0`), with `u = conj(W)`:
    `vals̄ = −λx[rows]·x[cols] − Σ_d u_d[rows]·λu_d[cols]`, `b̄ = λx`, `C̄ = conj(λu)`. Pattern indices get no cotangent (`None`).
- `quadratic_noise(W, q_vals, q_rows, q_cols, q_base)`: `S_d = Re(W_dᴴ Q W_d)` with `Q·w` as a `segment_sum`
  over the 272 noise entries, plus a dense `q_base` term for setups whose base noise matrix is not zero
  (all zeros here). This replaces the dense (F,751,751) noise matrix, its matrix-vector product and the third solve.
- `ENABLED = True`: module switch. Set `differometor.adjoint.ENABLED = False` to force the original path for A/B tests.

**2. `differometor/simulate.py`**
- `import differometor.adjoint`.
- `update()` split: new function `component_outputs(parameters, function_input_indices, output_indices,
  function_indices)` returns the values `update` writes. `update` now calls it and scatters, so **behaviour is unchanged**.
  The adjoint noise path uses the 272 values directly.
- `simulate()` signature: two **keyword-only** arguments after `*`: `signal_pattern_rows=None`,
  `signal_pattern_cols=None`. Keyword-only, so `run_build_step`'s positional zip (`co_argcount`) is unaffected.
- `simulate()` body:
  - `use_adjoint = signal_pattern_rows is not None and adjoint.ENABLED`.
  - If `use_adjoint`, the old signal solve is skipped. After the carrier selections are built, a new branch:
    gather `values = signal_matrix[:, rows, cols]` → vmapped `solve_signal_and_noise_weights` → `noise_values` via
    `component_outputs` → `noise_base = noise_matrix` with updated positions zeroed (exactly reproduces `.at[].set`) →
    `quadratic_noise` → the same `sqrt(2·S·UNIT_VACUUM·H_PLANCK·F0·0.25)` → `return carrier.T, signal.T, noise.squeeze()`.
  - Otherwise the original code runs **unchanged** (3 dense solves, dense noise matrix).
- `run_build_step()`: after building `simulation_arrays`, adds `adjoint.pattern_arrays(simulation_arrays)` (2 int32
  arrays, or nothing). All callers pass the dict as `**arrays`, so extra keys are safe (checked: `run`, `run_setups`,
  `run_simulation_step`, `examples/*_optimization.py`).

**3. New file `tests/test_adjoint.py`**: run with `python tests/test_adjoint.py` (or pytest).
- forward equals the original formulation (dense solves, `A⁻ᴴ` solve, third covariance solve)
- gradients w.r.t. `vals`, `b`, `C`, `q_vals` equal JAX autodiff of the original, 3 random seeds, 2 detectors, non-trivial
  real loss using `abs`, `log`, `real`, `imag` (relative error < 1e-10)
- batched over frequencies under `jit` + `jax.vjp` with the pullback returned and called in a second `jit` (benchmark style)
- `signal_pattern` handles the −1 right-hand-side column and duplicates

### What remains in the tape (316 MiB at 10 freqs) and who removes it
| Item | Size | Removed by |
|---|---:|---|
| LU factors (10,751,751) | 86 MiB | refactor in the backward pass once WS1's sparse factorisation exists (dense refactor would cost ~3.6 s) |
| 2 × dense signal matrix (10,751,752) + bool mask | 178 MiB | WS3: assemble directly on the pattern (carrier scaling and conj blocks as vector ops on `vals`) |
| 166 × (10,679) etc. | ~50 MiB | WS3: per-type assembly instead of the 19-way `switch` |

### Interfaces for the other workstreams
- **WS3 (assembly):** if assembly produces `values (F, nnz)` aligned to `signal_pattern_rows/cols` and
  `b (F, N)`, `simulate` can skip the dense matrix and call `solve_signal_and_noise_weights` directly.
- **WS1 (solver):** replace the inside of `_forward`/`_bwd` (`lu_factor` / `lu_solve` with `trans` 0, 1, 2) with the
  sparse factorisation. The maths and the custom_vjp contract stay the same. Note that this branch already contains WS1's
  "quick win (a)" (one LU, `trans=2`, `Re(wᴴQw)`), so coordinate to avoid doing it twice.

### How to reproduce
```bash
python tests/test_adjoint.py
python scripts/bench.py check --fd
python scripts/bench.py parts --parts pipeline,forward --trials 2 --ref 02ec741
```

---

## Step 2 (2026-11-21): integration with WS1 (branch `integration`)

`integration` = `origin/main` + merge `ws2-adjoint` + merge `ws1` (its `simulate.py` resolved to the WS2 version,
following `docs/WS1_WS2_INTEGRATION.md` §2) + this step: WS1's sparse solver moved under WS2's custom gradient.

### What changed, file by file

**`differometor/adjoint.py`** (rewritten around the same maths)
- `solve_signal_and_noise_weights` is now **batched over frequencies** (`vals (F,nnz)`, `b (F,n)`, `C (F,D,n)`), not vmapped.
  One host call per direction, instead of 101.
- `SOLVER = "sparse"` (default): WS1's SciPy SuperLU kernel (`_csc_structure` from WS1, COLAMD ordering) via
  `jax.pure_callback(..., vmap_method="sequential")`.
  - `_host_forward`: per frequency, one `splu`, then `x = solve(b, "N")` and `W = solve(Cᵀ, "H")` for all detectors at once.
  - `_host_adjoint`: refactorises (~0.6–1.5 ms), then `λx = solve(x̄, "T")` and `λu = solve(conj(W̄)ᵀ, "N")`.
  - **The tape keeps only `vals`, `x`, `W`; no LU factors** (WS1 guide §3 step 3).
- `SOLVER = "dense"`: the step-1 XLA LU path (factors on the tape), kept for A/B tests and as a fallback.
- `_bwd`: the same formula as step 1, written for the batch.
- `signal_pattern`: also wraps negative **row** indices (`rows mod n`) before `np.unique` (WS1 guide §5). Test added.
- `pattern_arrays` returns three more keys:
  - `noise_base_rows/cols`: non-zeros of the base noise matrix that `update` doesn't overwrite. This replaces the dense
    `q_base` matrix and its 751² product (WS1 guide §3 step 7; K = 0 for the fixture).
  - `noise_parameter_rows`: `[0]` when no swept parameter (carrier or signal sweep indices) is an input of a noise
    function, so the 66 noise functions run **once** instead of 101 times (WS1's idea, guide §3 step 5). Otherwise all rows.
    aLIGO's frequency sweep and a squeezer-dB sweep both feed noise functions, and the check correctly falls back.
- `quadratic_noise(W, q_vals, q_rows, q_cols)`: the base term moved into the sparse lists.

**`differometor/simulate.py`**
- Three more keyword-only arguments: `noise_base_rows`, `noise_base_cols`, `noise_parameter_rows` (default `None`).
- Adjoint branch: one batched call (`carrier_selections` swapped to `(F, D, N)`). The noise values are computed on
  `parameters[noise_parameter_rows]` and broadcast to F, concatenated with the base non-zeros, then `quadratic_noise`.
- The original path (`adjoint.ENABLED = False`, or setups failing the preconditions) is unchanged.
- WS1's positional `signal_pattern` / `noise_parameter_rows` arguments and its `sparse_solve` /
  `custom_linear_solve` helpers are **not** carried over (they would duplicate this path).

**`tests/test_adjoint.py`**: now 5 tests, each run for **both** solvers:
- forward results vs the original 3-solve formulation (< 1e-11)
- gradients vs JAX autodiff of the original (< 1e-10, 3 seeds)
- sparse == dense (< 1e-11 outputs, < 1e-10 gradients)
- under `jit` + `jax.vjp` with the pullback returned and called in a second `jit`, plus a tape-size assertion (no F·n² arrays)
- pattern wraps negative rows and columns

### Validation (laptop, Ryzen AI 7 350, 1 CPU)
| Check | Result |
|---|---|
| `tests/test_adjoint.py` | 5/5 PASS (sparse and dense) |
| `bench.py check --fd` (fixture, 10 freqs, all 193 grads) | **PASS**: `|Δln S|` vs reference 3.761e-6 (baseline 3.759e-6); signal 2.4e-8, noise 1.0e-8; gradients max 2.0e-5 relative (`boundary01.power`), cosine 1.0; carrier and powers bit-identical; `mb11.tuning` FD agrees |
| aLIGO frequency sweep vs Finesse-verified data | max `|Δln S|` 2.3e-10 |
| aLIGO squeezer-dB sweep, adjoint vs original path | 1.6e-10 |
| aLIGO, 25-point log sweep, adjoint vs original | signal 5.7e-11, noise 1.0e-11 |
| **Voyager**, 25-point log sweep, adjoint vs original | dense solver: 1.7e-13. **Sparse solver: noise up to 2.1e-3 (median 7.7e-6)**, see the known issue below |

Timing (10 freqs, 2 trials each):

| | Original `02ec741` | `ws1` only | **integration** |
|---|---:|---:|---:|
| forward run (first) | 5.05 s | 1.54 s | 1.14–1.41 s |
| warm forward+sensitivity+gradient | 4.23 s | 1.91 s | 1.59–2.05 s |
| first-run total | 69.8 s | 61.0 s | 62.7–64.4 s (compile-bound; same as ws1 within noise) |
| peak RSS | 2.57 GB | 2.02 GB | **1.82 GB** |

Warm medians, sparse vs dense solver on this branch (4 runs each):

| freqs | sparse fwd / grad | dense fwd / grad |
|---|---|---|
| 4 | 0.30 / 0.31 s | 1.09 / 0.44 s |
| 10 | 0.67–0.71 / 0.85–1.06 s | 1.55–1.83 / 0.64–0.66 s |
| 20 | **1.56 / 1.41 s** | 4.74 / 2.22 s |

At 10 freqs the sparse *gradient* is reproducibly slower than dense, although its host callbacks take only 12 ms. At 4 and 20
freqs sparse wins both passes, so it stays the default. **Unexplained; worth one profile on the cluster.**

Peak memory: 1.82 GB at 10 freqs, 3.13 GB at 30, so about **7.8 GB at 101**. Too much for the 7 GB laptop; run the official
`benchmark.py` on the cluster.

### Known issue: sparse vs dense LU on Voyager (not scored)
At 287 Hz the noise weight vector `w` spans ~21 orders of magnitude (max |w| ≈ 3.5e13, tiny at the noise-source rows).
Any LU solve has an absolute error of ~ε·max|w| ≈ 5e-3 there. Measured:
- `|Δw|` at the noise rows ≈ 5e-3 between the sparse and dense solves, while `W` agrees to 3e-14 *relative to its max*;
- so noise differs by 2e-3 and `cond(A)` ≈ 9e31;
- iterative refinement and other orderings (NATURAL, MMD_AT_PLUS_A) do not change it.

The dense path matches the original only because it uses the same LAPACK algorithm and rounding, not because it is more accurate.
**Neither result is verified to better than ~1e-3 for that setup.** The scored fixture is far better conditioned at the
relevant entries (sparse vs original 2e-8, reference check passes with 2.6× margin).
Options if it matters: `adjoint.SOLVER = "dense"` for such setups, or a precision study (e.g. extended-precision residuals).

---

## Step 3 (2026-11-21, on `main`): fix "noise matrix computed once" never triggering

**Bug (mine, from step 2's port):** `adjoint.pattern_arrays` decides whether the 66 noise-source functions can be evaluated once
(`noise_parameter_rows = [0]`) by intersecting the *swept* parameter indices with the noise functions' input indices. Two
details made the intersection always non-empty:
- Without a carrier sweep, `prepare_arrays` emits a **dummy one-value carrier "sweep" of parameter index 0**.
- Unused noise-input slots are **padded with index 0**.

So on `main`, Q was rebuilt for every frequency (`noise_parameter_rows` had F rows). WS1's original check used the swept
*names* and did not have this problem. **Correctness was never affected** (per-row is always right); only the claimed saving was missing.

**Fix:** only sweeps with more than one value count as swept. A real sweep of parameter 0 still falls back (conservative).

**Why Q is frequency-independent (physics):** Q holds the noise *sources*: vacuum (constant ½), loss-induced vacuum (∝ mirror
loss), squeezed vacuum (squeeze dB, angle). Quantum vacuum noise is white in the sideband picture. The frequency dependence of
the detector noise comes from the optical response A(f) through `w(f) = A(f)⁻ᴴ c`, which is still solved for every frequency.
Q changes per row only when a sweep changes a noise-source parameter. The fallback handles that.

| Check | Result |
|---|---|
| fixture `noise_parameter_rows` | (10,) → **(1,)** |
| aLIGO frequency sweep | (F,) → **(1,)** (frequency does not feed its noise either); noise vs original 3.5e-11 |
| aLIGO squeezer-dB sweep | stays per-row **(7,)** (correct fallback); noise vs original 4.5e-10 |
| `tests/test_adjoint.py`, `tests/test_assembly.py` | 5/5, ALL PASS |
| `bench.py check` | PASS, unchanged (`|Δln S|` 3.761e-6, gradients 2.0e-5, cosine 1.0) |
| `benchmark.py --cpus 1`, interleaved A/B on a busy laptop | fixed 31.6 / 33.8 s vs before 33.3 / 35.4 s (fixed faster both times; absolute values above the earlier 28.8 s because of machine load) |

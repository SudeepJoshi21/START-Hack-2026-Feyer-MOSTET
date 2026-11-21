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
  so a scatter-set equals a scatter-add. For this fixture: 2,668 positions, and all 2,537 actual non-zeros lie inside.
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

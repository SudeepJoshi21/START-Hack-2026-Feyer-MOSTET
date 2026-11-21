# Optimization plan: findings and 4-way work split

Goal: lowest **complete first-run time** of `python benchmark.py --cpus 1` (imports, setup, all
compilation, forward, sensitivity, gradient, transfer) with the same physics: 101 frequencies, 193
gradients, float64, `|log(sens) - log(ref)| <= 1e-5`.

For the physics and the algorithm, read [SIMULATOR_EXPLAINED.md](SIMULATOR_EXPLAINED.md) first. It states
the noise identity and the `vmap(lax.switch)` cost as hypotheses. The profiling below **measured** them.

All numbers here were measured on the HCTLRDS cluster: **AMD EPYC 7763, pinned to 1 CPU**, JAX 0.9.0,
SciPy 1.18.1, `benchmark.py --cpus 1 --repeats 3`. The README baseline (Xeon 8468, 44.2 s) is a different
machine, so only compare numbers taken on the same hardware.

## TL;DR

- 56% of the first run is **compilation** (37.6 of 66.8 s). 80% of the forward compile is the
  elementwise matrix-assembly graph, not LAPACK.
- The signal matrix is **751x751 but only 0.45% non-zero** (2537 entries, same pattern at all 101
  frequencies). The code stores and factorizes it **dense**, 2-3 times per forward pass.
- One of those solves is mathematically redundant, and a sparse LU is about **60x faster** than the dense one.
- Plan: a sparse-native pipeline (pattern-based assembly, sparse solve, hand-written adjoint). Target
  first run is roughly **7-10 s** (an estimate, not a measurement).

## 1. Baseline (66.79 s total)

| Phase | Seconds | Share |
|---|---:|---:|
| imports 0.48, init 0.01, load 0.01, build 1.37 | 1.86 | 3% |
| trace/lower + compile: forward 2.55 + 24.53, sensitivity 0.13 + 0.75, gradient 0.69 + 8.97 | 37.61 | 56% |
| forward run (includes AD residuals) | 22.52 (warm median about 19.5) | 34% |
| gradient run | 4.80 | 7% |
| sensitivity run, host transfer | about 0 | 0% |

Warm repeat totals: 27.75, 24.97, 22.54 s.

## 2. Measured findings

1. **Matrices are tiny-sparse but stored dense.**
   - Signal system: 751 unknowns (2x286 + 179), 101 frequencies.
   - `nnz(A) = 2537` (0.45%), at most 17 per row, identical pattern at every frequency.
     Only 1006 of the entries vary with frequency.
   - Right-hand side `b`: 71 non-zeros, constant in frequency.
   - Noise matrix `Q`: 272 non-zeros (8 off-diagonal), constant in frequency.
   - Detector selection vector: 2 non-zeros.
   - Carrier system: 286 unknowns, a single matrix, 680 non-zeros.
   - One dense `(101, 751, 752)` complex128 array is 0.91 GB. The README reports 14.4 GiB peak memory.
2. **Dense LU is about 12-15 s of the forward run.** A dense 751x751 complex LU takes 36 ms, so 3.6 s per
   batch of 101. In the forward+vjp program, replacing every solve with a matmul drops the run from
   22.5 s to 7.1 s.
3. **One solve is redundant.** The code computes `w = A^-H c`, `z = A^-1 Q w`, `S = Re(c^H z)`. Since
   `c^H A^-1 = w^H`, `S = Re(w^H Q w)`.
   - Verified: max relative difference to the current `noise` is **2.0e-8**.
   - The max log-sensitivity error vs the public reference is unchanged (5.374e-6).
   - The `A^H` solve can also reuse the LU of `A`: `jax.scipy.linalg.lu_solve(..., trans=2)`
     exists in JAX 0.9.0.
4. **Sparse LU is about 60x faster.**
   - SciPy SuperLU (`splu`, COLAMD): factor 0.61 ms, solve 0.07 ms, `nnz(L+U) = 8089`.
   - It agrees with the JAX dense solution to <= 4.1e-8 relative (`A^-1 b`) and <= 1.1e-8 (`A^-H c`),
     checked at frequencies 0, 50 and 100.
   - Reverse Cuthill-McKee reordering gives bandwidth 49 (natural order: 542 below, 691 above).
     SuperLU on the permuted matrix with natural column order: 0.48 ms.
5. **Compile time is the assembly graph, not LAPACK.**
   - Forward+vjp is 7481 HLO ops and compiles in 24.5 s. With every solve replaced by a matmul it still
     compiles in 19.6 s. `vmap(solve)` plus its vjp alone compiles in 0.55 s.
   - Cause: `update()` uses `vmap(lax.switch(i, FUNCTIONS, x))` with a **batched index**. JAX then
     evaluates all 19 branches for every call and picks with `select_n`
     (`jax/_src/lax/control_flow/conditionals.py:450-497`). Add dense scatter/gather and the three
     conj-block updates, plus their transposes.
   - The number of calls per function type is known at build time:

     | System | Calls K | Per type |
     |---|---:|---|
     | carrier | 77 | surface 54, space 16, laser 7 |
     | signal | 679 | space 71, space_lower 71, space_modulation 71, space_modulation_lower 71, signal_function 71, surface 54, susceptibility 54, force_left 54, force_right 54, optomech_left 54, optomech_right 54 |
     | noise | 66 | loss 54, vacuum 8, squeezer 4 |
6. **Autodiff roughly doubles the cost.** Forward without AD: 12.7 s run, 10.1 s compile (5147 ops).
   With AD: 22.5 s run, 24.5 s compile. The gradient adds 4.8 s run and 9.0 s compile, because the
   cotangents are dense 0.91 GB arrays.
7. **Setup is 1.86 s.** Imports 0.48 s (jax is about 0.3 s of that; `import differometor` itself is
   about 15 ms after jax). `build()` 0.84 s, `pairs_to_arrays` 0.10 s, `prepare_arrays` 0.18 s, the rest
   is eager `jnp` calls (for example `space(jnp.array(...))` once per edge).
8. **Numerically delicate.** `cond(A)` is about 1e30 at frequencies 0, 50 and 100 (2.8e30, 2.5e30,
   1.2e30). The system is badly scaled rather than singular, since dense and sparse solutions still agree
   to 4e-8. The baseline error vs the reference is already 5.37e-6 against the 1e-5 limit. No float32 and no
   iterative solvers.
9. **Callbacks are cheap to compile.** A `jit(pure_callback)` on a tiny graph: lower 9 ms, compile 5 ms,
   first call 0.4 ms. In JAX 0.9.0 `pure_callback` needs an explicit `vmap_method` and a `custom_vjp` for
   gradients. There is no numba, Cython, MKL or UMFPACK in the environment. SciPy SuperLU is available.

## 3. Harness constraints and guardrails

- **`arrays` is a jit argument** in `benchmark.py`. Index *values* are runtime data; only shapes and Python
  code are static. To group work by component type, emit one index array per type at build time (the
  shapes then become static).
- **Signature coupling.** `run_build_step` zips `simulate`'s argument names with the outputs of
  `build` and `pairs_to_arrays` (`differometor/simulate.py:469-479`). If you change `simulate`'s
  signature, change those three places together.
- **Keep the public contract:** `df.simulate(**arrays)` returns `(carrier (286,1), signal (751,101),
  noise (101,))`; `df.signal_detector`, `df.power_detector`, `df.run_build_step`,
  `constrain_inter_grid_cell_spaces`. The harness differentiates through all three outputs.
- **Do not edit** `benchmark.py` or the fixtures (they are SHA-checked). The compilation cache stays off.
  Do not ship precomputed or cached executables.
- **Keep other setups working** (`examples/aligo.py` vs `examples/data/aligo_test_data.json`, `cavity`,
  `voyager`). If the fast-path preconditions do not hold, fall back to the old dense path.
- **Gate for every merge:** `python benchmark.py --cpus 1 --repeats 3 --output benchmark-check.json`
  prints `PASS`. Also diff saved arrays against a baseline run made with `--save-arrays baseline.npz`.
  Suggested thresholds (not yet measured): noise and sensitivity relative 1e-7, gradient relative 1e-4.

## 4. Target architecture and interfaces

Agree on these in the first 30 minutes so the four workstreams can merge:

```
build (NumPy only)      -> static pattern rows, cols (nnz = 2537), per-type index groups, RCM permutation
assemble   [WS3]        -> vals: complex128 (F, nnz) aligned to the pattern, plus b, Q values
solve_pattern [WS1]     -> solve_pattern(vals, rows, cols, rhs, trans) -> x (F, n)
adjoint    [WS2]        -> custom_vjp around solve_pattern, cotangent only on the nnz pattern
noise      [WS1]        -> S = Re(w^H Q w), then the existing sqrt(2 * S * UNIT_VACUUM * h * F0 * 0.25)
```

Check that the pattern has no duplicate `(row, col)` positions. Today `.at[].set` is last-write-wins.

## 5. Workstreams

| WS | Owner | Files | Goal and first steps |
|---|---|---|---|
| **1. Solver + noise algebra** | _assign_ | `simulate.py::solve`, new `differometor/linsolve.py` | Forward solver cost 12-15 s down to under 0.5 s. (a) **Quick win, about 1 h:** one `lu_factor` per frequency, `lu_solve` with `trans=0` and `trans=2`, drop the third solve using `Re(w^H Q w)`. (b) SuperLU kernel via `jax.pure_callback`; fallback: RCM-banded LU in pure JAX (bandwidth 49). (c) Accuracy harness against the dense result; try row/column equilibration for the cond ~1e30 issue. |
| **2. Gradient / AD / memory** | _assign_ | new `differometor/adjoint.py`, tail of `simulate.py` | Gradient 4.8 s run + 9.0 s compile, plus the AD overhead in the forward, down to under 1 s of run time and about 1 GiB. `custom_vjp` for the pattern solve: `lam = A^-T xbar` (JAX uses the transpose, not the conjugate transpose, for holomorphic maps; verify with finite differences), `Abar[k] = -lam[row_k] * x[col_k]` on the nnz only. Refactor in the backward pass (about 0.06 s for 101 matrices) instead of carrying state. Compose with `jax.vjp` of the small assembly. Handle cotangents of carrier, signal and noise. |
| **3. Assembly + components** | _assign_ | `components.py`, `simulate.update` / `expand_parameters`, `build.py::pairs_to_arrays` | Remove the 7 s of dense-array traffic and most of the 19.6 s elementwise compile. Per-type groups instead of `vmap(lax.switch)`. Write results into `(F, nnz)` values. Evaluate the 1531 frequency-independent entries once and vectorize the 1006 frequency-dependent ones. Express the sideband conjugation and the carrier scaling (1004 entries) as masks and vector multiplies. Drop `standardize_output`'s list to `jnp.array` pattern. Unit test: densified new assembly equals the old dense matrix. |
| **4. Build, compile engineering, integration** | _assign_ | `build.py::build`, `differometor/__init__.py`, benchmark gates | Setup 1.9 s down to about 0.5 s: NumPy-only build with no eager `jnp`, and emit the pattern, groups and permutation for WS1-3. Lazy-import `plot`. Test XLA flags that do not change numerics (no fast-math) for compile time; ask the organisers if a flag is borderline. Own the `integration` branch, the benchmark gate, the before/after JSON and plots, and the short approach note for the submission. |

Quick wins that ship independently of each other: **WS1 (a)** and **WS3 per-type grouping**.

Estimated targets (estimates, not measurements):

| Path | First-run total |
|---|---:|
| Today (EPYC, 1 CPU) | 66.8 s |
| JAX-native sparse pipeline (WS1-3 done) | about 7-10 s |
| Plan B below | about 3-5 s |

## 6. Process

- Branches `ws1-solver`, `ws2-adjoint`, `ws3-assembly`, `ws4-build`. Merge into `integration` about every
  2 hours; only merge `integration` to `main` after the gate passes.
- Record every measurement with the CPU model and `--cpus 1`, and keep the JSON and PNG of each run.

**Plan B (special prize), for whoever finishes first.** Wrap the whole `simulate` (assembly, solve and
adjoint) as one `custom_vjp` over NumPy/SciPy callbacks. The XLA compile then costs milliseconds, and
the total is roughly imports 0.5 s + build 0.3 s + compute 1-2 s.

## Open questions for the organisers

- Is a numerics-neutral XLA flag set from inside `differometor/` acceptable for the scored run?
- Are callback-based solvers (SciPy SuperLU) acceptable under "make Differometor faster"? We assume yes,
  because every frequency, gradient and the double precision are kept and nothing is precomputed.

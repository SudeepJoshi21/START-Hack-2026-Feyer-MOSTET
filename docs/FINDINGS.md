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

### 1.8 What the compiled programs consist of (merged `main`, 101 freqs, laptop, 1 CPU)
`benchmark.py` times `jax.jit(f).lower(...)` (trace Python → jaxpr → StableHLO) and `.compile()` (XLA passes: simplification,
fusion, layout, then LLVM codegen) for 3 programs. On 1 CPU XLA compiles single-threaded, and the persistent cache is off.

| Program | lower | compile | StableHLO ops in | optimised HLO out | fusions | custom calls |
|---|---:|---:|---:|---:|---:|---:|
| forward (`jax.vjp(simulate)`, returns tape) | 1.8 s | 19.9 s | 3,312 | 14,642 | 551 | 4 (LAPACK carrier solve, SuperLU callback) |
| sensitivity | 0.1 s | 0.8 s | 56 | 208 | 13 | 0 |
| gradient (backward) | 0.5 s | 5.3 s | 1,539 | 2,650 | 101 | 3 |

Forward-program ops by source function (StableHLO, innermost `differometor` frame, from `lower().as_text(debug_info=True)`):
`incidence_angle_right` **889 (26.8 %)**, `incidence_angle_left` **413 (12.5 %)**, `standardize_output` 203 (6.1 %),
`space_modulation` 190, `simulate` 173, `assembly.component_outputs` 137, `phase_shift` 100, `expand_parameters` 97,
`surface` 88, `solve` (carrier) 82. Top lines: `components.py:1184` (`arcsin(n_l/n_r·sin(α))`) 490 ops, `:1151`
(`cos(radians(α))`) 413, `:1183` (`cos(...)`) 399.

**Cause:** the parameter vector is `complex128`, so the geometry helpers do *complex* `cos`/`sin`/`arcsin` (expanded by XLA into long
formulas), and they are called from 8 places (surface reflectivity/transmissivity, forces, optomechanical phases), traced once per
component group.

**Experiment (not committed):** doing that trig in float64 (`jnp.real` of angle and refractive indices, which are physically real):

| | forward ops | forward compile (2 runs) | gradient lower+compile |
|---|---:|---:|---:|
| complex (current) | 3,369 | 20.4 / 20.2 s | 5.9 / 6.6 s |
| real | **2,256 (−33 %)** | **15.8 / 15.5 s (−4.7 s)** | 5.5 / 5.9 s |

Sensitivity and all 193 gradients **bit-identical** (0 difference). **≈ −5 s on the 28.8 s total.** Same idea applies to other trig/exp
on physically real inputs (`phase_shift`, `space_modulation`, `standardize_output`'s list→array). Owner: WS3 (`components.py`).
Gradient caveat: `jnp.real` drops the imaginary cotangent, which is correct only because these inputs are real parameters
(optimised parameters are float64 inserted into the complex vector). Validate with `bench.py check` when implementing.

**Follow-up questions (checked on the fixture):**
- *Are α and the refractive indices fixed?* **Yes.** None of the 7 incidence-angle parameters (values 0° / 45°) or the refractive-index
  parameters (all 1.0) is optimised, linked or swept. The 4 optimised `angle`s are the **squeezer** angles. So `cos α` and
  `cos(arcsin(n_l/n_r·sin α))` could be computed once in NumPy at build time and passed in as a small array (a gather, no trig in
  the graph). This must stay conditional: precompute only calls whose α/n inputs are not optimised, linked or swept, and keep traced code otherwise.
  It needs changed component signatures (WS3). Extra gain over the real-trig change is small (real trig already removes ~1,100 of the
  ~1,300 ops), so do real trig first.
- *Are the call sites inside Python loops over components?* **No.** WS3's `assembly.component_outputs` loops over function
  **types** (17 bodies: carrier 3, signal 11, noise 3), each `vmap`ped over all its calls (e.g. 54 surfaces), so graph size does not
  grow with the component count. `lax.scan` doesn't apply: the bodies are different functions, and `vmap` already gives one body per type.
  The duplication comes from the helpers being **inlined into 5 different bodies** (surface in carrier and signal, force L/R,
  optomechanical phase L/R) and expanded as complex math. Remaining `vmap(lax.switch)`: only parameter linking (1 linked parameter), small.

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
| 2026-11-21 | `integration` | WS2 custom_vjp adjoint + WS1 SuperLU kernel (tape = vals, x, W; noise sparse and computed once) | Ryzen laptop, 10 | 62.7–64.4 vs 69.8 (orig) / 61.0 (ws1) | 54–58 | 1.59–2.05 vs 4.23 (orig) / 1.91 (ws1) | PASS (signal 2.4e-8, noise 1.0e-8, grad 2.0e-5 rel, cosine 1.0) | peak RSS 1.82 GB vs 2.57 (orig) / 2.02 (ws1); ~7.8 GB est. at 101; **needs cluster `benchmark.py` gate**. See `docs/WS2_CHANGELOG.md` step 2, `docs/INTEGRATION_GUIDE.md` |
| 2026-11-21 | `ws4` | WS4: build without eager XLA compiles. The ~60 small compilations drop to 14: `jnp.array` copies/casts become NumPy + `jnp.asarray`, `prepare_arrays` indexes on the host, and the space entries are one jitted batch. Lazy `plot`/plotly import. Mirror/beamsplitter coefficients stay eager, because jit fusion FMA-contracts their phase and changes the last bit. | EPYC 7763, 101 | 62.22 vs 64.83 (pipeline, 3 trials) | 36.81 vs 37.28 | 24.21 vs 25.12 | PASS (0 diff, all outputs and 193 gradients bitwise equal) | **setup: build 1.29 → 0.51 s, imports 0.49 → 0.46 s** (build part, 5 trials: 1.81 → 0.91 s). Other stages are unchanged within noise, so the real gain is ~0.8 s. Node shared with other jobs. `run_build_step` outputs bytewise identical on 16 setups (uifo fixtures, aLIGO ×3 modes ± opt, db sweep, voyager ×3, cavity, random uifo) with `PYTHONHASHSEED` fixed: the baseline itself is hash-seed dependent in `noise_system_matrix_indices` (set iteration over unused ports) |
| 2026-11-21 | `ws3` | WS3(a): grouped component evaluation (no `vmap(lax.switch)`: 17 instead of 57 traced function bodies) + signal system assembled on its 2597-entry pattern (`values`, `rhs`), densified once for the unchanged dense solves | EPYC 7763, 101 | 41.41 vs 64.90 (**1.57×**) | 20.06 vs 38.83 (**1.94×**) | 19.79 vs 25.03 (1.26×) | PASS (warn: signal 9.6e-8, noise 2.4e-8, grad 2.9e-6 rel, cosine 1.0) | `bench.py parts --trials 3 --repeats 1` vs saved baseline json; forward compile 24.7 → 13.2 s, gradient compile 10.0 → 3.9 s, gradient run 4.6 → 2.4 s, forward run 19.5 → 16.7 s (still dense LU: WS1); peak RSS 14.7 → 10.5 GB; trials 41.4/39.8/43.3; WS4 fork jobs may have shared the node. `tests/test_assembly.py`: matrices bitwise equal; aLIGO, voyager, cavity carrier sweep bitwise equal |
| 2026-11-21 | `ws3` (not kept) | WS3(b) experiment: assembly plan passed as a leafless pytree (`StaticPlan`, FINDINGS §1.1) so all plan indices are compile-time constants | EPYC 7763, 101 | 41.05 | 20.13 | 21.30 | PASS (identical to WS3(a)) | **no gain** vs WS3(a) (compile 20.06 → 20.13 s): §1.1's 6 s came from constant indices in the 19-branch `switch` graph, which WS3(a) already removed. Reverted to keep the code simple |
| 2026-11-21 | `main` @ `f5ea168` | **official `benchmark.py --cpus 1 --repeats 3`, 101 freqs** (WS1+WS2+WS3+WS4 merged) | Ryzen AI 7 350 laptop, 1 CPU | **28.82** vs 44.20 (organisers' Xeon) | 25.97 (fwd 18.21+1.78, grad 5.05+0.33) | **0.37** vs 17.68 | **PASS** (max \|Δln S\| 5.374e-6, FD check passed) | forward run 13.29 → 0.31 s, gradient run 4.34 → 0.22 s; setup 2.32 s; peak RSS 0.89 GB (vs ~14.4 GB); compile is now **90 %** of the total. Same commit **without** `--cpus` (16 CPUs): 15.96 s, compile 10.96 s (not comparable: jury pins 1 CPU). Laptop compiles ~2× slower than the Xeon, so the Xeon total should be lower (not measured). |
| 2026-11-21 | `main` + noise-once fix | `adjoint.pattern_arrays`: ignore one-value (dummy) sweeps, so Q is built once (WS2_CHANGELOG step 3) | Ryzen laptop, 1 CPU, 101 | 31.58 / 33.78 vs 33.28 / 35.45 (interleaved A/B, before fix) | – | ~0.43 | PASS | the machine was loaded during this A/B (imports and build also slower than in the 28.82 s run); compare only within the A/B pairs |
| 2026-11-21 | `host-native` @ `92a418b` | whole `simulate` as NumPy/SciPy host code behind one `custom_vjp` (two callbacks); component functions re-bound to NumPy + dual numbers (single-source physics); hand-written reverse sweep; + 1 step of SuperLU iterative refinement | EPYC 7763, 101 | **2.43 (`bench.py`, 3 trials) / 2.80 official** vs 19.8 / 20.3 (`main`) vs 58.3 / 62.2 (forked) | 0.59 | 0.59 | PASS (signal 5.3e-8, noise 1.5e-8, grad 1.6e-6 rel, cosine 1.0) | `benchmarks/epyc-host-native-run1.json`; stages: imports 0.39, build 0.72, fwd compile 0.04, fwd run 0.28, sens compile 0.26, grad compile 0.22, grad run 0.43; peak RSS 0.36 GB. `tests/test_host.py`: voyager 48 gradients vs JAX 7.6e-14 |
| 2026-11-21 | `host-native` @ `df70796` | SuperLU iterative refinement (1 step) — **fixes a WS1 regression on `main`**: Voyager example sensitivity differed from the original by \|Δln S\| = 4.2e-3 with SuperLU | EPYC 7763, voyager 30 f | — | — | — | Voyager noise vs dense LU 3.6e-3 → 2.8e-13; scored setup 3.0e-8 → 4.5e-9 | equilibration alone: 5.4e-12 on Voyager but 2.1e-7 on uifo, so refinement was chosen |
| 2026-11-21 | `host-native` | build: mirror/beamsplitter/space coefficients via the NumPy-re-bound component code (no eager XLA compiles); backward reuses the forward state + SuperLU factorisations (host-side residuals, keyed by input bytes, recompute on miss); `np.bincount` instead of `np.add.at` | EPYC 7763, 101 | **1.83** (1.75 / 1.83 / 1.87) vs 2.43 | 0.59 | 0.32 | PASS (unchanged errors) | build 0.72 → 0.30 s, gradient run 0.43 → 0.17 s. Base-matrix entries may now differ from eager XLA in the last bit (NumPy libm); all checks unchanged |
| 2026-11-21 | `host-native-xla-opt0` | `host-native` + `--xla_backend_optimization_level=0` set inside `differometor/__init__.py` (before the CPU backend starts; recorded in the report as `xla_flags`; opt-out `DIFFEROMETOR_XLA_OPT0=0`). **Needs organiser approval** | EPYC 7763, 101 | **1.71 median (1.66 / 1.78 / 1.71) vs 2.05 (2.05 / 2.05 / 2.08) for `host-native`, −0.34 s (−16%)** | 0.32 vs 0.57 | 0.38 vs 0.35 | PASS (all 6 runs; `bench.py check` PASS) | official `benchmark.py --cpus 1 --repeats 3`, branches interleaved in one job. Compile: forward 0.04 → 0.02, sensitivity 0.29 → 0.17, gradient 0.25 → 0.13 s; warm run ~+0.03 s. Same flag via env on `main`: 22.2 → 12.8 s (`benchmarks/epyc-main-*.json`) |
| 2026-11-21 | `the-prime-model` | (1) NumPy-first unpacking of index arrays (iterating a jax array compiled 4× `jit(_unstack)` in the build); (2) SciPy's SuperLU core loaded on its own instead of importing `scipy.sparse(.linalg)` (~0.12 s import), residual products of the refinement via `np.bincount` on the pattern | EPYC 7763, 101 | **1.28** (1.28 / 1.28 / 1.28) vs ~1.71–1.83 | 0.37 | 0.31 | PASS (tests/test_host, test_adjoint) | build 0.27–0.30 → 0.12 s, forward run 0.30–0.35 → 0.18 s. Tried and **rejected**: host-side `signal_detector` (readout compile only −0.03 s); SuperLU orderings MMD / reused COLAMD+NATURAL (2–4× slower factorisation) |
| 2026-11-21 | `the-prime-model` | final interleaved A/B, official `benchmark.py --cpus 1 --repeats 3`, 3 rounds, one job | EPYC 7763, 101 | **flag on: 1.47 median (1.37 / 1.47 / 1.47); flag off (`DIFFEROMETOR_XLA_OPT0=0`): 1.76 (1.64 / 1.76 / 1.85)** vs `host-native` 2.04 (2.00 / 2.04 / 2.38), `host-native-xla-opt0` 1.72 (1.68 / 1.72 / 2.14) | — | 0.32–0.39 | PASS (all 12 runs + `bench.py check`) | round 3 was a noisy period on the shared node (host-native 2.38). Official totals are ~0.2 s above `bench.py`'s scored-equivalent (benchmark imports more modules and reads inputs). Block-triangular form measured: no gain (612-core factorisation 69.6 ms vs 67.1 ms for the full 751) |
| 2026-11-21 | `the-prime-model-new` | dual numbers seeded only on the input slots each component reads; refinement only for noise-weight + adjoint solves (signal solve measured not to need it); `signal_detector`/`power_detector` as host callbacks with hand-written VJPs; SuperLU `PanelSize=4, Relax=4` (factorisation 90.8 → 76.2 ms); lazy `jax.scipy` import | EPYC 7763, 101 | **flag on: 1.24 median (1.22 / 1.28 / 1.24)**; flag off: 1.60 (1.60 / 1.64 / 1.46); `the-prime-model` (flag on) in the same job: 1.44 (1.39 / 1.44 / 1.48) | 0.29 (flag on) | 0.29 | PASS (all 9 runs; `bench.py check`: \|Δln S\| vs reference 5.375e-6, gradient 1.6e-6, cosine 1.0; tests/test_host, test_adjoint) | official stages (flag on, run 1): imports 0.45, build 0.14, fwd lower+compile 0.04, fwd run 0.17, sensitivity lower+compile 0.18, gradient lower+compile 0.08, gradient run 0.16. Tried, no gain: other XLA/JAX compile flags (`xla_llvm_disable_expensive_passes`, `jax_disable_most_optimizations`, legacy CPU runtime), SuperLU `Equil=NO`; pivot threshold 0.1 (−9 ms) not adopted (pivoting robustness) |
| 2026-11-21 | `the-prime-model-new` (round 2) | two more bitwise-neutral XLA compile options (`xla_cpu_use_fusion_emitters=false`, `xla_cpu_parallel_codegen_split_count=1`; XLA compile 0.35 → 0.15 s); WS2/WS3 JAX-path arrays left out of the build output when the host plan is used (102 → 31 jit leaves; `host.JAX_PATH_ARRAYS=True` keeps them); SuperLU solves batched over frequencies (one vectorised residual product per refinement step, bitwise identical); COLAMD order computed once and reused via symmetric permutation + `ColPerm=NATURAL` (factorisation 80 → 67 ms, same fill, round-off-level differences); dual-number tangents stored direction-first (Jacobians 64 → 35 ms, bitwise identical) | EPYC 7763, 101, shared node (6 interleaved rounds) | **flags on: 1.16 median (1.30 / 1.11 / 1.14 / 1.19 / 1.04 / 1.24, min 1.04)**; flags off: 1.46 (1.39 / 1.39 / 1.52 / 1.56 / 1.41 / 1.92); previous commit (opt0 flag) in the same job: 1.33 (1.44 / 1.28 / 1.29 / 1.38 / 1.34 / 1.31) | 0.24–0.31 | — | PASS (all 18 runs; `bench.py check` \|Δln S\| 5.374e-6 = original code, gradient 1.6e-6, cosine 1.0; test_adjoint, test_host (Voyager gradient 4.5e-13 vs JAX), test_assembly ALL PASS; with `REUSE_ORDERING=False` all saved arrays bitwise equal to the previous commit) | node had 78/128 cores busy, ±0.15 s noise. Evaluating frequency-independent component calls once: only ~5 ms left after the Dual change, not done. Remaining: imports ~0.44 s (fixed), benchmark's own compiles |
| 2026-11-21 | `the-prime-model-new` (round 3: scenarios) | `scripts/scenarios.py`: benchmark pipeline on 16 inputs (benchmark workload, perturbed design variables ±1/±10 %, 20/400 frequencies, wideband sweep, second trained UIFO, random 3×3 UIFOs in all three modulation modes, Voyager ×3 modes, aLIGO ×2, cavity carrier sweep), timed in fresh pinned processes and checked against the original dense code path. Fixes: (1) **sqrt-linked parameters** (amplitude modulation: amplitude = sqrt(power)) made `host_plan` fall back to the JAX path (5.2 s, 4.4 s compile) -> host support for identity and sqrt links with analytic derivative (refuses, with an error, a non-real derivative instead of a wrong gradient); (2) **signal-solve refinement back on**: unrefined, the second trained UIFO (`uifo_20_5000`) was 1.6e-5 off at one frequency (every refined solver and the original code agree to 1e-10); now 8.2e-10. Cost +21 ms forward | EPYC 7763, shared node | across 16 scenarios (3 rounds each): std of median totals 1.22 s -> 0.32 s, max 5.19 -> 2.00 s (400 frequencies, linear in work: 3.3 ms/frequency); all others 0.63-1.18 s; all on the host path, 13-47x faster than the original code path | — | — | ALL judging criteria PASS (scripts/check_correctness.py, flags on and off); test_adjoint, test_host PASS; every scenario finite, gradient non-zero, vs original path <= 1e-7 except Voyager modulation read-outs (ampl 8e-6, freqmod 7e-3) where per-frequency analysis shows the *original* dense LU is the less accurate one (refined SuperLU/LAPACK agree better; freqmod is ill-conditioned in float64 for every method) | 400-frequency reference skipped (original path needs > 48 GB) |

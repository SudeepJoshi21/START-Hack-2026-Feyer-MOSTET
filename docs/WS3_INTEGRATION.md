# WS3 (assembly) and how to integrate it with WS1 (sparse solver) and WS2 (adjoint)

Branches: `ws3` (this work, on top of `main` `b63ff54`), `ws1` (sparse SuperLU solve), `ws2-adjoint` (custom gradient),
`ws4` (build speed-ups). Earlier guide for WS1+WS2: [`WS1_WS2_INTEGRATION.md`](https://github.com/SudeepJoshi21/START-Hack-2026-Feyer-MOSTET/blob/ws1/docs/WS1_WS2_INTEGRATION.md) on `ws1`.

## 1. What WS3 changes

New module [`differometor/assembly.py`](../differometor/assembly.py). `simulate` gets one keyword-only argument,
`assembly=None`, which `run_build_step` fills with a build-time plan (NumPy). With `assembly=None`, or if a precondition
fails, the original code runs unchanged.

1. **Grouped component evaluation instead of `vmap(lax.switch)`.**
   - **Problem:** with a batched function index, `lax.switch` evaluates **all 19** component functions for
     every call and then selects. The signal fill has 679 calls and only 11 distinct types; carrier and noise use 3 types each.
     XLA therefore traced, compiled, ran and differentiated 57 function bodies, 19 for each of the carrier, signal and noise fills.
   - **What the build does now:** it sorts the calls by type. The plan holds, for each type `t`, a `(K_t, 7)` array of input indices.
     The tuple position fixes the function and the array shape fixes the call count, so both are static under `jit`
     even though the index values are traced arguments.
   - **At run time:** `assembly.component_outputs` vmaps each used function over its own calls only, 17 bodies in total.
   - **Remapping:** the output indices are remapped at build time to the grouped order, so no permutation is needed at run time.
2. **The signal system is assembled on its sparsity pattern.**
   - **The original pipeline:** start from the base matrix, scatter in the component outputs, scale the connector entries by −carrier, conjugate three blocks.
     The whole sequence ran on a dense (F, 751, 752) complex array (0.9 GB at 101 frequencies), with several full copies.
   - **What the build does now:** it records each possibly-non-zero position's recipe: source (base value or output),
     optional carrier index, optional conjugation. There are 2,597 matrix positions and 71 right-hand-side positions.
   - **At run time:** `assembly.signal_values_and_rhs` computes `values (F, 2597)` and `rhs (F, 751)` with a few
     gathers and `where`s.
   - **Standalone:** `ws3` still uses the original dense LU solves, so it densifies once (`assembly.densify`).
     Integrated with WS1/WS2, **nothing is densified** (§3).
3. **Carrier and noise fills** use the grouped evaluation through the same scatter as before (`assembly.update`).

Preconditions, checked at build time (otherwise the original path is used):
- no duplicate write positions in `signal_matrix_indices` or `signal_carrier_placing_indices` (with duplicates, `.at[].set` depends on scatter order);
- function indices are in range;
- the number of carrier indices matches the number of placing positions.

## 2. Correctness and speed

`python tests/test_assembly.py`. All of it passes.

| Check | Result |
|---|---|
| Benchmark setup, 8 frequencies: carrier matrix new vs original | **bitwise equal** |
| Signal system + right-hand side, `densify(values)` vs the original dense steps | **bitwise equal**; no original non-zero lies outside the pattern |
| Outputs inside `jit`, with vs without the plan | carrier equal; signal 6.8e-8, noise 1.1e-8 rel (XLA rounds the last bits differently inside the two programs; cond(A) ≈ 1e30 amplifies it) |
| Gradient of mean log sensitivity | 3.6e-7 max rel |
| aLIGO frequency sweep, aLIGO squeezer-db sweep, voyager frequency sweep, cavity with a **carrier** (tuning) sweep, via `df.run` | **bitwise equal** (plan used in all four) |

`bench.py check --freqs 101`: **PASS**.
- Sensitivity vs reference: 5.374e-6, the same as the baseline.
- Carrier and powers are identical to the baseline.
- Signal differs by 9.6e-8 and noise by 2.4e-8.
- Across all 193 gradients the worst relative error is 2.9e-6, with cosine similarity 1.0.

Speed: `bench.py parts --parts pipeline --trials 3 --repeats 1 --freqs 101`, EPYC 7763, 1 CPU:

| | Baseline `02ec741` | `ws3` | Speedup |
|---|---:|---:|---:|
| **First-run total (scored)** | 64.90 s | **41.41 s** | **1.57×** |
| Compile total | 38.83 s | 20.06 s | 1.94× |
| forward compile / gradient compile | 24.7 / 10.0 s | 13.2 / 3.9 s | 1.87× / 2.55× |
| forward run | 19.5 s | 16.7 s | 1.17× (still the dense LU solves) |
| gradient run | 4.6 s | 2.4 s | 1.90× |
| peak RSS | 14.7 GB | 10.5 GB | 1.41× |

For comparison, `ws1` alone measured 44.68 s. WS1 speeds up the run (forward run 4.9 s), WS3 the compile (20 s), so
they are complementary.

## 3. Integration: the target `simulate`

The workstreams fit together without overlapping:

```
build (WS4, NumPy) ──► assembly plan (WS3) ──► values (F, nnz), rhs (F, N)        [assembly.signal_values_and_rhs]
                                               │
                                               ▼
              adjoint.solve_signal_and_noise_weights(values, rows, cols, rhs, C)   [WS2: custom_vjp]
                    forward:  SuperLU factor once per f → x = A⁻¹b, W = A⁻ᴴC         [WS1 kernel via pure_callback]
                    backward: refactor → λx = A⁻ᵀx̄, λu = A⁻¹conj(W̄); vals̄ on the pattern only
                                               │
                                               ▼
              noise = Re(wᴴQw) with Q values = grouped noise outputs                  [WS2 quadratic_noise + WS3 groups,
                                                                                         computed once if f-independent (WS1)]
```

**The pattern is shared.**
- `assembly["signal"]["rows"/"cols"]` is the same 2,597 positions, in the same row-major order, as WS2's
  `adjoint.signal_pattern`. The WS1 `_structure_arrays` pattern is the same set too (verified on the fixture).
- Use the assembly plan as the single source and drop `signal_pattern_rows/cols` (WS2) and `signal_pattern` (WS1) as separate inputs.

## 4. Merge steps

Start from `main`. Merge in this order; each step must pass `bench.py check --freqs 101` before the next.

1. **`ws4`.** It only touches `build.py`, `setups.py` and `__init__.py`, and its outputs are bitwise identical, so it should merge cleanly.
2. **`ws2-adjoint`.** Should be clean: it only adds `adjoint.py`, `tests/test_adjoint.py`, docs, and edits `simulate.py`.
3. **`ws3`.** Expect conflicts in `simulate.py`: both add keyword-only arguments and both rewrite the signal and noise section.
   Resolve to:
   - signature: `*, assembly=None` (drop `signal_pattern_rows/cols`, which now come from the plan);
   - carrier: `assembly.update` (grouped) + the dense carrier solve (286×286, only one carrier value here);
   - signal: `values, rhs = assembly.signal_values_and_rhs(parameters, carrier, assembly["signal"], n)`, then
     `vmap(adjoint.solve_signal_and_noise_weights, in_axes=(0, None, None, 0, 1))(values, rows, cols, rhs, carrier_selections)`
     with `rows, cols = assembly["signal"]["rows"], assembly["signal"]["cols"]`. Do **not** call `densify`;
   - noise values: replace WS2's `vmap(component_outputs)(parameters, noise_function_input_indices, …)` with
     `vmap(assembly.component_outputs, in_axes=(0, None))(parameters, assembly["noise"]["groups"])[:, assembly["noise"]["outputs"]]`;
   - `adjoint.pattern_arrays` preconditions (unique noise positions, square noise matrix): keep them and fall back to the
     original path if either the assembly plan or the adjoint preconditions fail.
4. **WS1's kernel into `adjoint._fwd/_bwd`.** Follow `WS1_WS2_INTEGRATION.md` §3 on `ws1`:
   - one SuperLU factorisation per frequency inside one `pure_callback` per direction;
   - store `values, x, W` on the tape, not the LU factors;
   - compute the noise values once when no swept parameter feeds a noise function.
5. **Validate:**
   - `python tests/test_assembly.py`, `python tests/test_adjoint.py`;
   - `bench.py check --freqs 101`;
   - `bench.py parts --parts pipeline --trials 3 --freqs 101`;
   - `python benchmark.py --cpus 1 --repeats 3` (the gate).

## 5. What remains

**After the integration (estimate, not measured):**
- Compile is about WS3's 20 s or less: the dense LU and its AD are gone.
- Forward and gradient runs are about WS1/WS2's sparse numbers (a few seconds) minus the dense matrix traffic.
- So the first-run total is roughly 20–28 s, against 64.9 s for the baseline.
- **No dense (F, 751, 751) arrays anywhere.** The tape becomes `values`, `x`, `W` and the grouped outputs, all ≤ (101, 2,668)
  complex. Peak memory should fall from 9.5–14.7 GB to well under 1 GB, and the forward and gradient run times should fall
  to roughly the cost of SuperLU plus the elementwise component functions.
- **Compile becomes nearly all of the score.**

**Already tried, no gain:** passing the plan as compile-time constants (a leafless pytree, FINDINGS §1.1). Compile
stayed at 20.1 s, because §1.1's 6 s came from the 19-branch `switch` graph, which WS3 removes. Don't redo it.

**Next levers for compile** (re-profile the integrated graph first; the dense solves and their AD are gone there):
1. **Evaluate frequency-independent calls once.** Calls whose inputs include no swept parameter can run on one parameter
   row and be broadcast. The plan measured 1,531 of the 2,537 matrix entries as frequency-independent; `surface` and
   `force_calculation_*` are likely examples. Put these calls in a second group per type, but only if it doesn't double
   the traced function bodies; measure compile.
2. **XLA `--xla_backend_optimization_level=0`** (FINDINGS §1.7): ask the organisers first.
3. **Plan B:** one `custom_vjp` around all of `simulate` (see `OPTIMIZATION_PLAN.md` §6), if compile still dominates after
   the integration.

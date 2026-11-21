# Integrating WS1 (sparse solver) with WS2 (adjoint gradient)

Branches: `ws1` (this guide; same commits as the old local `ws1-solver`), `origin/ws2-adjoint` (`36405a8`).
Plan: [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md). WS2 details: [`WS2_CHANGELOG.md`](WS2_CHANGELOG.md) on `ws2-adjoint`.

## 1. What each branch did

| Change | WS1 (`ws1`) | WS2 (`ws2-adjoint`) | Keep in the merge |
|---|---|---|---|
| One LU per frequency, reused for `A` and `Aᴴ` | yes (dense path, `SPARSE_SIGNAL_SOLVE=False`) | yes (inside `custom_vjp`) | WS2's structure |
| Third solve removed: noise = `Re(wᴴQw)` | yes | yes | either (same maths) |
| Signal and noise-weight solves on the sparsity pattern with SciPy SuperLU | **yes** (`pure_callback`, 0.6 ms per factorisation) | no (dense `lu_factor` of the densified pattern) | **WS1's solver kernel** |
| Hand-written gradient (`custom_vjp`), cotangent only on the pattern, smaller AD tape | no (`lax.custom_linear_solve`, dense cotangents) | **yes** | **WS2** |
| Noise matrix `Q` not built densely | partly: built once instead of per frequency | **yes**: 272 values + `segment_sum` | **WS2**, plus WS1's "compute once" (§3 step 5) |
| New `simulate` arguments | 2 positional args appended (`signal_pattern`, `noise_parameter_rows`) | 2 **keyword-only** args (`signal_pattern_rows/cols`) | **WS2's keyword-only style** (does not touch `run_build_step`'s positional zip) |

The two branches solve **different halves of one problem**. WS1 makes the factorisation cheap; WS2 makes the
gradient cheap and removes the dense noise matrix. Neither gets the full gain alone:
- On `ws1` the gradient still runs on dense (F, 751, 751) cotangents. The gradient run stays at ~4.1 s.
- On `ws2-adjoint` every frequency still does a dense 751² LU: 36 ms × 101 ≈ 3.6 s in the forward run, plus the
  (101, 751, 751) LU factors (≈0.9 GB) kept on the tape.

## 2. Recommended approach: WS2's structure, WS1's kernel

`simulate.py` diverges a lot between the two branches, so **do not `git merge ws1`** into `ws2-adjoint`. Instead:

1. `git switch -c integration origin/main` and `git merge origin/ws2-adjoint`. This should be clean, since WS2 is based on a recent `main`.
2. Port WS1's host solver into `differometor/adjoint.py` by hand (§3).
3. Take from `ws1` only the `docs/FINDINGS.md` rows and this file (`git checkout ws1 -- docs/WS1_WS2_INTEGRATION.md`).
   **Do not** take WS1's `simulate.py`:
   - its `sparse_solve` / `custom_linear_solve` / `solve_with_lu` helpers become redundant;
   - its positional `signal_pattern` / `noise_parameter_rows` arguments are not needed.

## 3. Porting steps (WS1 owner)

1. **One host callback per direction, factor once, solve several right-hand sides.**
   WS1 currently factorises separately for each solve: once for `x`, and once per detector for `W`.
   In the merge, each callback factorises once per frequency:
   - `_host_forward(vals, b, C, rows, cols)`: for each frequency, `lu = splu(csc(vals[f]), permc_spec="COLAMD")`,
     then `x[f] = lu.solve(b[f], "N")` and `W[f, d] = lu.solve(C[f, d], "H")`.
   - `_host_backward(vals, xbar, Wbar, rows, cols)`: refactorise (0.6 ms; no factor objects survive between
     callbacks), then `lam_x = lu.solve(xbar, "T")` and `lam_u[d] = lu.solve(conj(Wbar[d]), "N")`.
   - Reuse WS1's `_csc_structure` (the column-major order is computed once on the pattern) and the loop in
     `_host_sparse_solve` (`simulate.py` on `ws1`).
2. **Make the custom_vjp batched over frequencies.** Do not vmap it. `solve_signal_and_noise_weights(vals (F,nnz), rows, cols,
   b (F,n), C (D,n) or (F,D,n))` with one `jax.pure_callback` each in `_fwd` and `_bwd`. This means one Python call per
   direction instead of 101. If it must stay vmapped, use `vmap_method="sequential"` (that is what WS1 uses and it works).
3. **Tape:** store `vals, x, W` (and the pattern), **not** LU factors. This drops the ~0.9 GB of dense factors from the
   tape at 101 frequencies. The backward formula in `_bwd` stays exactly as WS2 wrote it. Only `lu_solve(trans=…)` becomes the
   callback: trans 0 → `"N"`, 1 → `"T"`, 2 → `"H"`.
4. **Keep a dense fallback** behind a switch (WS2's `ENABLED`, plus WS1's `SPARSE_SIGNAL_SOLVE`), so A/B tests and
   unusual setups still work.
5. **Compute the noise values once, not per frequency.** On `ws2-adjoint`,
   `vmap(component_outputs, in_axes=(0, …))(parameters, …)` evaluates the 66 noise functions for all 101 rows. They
   are identical unless a swept parameter feeds a noise function. Port WS1's check (`_structure_arrays` on `ws1`):
   compare the swept parameter indices with `noise_function_input_indices`. If none overlap, evaluate on `parameters[:1]`
   and broadcast. Pass it as a third keyword-only array, or as a static flag.
6. **Use one canonical pattern.** Keep WS2's `adjoint.signal_pattern`. On the fixture it gives exactly the same
   2,597 positions as WS1's `_structure_arrays`:
   - the sets are equal, there are no negative row indices and no duplicates;
   - all 1,004 carrier-scaling positions are included;
   - WS2_CHANGELOG's "2,668" does not match what the code produces;
   - checked with a script on 2026-11-21.

   Add the row wrap (`rows % n`) as a cheap guard (§5), and a build-time assertion that every non-zero of the assembled
   matrix at one frequency lies on the pattern.
7. **Skip the dense `q_base` product when the base is zero.** In `quadratic_noise`, `W @ q_base.T` is a dense
   751×751 product per frequency and detector, and `q_base` is all zeros in the fixture. Decide at build time
   (NumPy) and drop the term.

## 4. Validation before merging to `main`

```bash
python tests/test_adjoint.py                    # WS2's unit tests; extend with sparse-vs-dense on random matrices
python scripts/bench.py check --freqs 101       # all outputs + 193 gradients vs baseline 02ec741
python scripts/bench.py parts --parts pipeline --ref 02ec741 --trials 3 --repeats 1 --freqs 101
python benchmark.py --cpus 1 --repeats 3        # official gate (cluster)
```
- WS2's tests compare against the original formulation to < 1e-10. That holds for dense LU. Sparse LU on the real
  matrices (cond ≈ 1e30) differs from dense by ~1e-8, so in the tests compare **sparse vs dense on well-conditioned
  random matrices** to 1e-10, and leave the real fixture to `bench.py check`.
- aLIGO generality: run the sparse and dense paths on `examples/data/aligo_test_data.json` with a frequency sweep **and**
  a squeezer-`db` sweep. A noise parameter is swept there, so step 5's fallback is exercised. On `ws1` this agreed to ≤ 1.4e-9.

## 5. Small guard for other setups

WS2's `signal_pattern` wraps negative *column* indices but not negative *row* indices. In the fixture all rows are
0…750, so nothing happens. In a setup with a negative row `r`, though, `(r, c)` and `(r + n, c)` would be two pattern
entries for the same matrix element. The gather `signal_matrix[:, rows, cols]` would then return that element's
cotangent twice, doubling its gradient. Wrapping rows with `% n` before `np.unique` removes the case for free.

## 6. Expected result (estimate, not measured)

From the stage timings on EPYC 7763 at 101 frequencies:
- **Forward run:** about 5 s → about 3–4 s. The dense signal-matrix traffic remains (WS3).
- **Gradient run:** about 4.1 s → about 1–2 s.
- **Compile:** unchanged at about 34 s, because it is assembly (WS3).
- **First-run total:** about 45 s → about 40 s.

After that, compile is ~85% of the score: see the ideas in §7.

## 7. Next optimisations after the merge (largest first)

1. **Assemble directly on the pattern (WS3).** Produce `values (F, nnz)` and `b (F, n)` per component type, instead of
   `vmap(lax.switch)` over 679 calls into a dense (101, 751, 752) array. This removes most of the 34 s compile and of the
   remaining forward run. Only 1,006 of the 2,537 entries vary with frequency: compute the other 1,531 once.
2. **Plan B: one `custom_vjp` around the whole of `simulate`.** NumPy/SciPy callbacks for assembly, solve and adjoint mean
   near-zero XLA compile. The assembly's parameter derivatives then need a hand-written or small-JAX pullback. Highest
   ceiling (plan estimate 3–5 s total) and highest risk.
3. **XLA `--xla_backend_optimization_level=0`.** Measured to halve compile (FINDINGS §1.7). **Needs organiser approval.**
4. **Solver micro-wins.** These matter little now, because the solve is < 0.1 s:
   - compute the COLAMD/RCM ordering once at build time and factorise with `permc_spec="NATURAL"` (measured 0.48 ms vs 0.61 ms);
   - BTF (FINDINGS §1.6) shrinks the factorised block from 751 to 612.

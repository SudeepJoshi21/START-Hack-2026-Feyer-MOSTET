# Where the remaining time goes (integrated `main`, `f5ea168`)

Branch `compile-profiling`. Script: [`scripts/profile_compile.py`](../scripts/profile_compile.py). Machine: AMD EPYC 7763, pinned to 1 CPU.

## 1. Official before/after (`benchmark.py --cpus 1 --repeats 10`, 101 frequencies)

The runs were interleaved (before, after, before, after) in one job, so drift on the shared node affects both versions alike.
Reports: [`benchmarks/epyc-before-run{1,2}.json/.png`](../benchmarks/) and [`benchmarks/epyc-after-run{1,2}.json/.png`](../benchmarks/).

| | Run 1 | Run 2 | Warm median | Check |
|---|---:|---:|---:|---|
| Before: forked code `02ec741` | 62.23 s | 60.39 s | 24.8 s | PASS |
| After: integration `f5ea168` (WS1+WS2+WS3+WS4) | 20.32 s | 20.95 s | 0.27 s | PASS |
| **Speedup** | **3.06×** | **2.88×** | **92×** | |

## 2. Contribution of each workstream in the final build (leave-one-out, `bench.py parts`, 3 trials)

| Switched off | First-run total | Cost of removing it |
|---|---:|---:|
| nothing (full) | 19.8 s | — |
| WS3 (back to `vmap(lax.switch)` assembly) | 46.2 s | +26.4 s (compile +14.6, runs +8.4) |
| WS1+WS2 (no adjoint; dense solves) | 38.9 s | +19.2 s (forward run 0.2 → 16.7 s) |
| WS1 only (dense LU inside the adjoint) | 25.4 s | +5.7 s |
| WS4 (old build) | 21.3 s | +1.5 s (build 0.82 → 1.55 s) |

No workstream is dead weight.

## 3. Compile is now 92% of the score

Stage split of the full build:
- imports 0.42 s, build 0.82 s;
- forward lower/compile 1.32 / 12.35 s, forward run 0.20 s;
- sensitivity 0.06 / 0.37 s;
- gradient lower/compile 0.29 / 3.84 s, gradient run 0.16 s.

That is **18.2 s of compile out of 19.8 s**.

### Attribution by stubbing (4 frequencies; compile barely depends on the frequency count)
`jax.clear_caches()` runs before every compile. Without it, monkeypatched globals hit the old jit cache.

| Stubbed out | Forward compile | Gradient compile | Optimized HLO instructions / fusions |
|---|---:|---:|---:|
| nothing | 12.00 s | 3.70 s | 14,091 / 513 |
| **component functions** (all → `p[0]·e₀`) | **2.96 s (−9.0)** | **1.71 s (−2.0)** | **1,588 / 104** |
| carrier dense solve (286) | 12.74 s (noise) | 4.03 s | 13,926 / 500 |
| linking `lax.switch` | 13.27 s (noise) | 4.09 s | 14,091 / 513 |

**≈ 11 of 15.7 s of compile is the physics component functions and their AD (JVP + transpose).** The solver, carrier and
glue code are cheap.

### Per component type, alone (`jit(vjp(vmap(f)))`)
| Type | Calls | Compile |
|---|---:|---:|
| surface | 54 | 0.90 s |
| corrected_optomechanical_phase_right | 54 | 0.85 s |
| corrected_optomechanical_phase_left | 54 | 0.48 s |
| space_modulation_lower / space_modulation | 71 / 71 | 0.45 / 0.42 s |
| 10 others | | 0.03–0.22 s each |
| sum | | 4.65 s |

Alone they sum to 4.65 s. Inside the full program the same functions cost about 11 s, because they appear in several
contexts: the forward-with-residuals program, the transposed gradient program, and the carrier and noise fills.

### XLA backend optimisation level 0 (diagnostic only; needs organiser approval, FINDINGS §1.7)
Forward 12.0 → 7.25 s, gradient 3.7 → 2.12 s, about −6.3 s. The run time is now ~0.3 s, so the slower code that level 0
produces would hardly matter. Do not ship this without approval.

## 4. Consequences

1. **The run is solved:** warm runtime is 0.27 s. Further solver work, a faster language for the numerics, or BTF/ordering
   tweaks can gain at most ~0.3 s.
2. **The lever is to stop compiling the physics.** The component functions, not the solver, are what XLA spends its time on.
   Options, largest first:
   - **Host-native simulate (branch `host-native`):** evaluate the components, the assembly, both solves and the noise
     on the host with NumPy/SciPy (compiled C/Fortran), behind one `jax.custom_vjp`. Component derivatives come from
     forward-mode dual numbers that run the *existing* component code on NumPy, so the physics stays single-source. The
     XLA graph shrinks to two callbacks. Expected total ≈ imports + build + ~1 s.
   - **Hybrid:** keep JAX for everything except the component functions, which get host Jacobians through a callback.
     Smaller change, roughly −11 s.
   - **XLA opt level 0:** about −6 s, but needs approval.
3. **Language choice.** The judged install is `pip install -r requirements.txt` + `pip install -e .`, with numpy and scipy pinned,
   and no numba, cython or pybind.
   - A C/C++/Rust extension would need a compiler on the jury machine at install time. That risk buys nothing: the
     remaining numeric work is ~0.3 s, and NumPy/SciPy are already compiled code.
   - Numba is not in the requirements, and its JIT time would be scored.
   - **NumPy/SciPy is the fastest safe "language" here.**

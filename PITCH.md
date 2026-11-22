# Faster Physics: Differometor in ~1 second

**START Hack Munich 2026 · Feyer challenge · team repo `SudeepJoshi21/START-Hack-2026-Feyer-MOSTET`, branch `main`**

> One full evaluation (101 frequencies, 193 gradients, float64) now takes **1.16 s** on one CPU core,
> down from **~61 s** for the original code on the same machine: **~53× faster**, with all judging criteria passing
> and the physics results unchanged.

## The result

| `benchmark.py --cpus 1` (unchanged scorer) | Original code | Our `main` |
|---|---:|---:|
| **First-run total (scored)**, AMD EPYC 7763, same machine | 60.4–62.2 s | **1.16 s** median of 6 (min 1.04 s) |
| Organisers' baseline, Intel Xeon 8468 | 44.2 s | — |
| Compile (all programs) | 22.9 s (Xeon) | **0.15 s** |
| Forward + gradient run (warm) | 17.7 s (Xeon) | **0.28 s** |
| Peak memory | ~14.4 GB | < 1 GB |
| Max \|Δ ln S\| vs reference (limit 1e-5) | 5.374e-6 | 5.374e-6 |

Sources: `benchmarks/cpu-baseline.json` (organisers), `benchmarks/epyc-prime-new2-xla-flags-run{1..6}.json` (ours),
original code on EPYC: commit `87859e5`. On our laptop (Ryzen AI 7 350, 1 core) `main` runs in 1.68–2.33 s (8 runs).

## What made it slow

1. **Three dense 751×751 solves per frequency**, for a matrix that is **99.5 % zeros**.
2. **A ~9 GB autodiff "tape"**, because JAX stored every intermediate for the gradient.
3. **About half the time was compiling**: XLA compiled all 19 component formulas for every matrix entry, plus their derivatives.

## What we did (in four ideas)

1. **Less linear algebra.** We saw that the noise equals `Re(wᴴQw)` with `Aᴴw = c`, so one of the three solves is redundant.
   The other two share **one sparse LU** (SciPy SuperLU) on the fixed 2,597-entry pattern, with one step of iterative
   refinement for accuracy: ~0.6 ms instead of ~36 ms per frequency.
2. **A hand-written gradient (adjoint method).** One transposed solve with the same factorisation gives the gradient,
   and the matrix gradient is formed only on the non-zeros. The 9 GB tape is gone.
3. **No physics left for XLA to compile.** The whole simulation runs as NumPy/SciPy code behind a single `jax.custom_vjp`.
   The physics formulas in `components.py` are **reused unchanged** (re-bound to NumPy), and their derivatives come from
   forward-mode dual numbers. XLA sees only two callbacks: compile time on the EPYC drops from 20.2 s (the JAX version, `benchmarks/epyc-main-default.json`)
   to 0.15 s.
4. **Startup and compile polish.** No needless imports or tiny compiles while building the setup, plus three XLA compile options
   set in `differometor/__init__.py`. They don't change any number, are recorded in the benchmark report, and can be
   switched off with `DIFFEROMETOR_XLA_OPT0=0`.

Everything is inside `differometor/`. `benchmark.py`, the input files and `requirements.txt` are byte-identical to the
organisers' version. No precomputed results and no compilation cache; float64, all 101 frequencies, all 193 gradients.

## Correctness and robustness

- **Judging criteria:** `scripts/check_correctness.py` → **ALL CRITERIA PASS** (full setup, positive sensitivities
  within 1e-5, everything finite, gradient non-zero, finite-difference check).
- **All 193 gradients** agree with the original code to 2e-5 (cosine similarity 1.000000000000), not just the one the
  benchmark spot-checks (`scripts/bench.py check`).
- **Different inputs** (`tests/test_robustness.py`, `scripts/scenarios.py`): perturbed design variables (±1 %, ±10 %),
  1 to 101 frequencies, wideband sweeps, a second trained UIFO, random UIFOs, Voyager, aLIGO and a Fabry-Perot cavity
  all match the original code; 400 frequencies also runs (1–2 s, no reference: the original code would need > 48 GB). Anything outside the fast path's assumptions **falls back automatically** to the JAX path.
- **More accurate where it matters:** on Voyager, a badly conditioned setup, the refined sparse solves agree with dense
  LU to 2.8e-13, where a plain sparse solve was off by 3.6e-3 (commit `df70796`).
- **Deterministic:** repeated runs give bit-identical outputs; only the timing varies (imports and first call).

## What we learned

- **Profile before optimising.** The "obvious" bottleneck, linear algebra, was only half the story. After fixing it,
  **compilation was ~90 % of the score**, and most of that was tiny formulas compiled in complex arithmetic.
- **Exploit structure.** The matrix pattern never changes across frequencies, which makes it sparse, factorisable once and
  differentiable on its non-zeros alone.
- **Keep one source of truth for the physics.** Running the *same* component code on NumPy and on dual numbers gave us
  speed without a second, hand-ported physics model to keep in sync.
- **Check every gradient, not one.** A strict harness against the original code (all outputs, all 193 gradients, many
  setups) caught a solver accuracy issue on Voyager, a deadlock at 101 frequencies and a NaN leak at singular inputs before
  submission.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python benchmark.py --cpus 1 --repeats 3 --output run.json --save-arrays run.npz   # prints PASS
python scripts/check_correctness.py run.json run.npz                                # ALL CRITERIA PASS
python tests/test_robustness.py                                                     # ALL PASS
```

Details: [`docs/CHANGES.md`](docs/CHANGES.md) (every change, simple), [`docs/FINDINGS.md`](docs/FINDINGS.md)
(all measurements), [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) (how we split the work).

**Team:** ZeyuuuChen, SudeepJoshi21 and teammates (commit authors on `main`).

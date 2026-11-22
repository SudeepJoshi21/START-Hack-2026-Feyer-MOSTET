# Faster Physics: the benchmark that did not run now takes 2.1 s

**START Hack Munich 2026 · Feyer challenge · team repo `SudeepJoshi21/START-Hack-2026-Feyer-MOSTET`, branch `main`**

> Every timing below was measured on **one machine**: an AMD Ryzen AI 7 350 laptop with 16 GB RAM, running WSL2 with 7 GB of
> it available, pinned to one core
> with the unchanged `benchmark.py --cpus 1`.
> - **Main benchmark** (`benchmark.py`, the `start_here.py` setup; 101 frequencies, 193 gradients, float64): the original
>   code **does not run** here; ours takes **2.1 s** median. The original code **cannot run in the
>   7 GB available to WSL2** (it needs ~14 GB of memory). For reference, the organisers' published baseline on their Xeon is 44.2 s.
> - **aLIGO benchmark** (organisers' `benchmark_aligo.py`): **71.9 s → 1.3 s, ~55× faster** on the same machine.
>
> All judging criteria pass, and the physics results are unchanged.

## The result (laptop, 1 core)

**Main benchmark** (`benchmark.py`, 101 frequencies, 193 gradients):

| | Original code | Our pure-JAX version | Our `main` |
|---|---:|---:|---:|
| **First-run total (scored)** | does not fit (needs ~14 GB) | 28.8 s | **2.09 s** median of 9 (1.76–2.96 s) |
| Compile (all programs) | — | 26.0 s | 0.32 s |
| Forward + gradient (first run) | — | 0.53 s | 0.72 s |
| Setup (imports + build) | — | 2.3 s | 1.0 s |
| Max \|Δ ln S\| vs reference (limit 1e-5) | 5.374e-6 (organisers) | 5.374e-6 | 5.374e-6 |

**Organisers' aLIGO benchmark** (101 frequencies, 29 gradients):

| | Original code | Our `main` |
|---|---:|---:|
| **First-run total (scored)** | 71.9 s | **1.29 / 1.33 s** |
| Compile (all programs) | 65.2 s | 0.27 s |
| Forward + gradient (first run) | 2.9 s | 0.12 s |
| Setup (imports + build) | 3.8 s | 0.9 s |
| Max \|Δ ln S\| vs reference (aLIGO limit 1e-6) | 1.2e-10 | 3.8e-11 |

Reports and plots: `benchmarks/laptop/` (index in `benchmarks/laptop/README.md`).

**Reference only, different machine:** the organisers' published baselines on an Intel Xeon 8468 are **44.2 s** (main) and
**25.3 s** (aLIGO): `benchmarks/cpu-baseline.json`, `benchmarks/aligo-cpu-baseline.json`.

Notes:
- **What "our pure-JAX version" is:** WS1–WS4 merged (commit `f5ea168`). It is the same physics inside JAX, before we moved
  it to the host.
- **How the laptop numbers vary:** the times swing by about ±20 % from run to run (Python imports and the first call; see
  `docs/FINDINGS.md` §1.10), so we report the median of 9 runs.
- **Without the XLA compile options** (`DIFFEROMETOR_XLA_OPT0=0`): 2.6 / 3.1 s.
- **The aLIGO original** was measured once on this laptop.

## What made it slow

1. **Three dense 751×751 solves per frequency**, for a matrix that is **99.5 % zeros**.
2. **A ~9 GB autodiff "tape"**, because JAX stored every intermediate for the gradient. That is why the original does not fit in
   the 7 GB available to WSL2.
3. **Compilation**: XLA compiled all 19 component formulas for every matrix entry, plus their derivatives. On aLIGO that
   is 65 of the 72 seconds.

## What we did (in four ideas)

1. **Less linear algebra.** We saw that the noise equals `Re(wᴴQw)` with `Aᴴw = c`, so one of the three solves is redundant.
   The other two share **one sparse LU** (SciPy SuperLU) on the fixed sparsity pattern, with iterative refinement for
   accuracy (the signal solve's residuals in 80-bit `longdouble`). On the laptop one dense 751×751 solve takes ~60 ms; the
   sparse factorisation takes a few ms.
2. **A hand-written gradient (adjoint method).** One transposed solve with the same factorisation gives the gradient,
   and the matrix gradient is formed only on the non-zeros. The 9 GB tape is gone.
3. **No physics left for XLA to compile.** The whole simulation runs as NumPy/SciPy code behind a single `jax.custom_vjp`.
   The physics formulas in `components.py` are **reused unchanged** (re-bound to NumPy), and their derivatives come from
   forward-mode dual numbers. XLA sees only two callbacks: on the laptop, compile drops from 26.0 s (pure-JAX version) to
   0.3 s.
4. **Startup and compile polish.** No needless imports or tiny compiles while building the setup; the setup stays in NumPy and
   moves everything to the device in one transfer. Plus three XLA compile options set in `differometor/__init__.py`. They
   don't change any number, are recorded in the benchmark report, and can be switched off with `DIFFEROMETOR_XLA_OPT0=0`.

Everything is inside `differometor/`. `benchmark.py`, `benchmark_aligo.py`, the input files and `requirements.txt` are
byte-identical to the organisers' latest version (upstream `83f8d2f`, merged into `main`). No precomputed results, no disk
caches and no compilation cache; float64 throughout, all frequencies, all gradients.

## It transfers to other optical setups

The final evaluation uses **a slight variation of the supplied example**. Nothing in our code is specific to the benchmark:
the sparsity pattern, the component groups and the solver are all derived from the setup definition at build time.

Variations on the laptop (`scripts/scenarios.py time`, fresh process, 1 core, our code):

| Setup | Size | Total | Fast path |
|---|---|---:|---|
| Benchmark design, all variables changed by ±10 % | 751 unknowns, 193 variables | 4.59 s (first run after a pause: imports alone 2.6 s) | yes |
| Random 3×3 UIFO (different topology) | 767 unknowns, 251 variables | 1.82 s | yes |
| Second trained UIFO | 769 unknowns, 192 variables | 1.80 s | yes |
| Benchmark setup, wideband 10 Hz–10 kHz | 751 unknowns, 193 variables | 1.58 s | yes |
| Glass beamsplitter, total internal reflection (edge case) | 751 unknowns, 193 variables | 1.70 s | yes |
| Voyager | 161 unknowns, 48 variables | 1.13 s | yes |
| aLIGO (organisers' test) | 29 variables | 1.29–1.33 s | yes |

All of these have finite outputs and a non-zero gradient. A setup outside the fast path's assumptions still runs, through the
JAX path: slower, with the same results.

## Correctness and robustness

On the laptop:
- **Judging criteria:** `scripts/check_correctness.py` → **ALL CRITERIA PASS** (full setup, positive sensitivities
  within 1e-5, everything finite, gradient non-zero, finite-difference check).
- **All 193 gradients** agree with the original code to 2e-5 per entry (cosine similarity 1.000000000000), not just the one
  the benchmark spot-checks (`scripts/bench.py check`, 10 frequencies).
- **Robustness tests** (`tests/test_robustness.py`): design variables changed by ±1 % and ±10 %, 1/2/7 frequencies, a
  mirror with reflectivity exactly 1.0, no state leaking between calls, results independent of what ran before. All PASS.
- **Deterministic:** repeated runs give bit-identical outputs; only the timing varies.

On the team's cluster node (accuracy only, too large for the laptop), `scripts/scenarios.py check` and
`tests/test_generalisation.py`:
- **20 further inputs pass every judging criterion**, including all the variations above.
- **Checked against** the original code and a high-accuracy reference.
- **Voyager:** a badly conditioned setup where refined sparse solves agree with dense LU to 2.8e-13 (a plain sparse solve was
  off by 3.6e-3).

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
- **Two libraries, two conventions.** Moving the physics from JAX to NumPy exposed a subtle difference: on a branch cut,
  JAX's complex `arcsin` and `sqrt` ignore the sign of a zero imaginary part and NumPy does not. For glass on a
  beamsplitter at 225° (total internal reflection) that picked the other branch and the sensitivity was 180× off,
  while still looking plausible. Only the comparison with the unmodified code on edge cases revealed it.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python benchmark.py --cpus 1 --repeats 3 --output run.json --save-arrays run.npz   # prints PASS
python scripts/check_correctness.py run.json run.npz                                # ALL CRITERIA PASS
python benchmark_aligo.py --cpus 1                                                  # organisers' aLIGO setup: PASS
python scripts/scenarios.py time --rounds 1 uifo-random3 uifo20-5000 voyager        # other setups, our code only
python tests/test_robustness.py                                                     # ALL PASS
bash scripts/gate.sh     # cluster only (48 GB): official runs vs every criterion, determinism, all tests, 20 inputs
```

Always pass `--cpus 1`: without it the benchmark uses every core and the numbers are not comparable.

Details: [`docs/CHANGES.md`](docs/CHANGES.md) (every change, simple), [`docs/FINDINGS.md`](docs/FINDINGS.md)
(all measurements), [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) (how we split the work).

**Team:** ZeyuuuChen, SudeepJoshi21 and teammates (commit authors on `main`).

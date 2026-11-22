# What we changed (simple version)

Every change on `main` since the organisers' starting point (`02ec741`), in the order it happened.
Details and all measurements: [`FINDINGS.md`](FINDINGS.md). Plan: [`../OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md).

**Not changed** (byte-identical to the organisers' version): `benchmark.py`, `plot_timings.py`, `start_here.py`,
`requirements.txt`, the input files in `examples/data/`. All speed-ups live in `differometor/`.

## The problem in one line
One run solved 101 big linear systems (751×751) three times each, kept ~9 GB of data for the gradient, and spent
about half of its time just **compiling** the code with XLA.

## The changes

| # | What changed | Why it is faster | Commit(s) | By |
|---|---|---|---|---|
| 1 | Measuring tools: `scripts/bench.py` (laptop-size runs, correctness vs the original code), `scripts/profile_stages.py` | Lets everyone measure speed-ups and check all 193 gradients | `e581c1f` | SudeepJoshi21 |
| 2 | **One matrix factorisation instead of three** per frequency; the third solve was removed with the identity noise = `Re(wᴴQw)` | 2/3 of the expensive work gone | `7b9190e` (WS1), `36405a8` (WS2) | ZeyuuuChen, SudeepJoshi21 |
| 3 | **Sparse solver** (SciPy SuperLU) instead of a dense one. The matrix is 99.5 % zeros | ~0.6 ms instead of ~36 ms per factorisation | `095df6e` (WS1), ported in `b9d8165` | ZeyuuuChen, SudeepJoshi21 |
| 4 | **Hand-written gradient** (adjoint method, `jax.custom_vjp`). Only the 2,597 non-zero matrix entries get a gradient | No 9 GB "tape", no full-size gradient matrices | `36405a8`, `b9d8165` | SudeepJoshi21 |
| 5 | **Matrices built by component type**, not with a 19-way `switch`; built directly as non-zero values | Much smaller program to compile; no full-size matrix copies | `83253e0` (WS3) | ZeyuuuChen |
| 6 | **Faster setup**: no tiny XLA compilations while building; plotting library loaded only when used | Build ~1.3 s → ~0.5 s | `cc0c3c1` (WS4) | ZeyuuuChen |
| 7 | Fix: **deadlock** of the solver callbacks at 101 frequencies on one CPU | Correctness (the run could hang) | `f5ea168` | ZeyuuuChen |
| 8 | Fix: **one step of iterative refinement** in the sparse solves | Correctness on hard setups (Voyager error 3.6e-3 → 3e-13) | `df70796` | ZeyuuuChen |
| 9 | Fix: noise sources (matrix Q) really computed **once**, not 101 times | Small saving; the earlier check never triggered | `2a52c90` | SudeepJoshi21 |
| 10 | **Whole simulation as NumPy/SciPy code** behind one hand-written gradient ("host-native"); the physics formulas are reused unchanged, derivatives via dual numbers | XLA has almost nothing left to compile (compile ~25 s → ~0.3 s) | `92a418b`, `70780c8` | ZeyuuuChen |
| 11 | **XLA compile options** set in `differometor/__init__.py` (allowed by the challenge, on by default, recorded in the benchmark report, off with `DIFFEROMETOR_XLA_OPT0=0`) | The little that is still compiled compiles ~2× faster | `fb2df71`, `445d9f2` | ZeyuuuChen |
| 12 | Polishing: no imports or compiles that are not needed, smaller dual numbers, reused solver ordering, detector read-outs on the host | Each a few tens of ms | `165fbed`, `ea69003`, `f3215ef`, `445d9f2` | ZeyuuuChen |
| 13 | Robustness: support for `sqrt`-linked parameters, refinement also for the signal solve, 16-input scenario tester (`scripts/scenarios.py`), judging-criteria checker (`scripts/check_correctness.py`), XLA flag checker | Works the same on other inputs, not just the benchmark | `3d4d9db`, `e864436`, `725a217` | ZeyuuuChen |
| 14 | Merged into `main`, flags on | — | `c51a3d2` | SudeepJoshi21 |
| 15 | Fix from the final review: at an exact singular point (e.g. a mirror with reflectivity exactly 1.0) the dual-number derivatives turned `∞ × 0` into NaN, which leaked into unrelated gradients (e.g. that mirror's tuning). A zero tangent now contributes exactly 0, as in JAX. Normal results are **bit-identical**. Plus `tests/test_robustness.py` | Correctness on unusual inputs | `61a6aa4` | SudeepJoshi21 |
| 16 | **Extra-precise signal solve** (mixed-precision refinement: residual in 80-bit `longdouble`). On Voyager in frequency-modulation mode, plain float64 was 1.1e-2 off in ln S (the original code: 3.7e-1); now 4.1e-6. The 16-input check now compares with a high-accuracy reference too | Correctness on hard setups (~60 ms) | `fc01438` | ZeyuuuChen |
| 17 | **Inputs read by each component** are found from the code structure, not from the first data seen, plus `tests/test_generalisation.py` (finite differences off the benchmark point, cache leaks, process history) | No data-dependent shortcut (anti-overfitting); results bit-identical | `5277be3` | ZeyuuuChen |
| 18 | **One BLAS thread inside the host code** (restored afterwards). Without CPU pinning, OpenBLAS started one thread per visible CPU, and `start_here.py` took 5–10 s on a shared node | `start_here.py` 5–10 s → 0.36 s; benchmark unaffected | `4c81eb9` | ZeyuuuChen |
| 19 | Fix found with the **edge-cases bundle** (glass on a beamsplitter, total internal reflection): JAX's complex `arcsin`/`sqrt` ignore the sign of a zero imaginary part on the branch cut, NumPy does not, so at 225° the NumPy version took the other branch (sensitivity 180× off). The NumPy version now follows JAX. Four edge cases added to `scripts/scenarios.py` (20 inputs) | Correctness on unusual setups; benchmark bit-identical | `911074c` | ZeyuuuChen |
| 20 | Research report "batched fixed-pivot LU" fact-checked and prototyped (`experiments/research_report/`): its facts about SciPy hold, its speed-ups do not apply here (measured), so nothing was adopted | — | `242f4be` | ZeyuuuChen |
| 21 | `scripts/gate.sh`: one command for all checks (repeated official runs against every judging criterion, determinism, all tests, 20 inputs vs the original code) | Verification | `a6d9bb5`, `57a0e25` | ZeyuuuChen |
| 22 | **Faster setup**: the build stays in NumPy and moves all arrays to the device in one transfer at the end; with a host plan, the dense 751×752 base matrices (~9 MB each), which only the JAX path reads, are left out (empty placeholders; running the JAX path on them raises an error instead of computing on them). Build outputs verified byte-identical apart from those placeholders | Setup 102.5 → 86.1 ms (interleaved A/B); benchmark bit-identical | `54ff4ae` | ZeyuuuChen |
| 23 | Merged the organisers' update (`benchmark_aligo.py`, updated `benchmark.py`, docs; upstream `83f8d2f`). `scripts/gate.sh` now accepts the new official `benchmark.py`. aLIGO: **1.29/1.33 s** on the laptop (organisers' Xeon baseline 25.3 s), \|Δ ln S\| 3.8e-11 vs the 1e-6 limit, fast path used | Keeps the scorer identical to the organisers'; shows the speed-up transfers to another setup | `dbc2288`, `ddbeaa1`, `8878249` | SudeepJoshi21 |

Branches merged on the way: `ws2-adjoint`, `ws1`, `ws3`, `ws4` → `integration` → `the-prime-model-new` → `main` → `the-prime-model-new2` (merged back into `main` for the submission).
Tried and **not** used: `trig-opt` (real-valued maths inside the JAX path, −8 s there, not needed once the physics left
JAX), static index pytrees (no gain after WS3), block-triangular solver split (no gain), other XLA flags (no gain).

## Final code review: will it break on a slightly different input?
Checked on `main` (laptop, 1 CPU). Test file: `tests/test_robustness.py` (compares with the original code path).

| Check | Result |
|---|---|
| Design variables changed by ±1 % and ±10 % | PASS: sensitivity within 6e-9, gradients within 3e-8 (norm-wise) |
| 1, 2 and 7 random frequencies; 20 frequencies (`scripts/scenarios.py`) | PASS: sensitivity within 1.2e-8, gradients within 3.2e-7 (norm-wise) |
| Mirror reflectivity exactly 1.0 (singular point) | PASS after fix #15: non-finite gradient at exactly the same entries as the original code, all others agree |
| Gradient after another forward call with different inputs (host-side cache) | PASS: identical to a fresh gradient (the cache is keyed by the exact input bytes) |
| Same input twice | PASS: bit-identical |
| Other setups run earlier in the same process | PASS: bit-identical gradient |
| 8 official runs, random hash seeds | all outputs **bit-identical**; only the timing varies (see below) |
| Other setups (aLIGO, Voyager, cavity, random UIFOs, 400 frequencies, wideband, second trained UIFO) and the 4 edge cases of the edge-cases bundle (total internal reflection at 45/135/225°, glass) | 20-input check on EPYC (`scripts/gate.sh`, commit `54ff4ae`): ALL criteria PASS vs the original code and a high-accuracy reference (`benchmarks/final/scenarios-check-20.txt`); the edge-cases bundle's own `check_setups.py`: 11/11 setups ok |
| Anything the fast path does not support | falls back to the JAX path automatically (slower, same results); a non-real link derivative raises a clear error instead of returning a wrong gradient |

Known limits (same for the original code): at reflectivity exactly 1.0 some gradients are infinite; very small gradient
entries (~1e-3 of the largest) carry round-off of ~1e-7 × the largest gradient, because the system matrix is extremely
ill-conditioned (cond ≈ 1e30).

**Why repeated runs show different times:** the physics numbers are bit-identical every time. The total time varies by
~±15 % (1.68–2.33 s over 8 runs), almost all of it from Python imports and the first call (disk/cache state, CPU clock,
background processes). Two benchmarks started at the same time share CPU 0 (`--cpus 1` always picks the first CPU).
Measure with nothing else running and report the median of ≥ 5 runs. Details: `FINDINGS.md` §1.10.

## Result
| | Original code | Our `main` |
|---|---:|---:|
| Official `benchmark.py --cpus 1` (organisers' Xeon / same EPYC) | 44.2 s (Xeon) / 72.5 s (EPYC) | **0.83 s** median of 5 on the EPYC (`benchmarks/final/`), ~2 s on our laptop (earlier version) |
| Compile | 22.9 s (Xeon) / 38.5 s (EPYC) | 0.11 s (EPYC) |
| Forward + gradient run | 17.6 s (Xeon) / 25.7 s (EPYC) | 0.27 s (EPYC) |
| Setup (build inputs) | 3.7 s (Xeon) / 7.7 s (EPYC) | 0.095 s (EPYC) |
| Peak memory | ~14.4 GB | < 1 GB |
| Correctness | — | all judging criteria PASS on every run; all 193 gradients within 3.6e-8 (relative to the largest) of the original code; 20 further inputs PASS |

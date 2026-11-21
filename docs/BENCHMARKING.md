# Measuring speed and correctness on a laptop

The merge gate in [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md) §3 is the full
`benchmark.py`, which needs ~15 GB RAM and ~5 min. For day-to-day work on smaller machines use
[`scripts/bench.py`](../scripts/bench.py), then confirm on the cluster with `benchmark.py` before merging
to `main`.

How it stays honest:
- **Nothing is added to `differometor/`**. All timing happens outside the code under test.
- **Each measurement runs in a fresh Python process**, so every compile is a real first compile and memory is isolated.
- **The baseline runs from a git worktree** (`.bench/ref-<sha>`, gitignored), so before/after are on the same machine.
- **CPU pinning:** `--cpus 1` (default) pins to one CPU like the jury.
- **Frequency subset:** `--freqs N` uses N evenly spaced frequencies of the 101 (default 10, about 2.6 GB peak). Compile time barely
  depends on N; run time scales with it. `--freqs 101` is the full workload.

## Commands

```bash
source .venv/bin/activate

# Correctness of your working tree vs the unmodified code (baseline commit `02ec741`, the default `--ref`):
# sensitivities vs the public reference, every output, and ALL 193 gradients vs the baseline.
python scripts/bench.py check
python scripts/bench.py check --fd          # + finite-difference table (informational, see FINDINGS §1.2)
python scripts/bench.py check --ref origin/integration   # compare against another branch instead

# Speed: parts in fresh processes, median of N trials, speedup table vs a baseline ref
python scripts/bench.py parts --ref 02ec741 --trials 3
python scripts/bench.py parts --parts pipeline --ref 02ec741 --trials 3   # scored-equivalent total
python scripts/bench.py parts --parts solve --freqs 4                           # fast kernel iteration

# Compare two saved runs
python scripts/bench.py compare .bench/parts-<base>-f10.json .bench/parts-<cand>-f10.json

# One-process stage profile, optional HLO dump
python scripts/profile_stages.py --freqs 10 --cpus 1 --dump-hlo /tmp/hlo
```

## Parts and which workstream should watch them

| Part | Times | Watch it for |
|---|---|---|
| `pipeline` | the exact `benchmark.py` stages: forward+residuals, sensitivity, gradient, transfer. Prints a scored-equivalent total | **everyone**: this is the number that matters |
| `forward` | `jit(simulate)` lower/compile/run, arrays as jit args, no AD | WS1, WS3 |
| `build` | imports + `Setup.from_data` + `run_build_step` | WS4 |
| `assembly` | signal fill through `simulate.update` (+ its VJP) | WS3. Uses internal API, so it reports FAILED once `update` is replaced (expected) |
| `solve` | `simulate.solve` on the real assembled signal matrices (+ VJP), with a backward-error residual | WS1. Uses internal API |

Add a part for your own kernel with the `@part("name")` decorator in `scripts/bench.py`.
A part gets `(T, W, repeats, freqs)`: time stages with `T("label", "compile"|"execute"|"setup", fn)`.
`W` holds the workload (`arrays`, `params`, `ports`, `expected`, `frequencies`, `pairs`).

## Pass/fail thresholds used by `check`
- FAIL: `|Δln S|` vs the public reference > 1e-5 (the official rule, on the subset).
- FAIL: `|Δln S|` vs the baseline > 5e-6.
- FAIL: carrier, signal, noise or powers differ by > 1e-5 relative (norm-wise).
- FAIL: gradient relative error > 1e-3. Components below 1e-6 × max|g| are treated as ~0.
- FAIL: any non-finite output, or an all-zero gradient.
- WARN at 1e-8 (outputs) and 1e-6 (gradient).

These are stricter than the official checks for outputs and looser than the plan's suggested gradient
threshold (1e-4); tighten if needed.

## Noise
Identical code varies by up to ~20% per stage between fresh runs. Use `--trials 3`, and report ratios
measured on one machine. Log results in [`docs/FINDINGS.md`](FINDINGS.md) §4.

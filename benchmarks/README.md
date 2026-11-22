# Benchmark reports

All reports were produced by the unchanged `benchmark.py --cpus 1` (JSON + five-bar plot). The machine is an AMD
EPYC 7763 pinned to one CPU unless the file name says otherwise. The node is shared, so totals vary by about ±0.1–0.2 s;
compare medians.

## Final submission (`main`, commit 54ff4ae; outputs bit-identical across all runs)

| File | What |
|---|---|
| `final/epyc-final-xla-flags-run{1..5}.json/.png` | **final code, default settings (XLA compile options on)**: 0.822 / 0.820 / 0.833 / 0.836 / 0.830 s, **median 0.83 s** |
| `final/epyc-final-default-run{1..3}.json/.png` | final code with `DIFFEROMETOR_XLA_OPT0=0` (no XLA options): 1.61 / 1.51 / 1.58 s (node load ~52; 1.24 s on a quieter node earlier) |
| `final/epyc-original-02ec741.json/.png` | the unmodified original code on the same machine: 72.5 s |
| `final/scenarios-check-20.txt` | judging criteria on 20 further inputs (other setups, sweeps, perturbed values, the edge-cases bundle) vs the original code and a high-accuracy reference: ALL PASS |
| `final/test_robustness.log`, `final/test_generalisation.log` | robustness tests of the final code |
| `final/edge-cases-bundle-check.txt` | the edge-cases bundle's own check (`check_setups.py`, unmodified-code reference, one CPU): 11/11 setups ok |
| `cpu-baseline.json/.png` | the organisers' baseline (Intel Xeon 8468): 44.2 s |

Stage medians of the final runs: imports 0.35 s, setup 0.095 s, compile 0.11 s, forward 0.16 s, gradient 0.11 s
(original: 0.66 / 7.68 / 38.46 / 20.34 / 5.32 s).

## History (intermediate branches, kept for the record)

| Files | Stage |
|---|---|
| `epyc-main-*` | JAX-native integrated version (WS1–WS4, ~20 s) |
| `epyc-host-native-*` | whole simulation as NumPy/SciPy host code (~1.9 s) |
| `epyc-prime-*`, `epyc-prime-new-*`, `epyc-prime-new2-*` | successive tuning rounds (1.47 → 1.24 → 1.16 s) |
| `new2/` | `the-prime-model-new2` before merging `main`: 16-scenario timings and checks |
| (git history of `final/`) | final code before the faster setup (commit 57a0e25): 0.92 s median of 5 |

Every number and its context: [`../docs/FINDINGS.md`](../docs/FINDINGS.md).

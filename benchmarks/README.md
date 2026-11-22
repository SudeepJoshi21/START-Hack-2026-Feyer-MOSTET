# Benchmark reports

All reports were produced by the unchanged `benchmark.py --cpus 1` (JSON + five-bar plot). The machine is an AMD
EPYC 7763 pinned to one CPU unless the file name says otherwise. The node is shared, so totals vary by about ±0.1–0.2 s;
compare medians.

## Final submission (`main`, commit 57a0e25; outputs bit-identical across all runs)

| File | What |
|---|---|
| `final/epyc-final-xla-flags-run{1..5}.json/.png` | **final code, default settings (XLA compile options on)**: 1.049 / 0.917 / 1.015 / 0.903 / 0.916 s, **median 0.92 s** |
| `final/epyc-final-default-run1.json/.png` | final code with `DIFFEROMETOR_XLA_OPT0=0` (no XLA options): 1.24 s |
| `final/epyc-original-02ec741.json/.png` | the unmodified original code on the same machine: 72.5 s |
| `final/scenarios-check-20.txt` | judging criteria on 20 further inputs (other setups, sweeps, perturbed values, the edge-cases bundle) vs the original code and a high-accuracy reference: ALL PASS |
| `final/test_robustness.log`, `final/test_generalisation.log` | robustness tests of the final code |
| `cpu-baseline.json/.png` | the organisers' baseline (Intel Xeon 8468): 44.2 s |

Stage medians of the final runs: imports 0.41 s, setup 0.12 s, compile 0.11 s, forward 0.17 s, gradient 0.12 s
(original: 0.66 / 7.68 / 38.46 / 20.34 / 5.32 s).

## History (intermediate branches, kept for the record)

| Files | Stage |
|---|---|
| `epyc-main-*` | JAX-native integrated version (WS1–WS4, ~20 s) |
| `epyc-host-native-*` | whole simulation as NumPy/SciPy host code (~1.9 s) |
| `epyc-prime-*`, `epyc-prime-new-*`, `epyc-prime-new2-*` | successive tuning rounds (1.47 → 1.24 → 1.16 s) |
| `new2/` | `the-prime-model-new2` before merging `main`: 16-scenario timings and checks |

Every number and its context: [`../docs/FINDINGS.md`](../docs/FINDINGS.md).

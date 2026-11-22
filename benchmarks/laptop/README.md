# Laptop benchmark reports (one machine)

AMD Ryzen AI 7 350 laptop, 16 GB RAM, WSL2 with 7 GB available, `--cpus 1`. Official `benchmark.py` / `benchmark_aligo.py` (unchanged), JSON + plot per run.
The original code cannot run the main benchmark in the 7 GB available to WSL2 (needs ~14 GB); see PITCH.md.

| File | First-run total | Commit | XLA options | Max \|Δ ln S\| |
|---|---:|---|---|---:|
| `aligo-main-run1.json` | 1.279 s | `d2548da` | on | 3.78e-11 |
| `aligo-main-run2.json` | 1.427 s | `d2548da` | on | 3.78e-11 |
| `aligo-main-run3.json` | 1.512 s | `d2548da` | on | 3.78e-11 |
| `aligo-original.json` | 71.941 s | `02ec741` | off | 1.23e-10 |
| `grid-main-noflags-run1.json` | 2.522 s | `d2548da` | off | 5.37e-06 |
| `grid-main-noflags-run2.json` | 2.464 s | `d2548da` | off | 5.37e-06 |
| `grid-main-run1.json` | 4.728 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run2.json` | 2.131 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run3.json` | 1.931 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run4.json` | 1.886 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run5.json` | 1.825 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run6.json` | 1.808 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run7.json` | 1.748 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run8.json` | 1.802 s | `d2548da` | on | 5.37e-06 |
| `grid-main-run9.json` | 1.818 s | `d2548da` | on | 5.37e-06 |
| `grid-purejax-f5ea168.json` | 28.820 s | `f5ea168` | off | 5.37e-06 |

`grid-main-run1..9`, `grid-main-noflags-run1..2` and `aligo-main-run1..3` are the current `main` (commit `d2548da`): main benchmark median
1.82 s (run 1 was a cold start), aLIGO median 1.43 s. `grid-purejax-f5ea168` is our earlier pure-JAX version. `aligo-original` is
the unmodified code (`02ec741`) with the organisers' `benchmark_aligo.py`, run once.

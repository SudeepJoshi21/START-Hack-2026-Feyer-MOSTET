# Laptop benchmark reports (one machine)

AMD Ryzen AI 7 350 laptop, 16 GB RAM, WSL2 with 7 GB available, `--cpus 1`. Official `benchmark.py` / `benchmark_aligo.py` (unchanged), JSON + plot per run.
The original code cannot run the main benchmark in the 7 GB available to WSL2 (needs ~14 GB); see PITCH.md.

| File | First-run total | Commit | XLA options | Max \|Δ ln S\| |
|---|---:|---|---|---:|
| `aligo-main-run1.json` | 1.292 s | `dbc2288` | on | 3.78e-11 |
| `aligo-main-run2.json` | 1.334 s | `dbc2288` | on | 3.78e-11 |
| `aligo-original.json` | 71.941 s | `02ec741` | off | 1.23e-10 |
| `grid-main-noflags-run1.json` | 2.576 s | `7804f23` | off | 5.37e-06 |
| `grid-main-noflags-run2.json` | 3.058 s | `7804f23` | off | 5.37e-06 |
| `grid-main-run1.json` | 1.978 s | `7804f23` | on | 5.37e-06 |
| `grid-main-run2.json` | 2.089 s | `7804f23` | on | 5.37e-06 |
| `grid-main-run3.json` | 2.346 s | `7804f23` | on | 5.37e-06 |
| `grid-main-run4.json` | 2.251 s | `7804f23` | on | 5.37e-06 |
| `grid-main-run5.json` | 2.963 s | `7804f23` | on | 5.37e-06 |
| `grid-main-run6.json` | 2.556 s | `dbc2288` | on | 5.37e-06 |
| `grid-main-run7.json` | 1.781 s | `dbc2288` | on | 5.37e-06 |
| `grid-main-run8.json` | 1.758 s | `dbc2288` | on | 5.37e-06 |
| `grid-main-run9.json` | 1.820 s | `ddbeaa1` | on | 5.37e-06 |
| `grid-purejax-f5ea168.json` | 28.820 s | `f5ea168` | off | 5.37e-06 |

Main-benchmark runs `grid-main-run1..9` share the same simulator code (`7804f23`/`dbc2288`/`ddbeaa1`: later commits changed docs and the
organisers' harness only): median 2.09 s. `grid-purejax-f5ea168` is our earlier pure-JAX version. `aligo-original` is the
unmodified code (`02ec741`) with the organisers' `benchmark_aligo.py`, run once.

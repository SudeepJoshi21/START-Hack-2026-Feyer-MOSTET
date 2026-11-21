# Fact-check of the research report "Batched fixed-pivot LU refactorisation within SciPy 1.18 limits"

Branch `the-prime-model-new2`. Every claim below was tested on this cluster (AMD EPYC 7763, one pinned CPU, the
benchmark's 751×751 signal systems at 101 frequencies, plus Voyager and aLIGO). Scripts in this folder reproduce
each line (`python <script> <repo>` on a compute node).

**Outcome:** the report's facts about SciPy are correct, but none of its speed-up recommendations pays off for this
workload in pure NumPy/SciPy. Nothing from it was adopted into `differometor/`. The simulator code is unchanged
from `the-prime-model-new` (commit a6d9bb5).

## Claims about SciPy 1.18 (`factcheck_superlu.py`)

| Claim | Verdict | Measurement |
|---|---|---|
| `SuperLU` exposes `L`, `U`, `perm_r`, `perm_c` | ✅ correct | present; the convention is M[perm_r[i], perm_c[j]] = A[i, j] |
| `Fact=SamePattern(_SameRowPerm)` cannot be used through SciPy | ✅ correct | both options **segfault** (exit −11) |
| MC64 row matching not available | ✅ correct | `ValueError: invalid value for 'RowPerm'` |
| (not in report) exported L/U drop exact zeros | ⚠️ pitfall | 60 structural zeros at the reference frequency vanish, and so does cancelled fill. The fill pattern must be computed symbolically. |

## Recommendations

| # | Recommendation | Verdict | Measurement |
|---|---|---|---|
| 1 | Batched fixed-pivot refactorisation with an **exact** partial-pivoting check | ❌ rejected | The reference pivot sequence is a valid partial-pivoting sequence at **0/101** frequencies (`pivot_reuse.py`): multipliers reach \|l\| = 5.5 (benchmark) and 34 (Voyager), and there are 29 distinct row orders over 101 frequencies. The exact check would send every frequency to SuperLU. |
| 1′ | Same, with **threshold** pivoting (\|l\| ≤ 10, UMFPACK/MA48 default 0.1) | ❌ rejected | Accepts 101/101 on the benchmark (81/101 Voyager) and agrees with SuperLU to 5e-8 (`batched_lu.py`, `batched_lu_vs_superlu.py`). But: elimination-program build ~21 ms, batched factorisation 31–40 ms vs ~47–65 ms for 101 SuperLU calls; net gain ≤ ~10 ms, and the pivoting is weaker. The DAG is shallow (87 levels, ~18.6k multiply-adds per system), yet NumPy gathers cost a few ns per element, which is about what SuperLU's C code spends in total. |
| 2 | Level-scheduled triangular solves, batched over frequencies | ❌ rejected | 11–40 ms per solve batch vs 8–16 ms for 101 SuperLU solves (46–84 levels per triangular factor) |
| 3 | Frequency-summed analytic VJP for component functions | ⚠️ not worth it | Only frequency-independent calls benefit (≤ ~5 ms of the 35 ms Jacobian time, measured on `the-prime-model-new`). Full analytic derivatives would mean hand-coding 19 component functions for ~2 %, replacing exact forward-mode AD. |
| 4 | One block-diagonal SuperLU call for all 101 systems | ❌ rejected | Per-call overhead is only 16 µs of 462 µs. Factorisation 64.7 vs 63.5 ms; solve 11.9 vs 8.3 ms (bitwise identical results). |
| 5 | Refinement only where the componentwise backward error ω demands it | ❌ rejected | ω/u is about 1e15 on **every** frequency of every setup (`backward_error.py`): structurally tiny solution components have relative residuals near 1. Nothing would be skipped. Same finding as the earlier adaptive extended-refinement test. |
| 6 | Ruiz equilibration | not pursued | Robustness only; a Ruiz test on Voyager earlier did not improve accuracy (8.5e-3 vs 1.1e-2). |
| 7 | Raw CSC arrays straight to `gstrf` | already done | `adjoint._gstrf`, no `scipy.sparse` import |
| — | JAX import / XLA start-up: nothing new to gain | ✅ agrees | Imports ~0.4–0.5 s, outside our control |
| — | `--xla_cpu_use_thunk_runtime=false` gone in newer JAX | ⚠️ not on JAX 0.9.0 | The flag is still accepted here (earlier A/B: no gain). |

# Integration guide: merging WS3 (assembly) and WS4 (build / compile) into `integration`

**Branch `integration`** = `main` + **WS2** (`ws2-adjoint`, custom-gradient adjoint) + **WS1** (`ws1`, sparse SuperLU),
combined as recommended in [`WS1_WS2_INTEGRATION.md`](WS1_WS2_INTEGRATION.md): WS2's structure with WS1's solver kernel.
Every change and every measurement: [`WS2_CHANGELOG.md`](WS2_CHANGELOG.md) (steps 1 and 2).
Plan: [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md).

**Not merged to `main` yet.** The plan's gate (`benchmark.py --cpus 1` PASS on the cluster at 101 frequencies) has not been
run for this branch. Peak memory ≈ 7.8 GB at 101 frequencies, which is beyond the 7 GB laptop where this was built.

---

## 1. What `integration` contains

| Area | State on `integration` | Owner |
|---|---|---|
| Signal solve + noise weights | one SuperLU factorisation per frequency (host callback); `x = A⁻¹b`, `W = A⁻ᴴC` | WS1 kernel, WS2 wrapper |
| Third (covariance) solve | removed: noise = `Re(wᴴQw)` | WS1 + WS2 |
| Gradient of the solves | `jax.custom_vjp`; backward refactorises; matrix cotangent only on the 2,597-entry pattern; **tape keeps `vals`, `x`, `W` only** | WS2 |
| Noise matrix Q | never built densely; 272 values via `component_outputs` + `segment_sum`; computed **once** when no swept parameter feeds a noise function | WS2 + WS1 idea |
| Matrix assembly | **unchanged** (`vmap(lax.switch)` into dense `(F, 751, 752)`) | **WS3** |
| Carrier solve (286) | **unchanged** (`jnp.linalg.solve`, autodiff) | open |
| `run_build_step` | adds 5 arrays via `adjoint.pattern_arrays` (NumPy) | WS4 may move/speed up |
| Fallbacks | `adjoint.ENABLED = False` gives the original code; `adjoint.SOLVER = "dense"` gives the XLA LU; setups failing the preconditions use the original path automatically | — |

Measured (laptop, 1 CPU, vs original `02ec741`): forward run ~3.6× faster, warm forward+gradient ~2.1× faster, peak
memory 1.4× lower, all 193 gradients within 2e-5. **First-run total is still ~90% compile, and that is assembly (WS3).**

## 2. Interfaces WS3 and WS4 must respect

### 2.1 `simulate` signature
- **Positional arguments are unchanged** from the original. `run_build_step` maps outputs to them by position (`co_argcount`).
- New arguments are **keyword-only** (after `*`), all defaulting to `None` (None = original path):

| Keyword | Shape / dtype | Meaning | Produced by |
|---|---|---|---|
| `signal_pattern_rows`, `signal_pattern_cols` | `(nnz,)` int32, nnz = 2,597 | positions that can be non-zero in the signal matrix A (base non-zeros ∪ `update` writes; RHS column −1 excluded; rows/cols wrapped; sorted, unique) | `adjoint.signal_pattern` |
| `noise_base_rows`, `noise_base_cols` | `(K,)` int32, K = 0 here | base noise-matrix non-zeros not overwritten by `update` | `adjoint.pattern_arrays` |
| `noise_parameter_rows` | `(1,)` or `(R,)` int32 | `[0]` if no swept parameter feeds a noise function, else all parameter rows | `adjoint.pattern_arrays` |

**WS3/WS4: if you add structure arrays, add them as keyword-only arguments** in the same way (or inside a static pytree, FINDINGS §1.1).
Don't append positional arguments: that changes `run_build_step`'s zip and conflicts with every branch.

### 2.2 The adjoint entry point (what WS3's assembly should feed)
```python
signal, noise_weights = adjoint.solve_signal_and_noise_weights(
    values,                 # (F, nnz) complex: A's values on (signal_pattern_rows, signal_pattern_cols)
    signal_pattern_rows,    # (nnz,) int
    signal_pattern_cols,    # (nnz,) int
    b,                      # (F, N) complex: right-hand side (today: signal_matrix[:, :, -1])
    C,                      # (F, D, N) complex: detector selections (today: swapaxes(carrier_selections, 0, 1))
)   # -> x (F, N), W (F, D, N); fully differentiable w.r.t. values, b, C
```
Today `simulate` builds the dense `signal_matrix (F, 751, 752)` and then **gathers** `values = signal_matrix[:, rows, cols]` and `b = signal_matrix[:, :, -1]`.
**WS3's target:** produce `values` and `b` directly from per-type component evaluations, writing into position `k` of
the pattern instead of a dense matrix. Then the dense matrix, its carrier-scaling and conjugate-block `.at[].set` copies,
and the gather all disappear. That is the remaining ~178 MiB of tape at 10 freqs (~1.8 GB at 101) and most of the compile.

Things WS3 must reproduce exactly, all currently applied to the dense matrix before the gather
(`simulate.py`, signal section, lines ~340–365 on `integration`):
1. **Carrier scaling:** at `signal_carrier_placing_indices` (1,004 positions, all inside the pattern),
   `value = −value × carrier[signal_carrier_indices]`.
2. **Conjugate blocks** (`cs` = carrier size = 286): conjugate entries in rows `[cs:2cs]` × cols `[cs:2cs]`, rows `[2cs:]` × cols `[:cs]`,
   and rows `[cs:2cs]` × cols `[2cs:]`. The **last** block includes the RHS column, so the corresponding `b` entries are conjugated too.
   On the pattern this is a static boolean mask: `values = where(mask, conj(values), values)`.
3. **Last write wins:** `update` uses `.at[].set`. The pattern is unique, but if two `update` writes target the same position, the later
   one wins. Check that per-type assembly preserves the order (or assert there are no duplicate writes; there are none in the fixture: 1,893 unique).
4. **RHS through column −1:** 71 `update` writes go to column index −1 (= `b`). Normalise with `mod (N+1)`.

Acceptance test for WS3: densify your `values`/`b` and compare with `signal_matrix[:, :, :-1]` / `signal_matrix[:, :, -1]`
from the current code at a few frequencies (bit-identical expected). `tests/test_adjoint.py` shows the densify helper pattern.

### 2.3 `component_outputs` (new helper in `simulate.py`)
`update(...)` is now `component_outputs(...)` + scatter, with identical behaviour. The adjoint noise path calls
`component_outputs` directly for the 272 noise values. **If WS3 replaces `update` / `lax.switch`, keep a function that
returns the noise values `(rows_of_parameters, M)` aligned with `noise_system_matrix_indices`**, or edit the adjoint branch
of `simulate` accordingly (search for `[WS1+WS2]`).

## 3. Expected merge conflicts and how to resolve them

Merge order suggestion: **WS4 first** (build-side, mostly separate files), **then WS3**.

| File / region | Who touches it | Conflict risk | Resolution |
|---|---|---|---|
| `simulate.py::update` / new `component_outputs` | WS3 rewrites assembly | **high** | keep WS3's assembly; re-provide noise values for the adjoint branch (§2.3) |
| `simulate.py::simulate`, signal section (assembly → carrier scaling → conj blocks) | WS3 | **high** | WS3's per-type assembly should end in `values (F,nnz)`, `b (F,N)` and call `solve_signal_and_noise_weights` (§2.2). Keep the `use_adjoint` branch as the only path that WS3's code feeds; the `else` (original) path may stay dense |
| `simulate.py::simulate` signature | WS3/WS4 adding arrays | medium | keyword-only only (§2.1) |
| `simulate.py::run_build_step` | WS4 (NumPy-only build), WS3 (new structure) | medium | keep the line `simulation_arrays.update(adjoint.pattern_arrays(simulation_arrays))` (or move its NumPy work into WS4's build, same keys) |
| `differometor/adjoint.py` | nobody else | low | — |
| `differometor/__init__.py` (WS4 lazy `plot` import) | WS4 | low | independent |
| `build.py`, `components.py` | WS3/WS4 | none with integration | — |
| `docs/FINDINGS.md` results table | everyone appends rows | low (adjacent appends) | keep all rows |

Useful commands:
```bash
git fetch origin
git switch integration && git pull
git merge --no-ff origin/ws4          # resolve, then validate (§4)
git merge --no-ff origin/ws3          # resolve, then validate (§4)
git diff integration@{1} -- differometor/simulate.py   # review what the merge changed
```

## 4. Validation checklist after each merge (copy into the PR / commit message)

```bash
python tests/test_adjoint.py                                   # 5/5 PASS, both solvers
python scripts/bench.py check --fd                             # PASS; all 193 grads vs 02ec741
python scripts/bench.py parts --parts pipeline --trials 3 --ref origin/integration   # vs pre-merge state
python - <<'EOF'                                               # other setups: adjoint vs original path
import jax.numpy as jnp, numpy as np, differometor as df
from differometor import adjoint, setups
f = jnp.logspace(1, 3.5, 25)
for name in ("aligo", "voyager"):
    S, _ = getattr(setups, name)(); out = {}
    for on in (True, False):
        adjoint.ENABLED = on
        arrays, *_ = df.run_build_step(S, [("f", "frequency")], f); out[on] = df.simulate(**arrays)
    adjoint.ENABLED = True
    print(name, [float(np.max(np.abs(a - b)) / np.max(np.abs(b))) for a, b in zip(out[True], out[False])])
EOF
# cluster (the gate), 101 freqs:
python benchmark.py --cpus 1 --repeats 3 --output benchmark-integration.json --save-arrays integration.npz
```
Expected today (this snippet prints carrier, signal, noise errors relative to the largest value): aLIGO `[0, 5.7e-11, 1.0e-11]`;
Voyager `[0, 1.8e-9, 6.0e-5]`. The Voyager noise is up to **2e-3 at its worst single frequency** with the sparse solver (known, §5).
Log every result as a row in `docs/FINDINGS.md` §4.

## 5. Known issues and open items

1. **Voyager noise, sparse vs dense: 2e-3.** An ill-conditioning effect (|w| spans ~21 orders of magnitude); neither solver is
   accurate to better than ~1e-3 at those entries. Not scored. Details: WS2_CHANGELOG step 2. Decide whether a non-scored setup
   needs `SOLVER="dense"` or a precision study.
2. **10-frequency anomaly:** at 10 freqs the sparse *gradient* run is reproducibly slower than the dense one (0.85–1.06 s vs
   0.65 s), while at 4 and 20 freqs sparse wins both passes. Host callbacks take only ~12 ms. Profile once on the cluster at 101.
3. **Not measured at 101 frequencies.** Run `benchmark.py` on the cluster (gate) before merging to `main`.
4. **Carrier solve (286)** still uses dense autodiff. It's small, and could use the same custom_vjp with its own pattern.
5. **Compile** (~55 s laptop / ~34 s EPYC) is untouched by WS1/WS2: assembly (WS3), static structure (FINDINGS §1.1),
   XLA opt level 0 (FINDINGS §1.7, **needs organiser approval**).

## 6. After WS3 lands: next WS2 step
Once assembly yields `values`/`b` per component type, one `custom_vjp` can cover **assembly + solve + noise**. Its backward is
then a handful of vectorised per-type derivative formulas, so the gradient program (today ~12 s compile) should shrink to seconds.
That needs WS3's per-type functions to expose their parameter derivatives (or `jax.vjp` of each small per-type function).

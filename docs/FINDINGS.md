# Findings addendum + results log

**The plan is [`OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md)** (workstreams WS1–WS4, interfaces,
process). This file only adds what that plan doesn't cover, corrects one point, and holds the shared
**results log** (§4). How to measure on a laptop: [`docs/BENCHMARKING.md`](BENCHMARKING.md).

Laptop numbers below: AMD Ryzen AI 7 350, WSL2, 7 GB RAM, `--cpus 1`, usually 10 of 101 frequencies.
Compare only ratios measured on one machine.

---

## 1. Additions to OPTIMIZATION_PLAN.md

### 1.1 Index *values* can be compile-time constants (WS3/WS4) — measured ~6 s of compile
OPTIMIZATION_PLAN §3 says index values are runtime data because `arrays` is a jit argument. They don't
have to be: jit treats a **registered pytree node's `aux_data` as static**. If `run_build_step` returns the
structure (pattern `rows/cols`, per-type groups, RCM permutation, scatter indices) inside a node like

```python
@jax.tree_util.register_pytree_node_class
class Static:
    def __init__(self, **arrays): self.d = {k: np.asarray(v) for k, v in arrays.items()}
    def tree_flatten(self): return (), self            # no leaves -> nothing traced
    @classmethod
    def tree_unflatten(cls, aux, children): return aux
    def __hash__(self): return hash(tuple((k, v.dtype.str, v.shape, v.tobytes()) for k, v in sorted(self.d.items())))
    def __eq__(self, o): return isinstance(o, Static) and self.__hash__() == o.__hash__()
```

then `dict(arrays, optimized_parameters=p)` still passes it into `simulate(**...)`, but XLA sees
constant gathers/scatters. Measured on the unchanged code: forward+vjp compile **20.4 s with indices as
jit args vs 14.2 s as constants** (10 freqs, laptop). Keep *numeric* inputs (base parameters,
frequencies) as leaves so nothing physical is precomputed outside the timer.

### 1.2 Finite differences are unreliable for most tunings (WS2)
At the benchmark's steps (1e-3, 5e-4), FD for `ml11`, `mt23`, `m43` tunings does not converge (the two
steps disagree by 30–100%); only `mb11` is roughly stable, which is presumably why the benchmark checks
only that one. **Validate gradients against the baseline's AD gradient on all 193 entries**
(`scripts/bench.py check` does this), not against FD.

### 1.3 Timing noise is ~±20% per stage
Same code, two fresh runs: forward compile 43.2 s vs 36.1 s, first-run total 70.0 vs 64.1 s. Use
`bench.py parts --trials 3` and don't trust single-stage gains < ~1.2× without repeating.

### 1.4 Laptop vs EPYC vs Xeon stage split (unchanged code)

| Stage | Xeon 8468, 101 f (README) | EPYC 7763, 101 f (plan) | Ryzen laptop, 10 f |
|---|---:|---:|---:|
| imports + build | 3.6 | 1.9 | 3.2–3.8 |
| forward lower+compile | 17.0 | 27.1 | 38–49 |
| forward run | 13.3 | 22.5 | 5.2–5.6 |
| gradient lower+compile | 5.6 | 9.7 | 12–14 |
| gradient run | 4.3 | 4.8 | 1.3–1.7 |
| **total** | **44.2** | **66.8** | **64–77** |

The laptop compiles 2–3× slower than the Xeon; the jury's machine is relatively *run*-heavy, so WS1/WS2
run-time wins matter at least as much as compile wins there.

### 1.5 Assembly-only compile, isolated (supports WS3)
Signal fill `vmap(update)` alone: compile 4.1–4.9 s, its VJP 8.4–10.5 s; dense `vmap(solve)` + VJP:
~1.3 s total. Benchmark-style forward (returning residuals) vs `simulate` under `vjp` with residuals
dropped: 36–49 s vs 20 s compile, so the **residual outputs themselves** roughly double forward cost
(supports WS2's custom_vjp keeping residuals small).

## 2. Correction to my earlier notes
- Noise matrix `Q`: **66 component-function calls producing 272 non-zeros** (the plan's number is right;
  I had written "66 non-zeros").
- Plotly / `import differometor` is negligible (~15–80 ms after jax); lazy-importing `plot` is not worth
  much on its own.

## 3. Baseline correctness margin
Unmodified code: max `|Δln S|` = 5.37e-6 (Xeon and EPYC, 101 f), 3.8e-6 (laptop, 10 f), against the 1e-5
limit. `bench.py check` fails a candidate whose sensitivity drifts > 5e-6 (in ln) from the *baseline*.

## 4. Results log (append one row per measured change)

| Date | Branch / commit | Change | Machine, freqs | First-run total | Compile | Warm median | Check | Notes |
|---|---|---|---|---:|---:|---:|---|---|
| 2026-11-16 | upstream | baseline (organisers) | Xeon 8468, 101 | 44.20 | 22.92 | 17.68 | PASS | `benchmarks/cpu-baseline.json` |
| 2026-11-21 | upstream | baseline (plan §1) | EPYC 7763, 101 | 66.79 | 37.61 | ~24.97 | PASS | `OPTIMIZATION_PLAN.md` |
| 2026-11-21 | upstream vs itself | noise calibration (`bench.py parts --ref 02ec741`) | Ryzen laptop, 10 | 70.0 / 64.1 | 60.3 / 53.6 | 4.58 / 4.46 | PASS (0 diff) | same code, ~9% apart |

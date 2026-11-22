#!/usr/bin/env python3
"""Laptop-friendly benchmarking + correctness harness for the Faster Physics challenge.

Nothing here touches the timed code: every measurement is taken from outside
`differometor/`, each part runs in a FRESH Python process (clean first-run compile,
isolated memory), and baselines are run from a separate git worktree so speedups are
always measured on the same machine.

Subcommands
-----------
  parts    time named parts (build / forward / pipeline / kernels) -> JSON + table
  check    correctness of the current tree vs a baseline ref (all outputs, all 193 grads)
  compare  speedup table between two `parts` JSON files

Examples
--------
  python scripts/bench.py parts                          # all default parts, 10 freqs, 1 CPU
  python scripts/bench.py parts --ref 02ec741             # + same parts on unmodified baseline, speedup table
  python scripts/bench.py parts --parts forward,pipeline --freqs 20 --repeats 5
  python scripts/bench.py check                          # vs unmodified baseline 02ec741, 10 freqs
  python scripts/bench.py check --fd                     # + finite-difference spot checks
  python scripts/bench.py compare .bench/a.json .bench/b.json

`--freqs 101` reproduces the full workload (needs ~15 GB RAM; use the grid).
The official score is still `python benchmark.py --cpus 1` -- this is a dev tool.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCH_DIR = REPO / ".bench"
DEFAULT_REF = "02ec741d8a1bf1a2278294ea74f9fc1d68f08f1f"  # unmodified organiser commit (in every clone's history)
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
DEFAULT_PARTS = ["build", "forward", "pipeline", "assembly", "solve"]


# =============================================================================
# Worker side (runs in a fresh process; imports JAX lazily after env setup)
# =============================================================================

def _setup_env(cpus: int | None):
    if cpus:
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:cpus])
        for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[n] = str(cpus)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"


class Timer:
    def __init__(self):
        from time import perf_counter
        self.now = perf_counter
        self.rows: list[tuple[str, str, float]] = []  # (stage, category, seconds)

    def __call__(self, stage, category, fn):
        import jax
        t = self.now()
        out = fn()
        jax.block_until_ready(out)
        self.rows.append((stage, category, self.now() - t))
        return out


def _load_workload(freqs: int):
    """Same inputs as benchmark.py, restricted to `freqs` evenly spaced frequencies."""
    import numpy as np
    import jax.numpy as jnp
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces

    data = REPO / "examples" / "data"
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
    idx = np.unique(np.linspace(0, len(reference["frequencies"]) - 1, freqs).round().astype(int))
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    comps = fixture["nodes"] | fixture["edges"]
    values = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        values.append(comps[name]["properties"][prop])
    params = jnp.asarray(values, dtype=jnp.float64)
    frequencies = jnp.asarray(np.asarray(reference["frequencies"])[idx], dtype=jnp.float64)
    expected = jnp.asarray(np.asarray(reference["sensitivities"])[idx], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays = dict(arrays)
    arrays["optimized_parameters"] = params
    return dict(arrays=arrays, ports=jnp.asarray(ports), params=params, expected=expected,
                frequencies=frequencies, pairs=pairs, freq_index=idx)


# Benchmark-identical staged functions -------------------------------------------------

def _fwd(p, a):
    import jax
    import differometor as df
    return jax.vjp(lambda q: df.simulate(**dict(a, optimized_parameters=q)), p)


def _readout(outs, ports, ref):
    import jax.numpy as jnp
    import differometor as df
    c, s, n = outs
    sig = df.signal_detector(c, s)[ports]
    det = sig.squeeze() if ports.shape[0] == 1 else sig[0] - sig[1]
    sens = (n / jnp.abs(det)).squeeze()
    return jnp.mean(jnp.log10(sens / ref)), (sens, df.power_detector(c))


def _sens(outs, ports, ref):
    import jax
    return jax.vjp(lambda v: _readout(v, ports, ref), outs, has_aux=True)


def _bwd(sp, rp, ct):
    return sp(rp(ct)[0])[0]


def _compile(T, label, fn, *args):
    import jax
    low = T(f"{label}.lower", "compile", lambda: jax.jit(fn).lower(*args))
    return T(f"{label}.compile", "compile", low.compile)


def _run_pipeline(W, T, repeats):
    """Exactly the benchmark's scored stages; returns outputs + warm samples."""
    import jax
    import jax.numpy as jnp
    a, ports, p, ref = W["arrays"], W["ports"], W["params"], W["expected"]
    one = jnp.asarray(1.0, dtype=jnp.float64)
    F = _compile(T, "forward", _fwd, p, a)
    outs, spb = T("forward.run", "execute", lambda: F(p, a))
    S = _compile(T, "sensitivity", _sens, outs, ports, ref)
    loss, rpb, (sens, powers) = T("sensitivity.run", "execute", lambda: S(outs, ports, ref))
    G = _compile(T, "gradient", _bwd, spb, rpb, one)
    grad = T("gradient.run", "execute", lambda: G(spb, rpb, one))
    host = T("host_transfer", "execute", lambda: jax.device_get((outs, sens, powers, loss, grad)))
    warm = []
    for _ in range(repeats):
        del spb, rpb
        t = T.now()
        outs, spb = F(p, a)
        loss, rpb, (sens, powers) = S(outs, ports, ref)
        grad = G(spb, rpb, one)
        jax.block_until_ready(jax.device_get((outs, sens, powers, loss, grad)))
        warm.append(T.now() - t)
    return host, warm


# Parts ---------------------------------------------------------------------------------
# Each part: fn(T, W, repeats) -> dict(extra info, optional "warm": [...]).
# Stage rows are recorded on T. Add your own part with @part("name").

PARTS = {}


def part(name, needs_workload=True):
    def deco(fn):
        PARTS[name] = (fn, needs_workload)
        return fn
    return deco


@part("build", needs_workload=False)
def part_build(T, W, repeats, freqs):
    W = T("build", "setup", lambda: _load_workload(freqs))
    return {}


@part("forward")
def part_forward(T, W, repeats, freqs):
    """simulate() forward only, arrays passed as jit args (benchmark style), no AD residuals."""
    import jax
    import differometor as df
    fn = lambda p, a: df.simulate(**dict(a, optimized_parameters=p))
    C = _compile(T, "forward", fn, W["params"], W["arrays"])
    T("forward.run", "execute", lambda: C(W["params"], W["arrays"]))
    warm = []
    for _ in range(repeats):
        t = T.now(); jax.block_until_ready(C(W["params"], W["arrays"])); warm.append(T.now() - t)
    return {"warm": warm}


@part("pipeline")
def part_pipeline(T, W, repeats, freqs):
    """benchmark.py's scored stages (forward+residuals, sensitivity, gradient, transfer)."""
    _, warm = _run_pipeline(W, T, repeats)
    return {"warm": warm}


def _expanded_params(W, which):
    """Parameter rows exactly as simulate() feeds them to update() (internal API)."""
    import jax.numpy as jnp
    from jax import vmap, lax
    from differometor.simulate import expand_parameters
    from differometor.build import LINKING_FUNCTIONS
    a = W["arrays"]
    P = a["parameters"].at[[0], a["optimized_parameter_indices"]].set(W["params"][a["optimized_value_indices"]])
    out = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(a["linking_function_indices"], P[0, a["linked_indices"]])
    P = P.at[[0], a["indices_to_link"]].set(out)
    P = expand_parameters(P, a["carrier_changing_parameter_indices"], a["carrier_changing_values"])
    if which == "carrier":
        return P
    return expand_parameters(P, a["signal_changing_parameter_indices"], a["signal_changing_values"])


@part("assembly")
def part_assembly(T, W, repeats, freqs):
    """KERNEL (internal API, may break after refactors): signal-matrix fill + its VJP."""
    import jax
    import jax.numpy as jnp
    from jax import vmap
    from differometor.simulate import update
    a = W["arrays"]
    P = _expanded_params(W, "signal")
    fill = lambda P: vmap(update, in_axes=(0, None, None, None, None, None))(
        P, a["signal_matrix"], a["signal_function_input_indices"], a["signal_output_indices"],
        a["signal_matrix_indices"], a["signal_function_indices"])
    C = _compile(T, "fill", fill, P)
    out = T("fill.run", "execute", lambda: C(P))
    vjp = lambda P, ct: jax.vjp(fill, P)[1](ct)[0]
    ct = jnp.ones_like(out)
    CV = _compile(T, "fill_vjp", vjp, P, ct)
    T("fill_vjp.run", "execute", lambda: CV(P, ct))
    warm = []
    for _ in range(repeats):
        t = T.now(); jax.block_until_ready(C(P)); warm.append(T.now() - t)
    return {"warm": warm, "entries": int(a["signal_function_indices"].shape[0])}


@part("solve")
def part_solve(T, W, repeats, freqs):
    """KERNEL: batched signal solve on the REAL assembled signal matrices, fwd + VJP.

    Swap `differometor.simulate.solve` for your solver to compare (same signature).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import vmap
    import differometor.simulate  # noqa: F401
    sm = sys.modules["differometor.simulate"]
    a = W["arrays"]
    # Capture the real assembled signal matrices via an un-timed forward pass.
    cap, orig = [], sm.solve
    def spy(M, b):
        jax.debug.callback(lambda m, bb: cap.append((np.asarray(m), np.asarray(bb))), M, b)
        return orig(M, b)
    sm.solve = spy
    try:
        jax.block_until_ready(sm.simulate(**dict(a, optimized_parameters=W["params"])))
        jax.effects_barrier()
    finally:
        sm.solve = orig
    n = a["signal_matrix"].shape[0]
    sig = [x for x in cap if x[0].shape[-1] == n][: len(W["frequencies"])]
    M = jnp.asarray(np.stack([m for m, _ in sig])); b = jnp.asarray(np.stack([v for _, v in sig]))
    C = _compile(T, "solve", lambda M, b: vmap(sm.solve)(M, b), M, b)
    x = T("solve.run", "execute", lambda: C(M, b))
    vjp = lambda M, b, ct: jax.vjp(lambda M, b: vmap(sm.solve)(M, b), M, b)[1](ct)
    CV = _compile(T, "solve_vjp", vjp, M, b, x)
    T("solve_vjp.run", "execute", lambda: CV(M, b, x))
    warm = []
    for _ in range(repeats):
        t = T.now(); jax.block_until_ready(C(M, b)); warm.append(T.now() - t)
    # Backward-error style residual: max|Ax-b| / (max|A| max|x| + max|b|); ~1e-15 is perfect.
    resid = float(jnp.max(jnp.abs(jnp.einsum("bij,bj->bi", M, x) - b))
                  / (jnp.max(jnp.abs(M)) * jnp.max(jnp.abs(x)) + jnp.max(jnp.abs(b))))
    return {"warm": warm, "matrix_shape": list(M.shape), "relative_residual": resid,
            "nnz_per_matrix": int(np.count_nonzero(np.asarray(M[0])))}


def worker_parts(args):
    from time import perf_counter
    t0 = perf_counter()
    _setup_env(args.cpus)
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import resource
    import jax
    import differometor as df
    t_import = perf_counter() - t0
    jax.config.update("jax_enable_compilation_cache", False)
    T = Timer()
    T.rows.append(("imports", "setup", t_import))
    fn, needs = PARTS[args.part]
    if args.part in ("assembly", "solve"):          # these parts time pieces of the JAX path: keep its arrays
        from differometor import host as _host
        if hasattr(_host, "JAX_PATH_ARRAYS"):
            _host.JAX_PATH_ARRAYS = True
    W = T("build", "setup", lambda: _load_workload(args.freqs)) if needs else None
    info = fn(T, W, args.repeats, args.freqs) or {}
    warm = info.pop("warm", [])
    res = {
        "part": args.part, "repo": str(Path(args.repo).resolve()), "differometor": df.__file__,
        "freqs": args.freqs, "cpus": args.cpus,
        "stages": [{"stage": s, "category": c, "seconds": x} for s, c, x in T.rows],
        "first_run_total": sum(x for _, _, x in T.rows),
        "compile_total": sum(x for _, c, x in T.rows if c == "compile"),
        "warm": warm, "warm_median": statistics.median(warm) if warm else None,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "info": info,
    }
    Path(args.out).write_text(json.dumps(res))


def worker_outputs(args):
    """Compute all physical outputs + gradient (benchmark-style) and save .npz."""
    _setup_env(args.cpus)
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import numpy as np
    import jax
    import differometor as df
    jax.config.update("jax_enable_compilation_cache", False)
    W = _load_workload(args.freqs)
    T = Timer()
    (outs, sens, powers, loss, grad), _ = _run_pipeline(W, T, 0)
    extra = {}
    if args.fd:
        # Central differences on a few parameters, same loss as the benchmark (on this subset).
        a, ports, ref, p = W["arrays"], W["ports"], W["expected"], W["params"]
        L = jax.jit(lambda q: _readout(df.simulate(**dict(a, optimized_parameters=q)), ports, ref)[0])
        # Only tuning parameters (degrees): safe to perturb, like benchmark.py's mb11.tuning check.
        names = [x if not isinstance(x[0], list) else x[0] for x in W["pairs"]]
        tunings = [i for i, nm in enumerate(names) if nm[1] == "tuning"]
        pick = [tunings[k] for k in np.linspace(0, len(tunings) - 1, args.fd_count).astype(int)]
        idxs = sorted({W["pairs"].index(["mb11", "tuning"])} | set(pick))
        steps = (1e-3, 5e-4)  # same steps as benchmark.py
        fd = np.array([[(float(L(p.at[i].add(h))) - float(L(p.at[i].add(-h)))) / (2 * h) for h in steps] for i in idxs])
        extra = {"fd_index": np.array(idxs), "fd_value": fd, "fd_steps": np.array(steps)}
    np.savez(args.out, carrier=outs[0], signal=outs[1], noise=outs[2], sensitivity=sens, powers=powers,
             loss=loss, gradient=grad, expected=np.asarray(W["expected"]), params=np.asarray(W["params"]),
             frequencies=np.asarray(W["frequencies"]), pair_names=np.array([json.dumps(x) for x in W["pairs"]]),
             **extra)


# =============================================================================
# Driver side
# =============================================================================

def _git(*a, cwd=REPO):
    return subprocess.check_output(["git", "-C", str(cwd), *a], text=True, stderr=subprocess.DEVNULL).strip()


def _ref_worktree(ref: str) -> Path:
    """Detached worktree of `ref` under .bench/ (reused if it exists)."""
    sha = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    path = BENCH_DIR / f"ref-{sha[:10]}"
    if not (path / "differometor").is_dir():
        BENCH_DIR.mkdir(exist_ok=True)
        subprocess.check_call(["git", "-C", str(REPO), "worktree", "add", "--detach", "-f", str(path), sha],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path


def _label(repo: Path) -> str:
    try:
        sha = _git("rev-parse", "--short", "HEAD", cwd=repo)
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", "differometor", cwd=repo))
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return repo.name


def _spawn(mode, repo, extra, out):
    cmd = [sys.executable, str(Path(__file__).resolve()), mode, "--repo", str(repo), "--out", str(out), *extra]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(l for l in r.stderr.splitlines() if "WARNING" not in l)[-2000:]
        return None, tail
    return out, None


def _median_trials(trials: list[dict]) -> dict:
    """Combine fresh-process trials of one part: per-stage median, all samples kept."""
    r = dict(trials[0])
    stages = []
    for i, st in enumerate(trials[0]["stages"]):
        xs = [t["stages"][i]["seconds"] for t in trials]
        stages.append({**st, "seconds": statistics.median(xs), "trials": xs})
    r["stages"] = stages
    r["compile_total"] = statistics.median(t["compile_total"] for t in trials)
    r["first_run_total"] = statistics.median(t["first_run_total"] for t in trials)
    r["first_run_total_trials"] = [t["first_run_total"] for t in trials]
    warm = [w for t in trials for w in t["warm"]]
    r["warm"], r["warm_median"] = warm, (statistics.median(warm) if warm else None)
    r["peak_rss_mb"] = max(t["peak_rss_mb"] for t in trials)
    r["trials"] = len(trials)
    return r


def run_parts(repo: Path, parts, a) -> dict:
    results = {"label": _label(repo), "repo": str(repo), "freqs": a.freqs, "cpus": a.cpus,
               "trials": a.trials, "parts": {}}
    for name in parts:
        extra = ["--part", name, "--freqs", str(a.freqs), "--repeats", str(a.repeats)]
        if a.cpus:
            extra += ["--cpus", str(a.cpus)]
        trials, err = [], None
        for t in range(a.trials):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                out = Path(f.name)
            print(f"  [{results['label']}] {name} trial {t + 1}/{a.trials} ...", end="", flush=True)
            ok, err = _spawn("_worker_parts", repo, extra, out)
            if ok:
                r = json.loads(out.read_text()); trials.append(r)
                print(f" total {r['first_run_total']:.2f}s, compile {r['compile_total']:.2f}s, peak {r['peak_rss_mb']:.0f} MB")
            else:
                print(" FAILED (part skipped):\n    " + (err.splitlines()[-1] if err else "?"))
            out.unlink(missing_ok=True)
            if not ok:
                break
        results["parts"][name] = _median_trials(trials) if trials and not err else {"error": err}
    return results


def print_parts(res):
    print(f"\n== {res['label']}  (freqs={res['freqs']}, cpus={res['cpus']}) ==")
    for name, r in res["parts"].items():
        if "error" in r:
            print(f"\n[{name}] FAILED"); continue
        print(f"\n[{name}]  peak RSS {r['peak_rss_mb']:.0f} MB" + (f"  info {r['info']}" if r["info"] else ""))
        for s in r["stages"]:
            print(f"   {s['stage']:24s} {s['category']:8s} {s['seconds']:9.3f} s")
        if r["warm"]:
            print(f"   {'warm median':24s} {'':8s} {r['warm_median']:9.3f} s  (n={len(r['warm'])})")
        if name == "pipeline":
            tr = r.get("first_run_total_trials", [r["first_run_total"]])
            print(f"   {'SCORED-EQUIVALENT TOTAL':24s} {'':8s} {r['first_run_total']:9.3f} s" +
                  (f"  (trials: {', '.join(f'{x:.2f}' for x in tr)})" if len(tr) > 1 else ""))


def _metrics(r):
    if "error" in r:
        return {}
    m = {s["stage"]: s["seconds"] for s in r["stages"]}
    if r["compile_total"] > 0:
        m["compile_total"] = r["compile_total"]
    m["first_run_total"] = r["first_run_total"]
    if r["warm_median"] is not None:
        m["warm_median"] = r["warm_median"]
    m["peak_rss_mb"] = r["peak_rss_mb"]
    return m


def print_compare(base, cand):
    print(f"\n== speedup: {base['label']} (baseline) -> {cand['label']} (candidate), freqs={cand['freqs']}, "
          f"trials={cand.get('trials', 1)} ==")
    print("   NOTE: identical code varies up to ~20% per stage between runs. Use --trials 3+ and\n"
          "   don't trust speedups below ~1.2x on single stages without repeating.")
    print(f"{'part.metric':38s} {'baseline':>10s} {'candidate':>10s} {'speedup':>8s}")
    for name in cand["parts"]:
        mb, mc = _metrics(base["parts"].get(name, {"error": 1})), _metrics(cand["parts"][name])
        if not mb or not mc:
            print(f"{name:38s} {'n/a (part failed on one side)':>30s}"); continue
        for k in mc:
            if k in mb and mb[k] > 0:
                unit = "MB" if k == "peak_rss_mb" else "s"
                sp = mb[k] / mc[k] if mc[k] > 0 else float("inf")
                star = "  <==" if k in ("first_run_total", "warm_median", "compile_total") else ""
                print(f"{name + '.' + k:38s} {mb[k]:9.3f}{unit[0]} {mc[k]:9.3f}{unit[0]} {sp:7.2f}x{star}")


def cmd_parts(a):
    parts = a.parts.split(",") if a.parts else DEFAULT_PARTS
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        sys.exit(f"unknown parts {unknown}; available: {sorted(PARTS)}")
    BENCH_DIR.mkdir(exist_ok=True)
    print(f"Candidate: {REPO}")
    cand = run_parts(REPO, parts, a)
    print_parts(cand)
    out = Path(a.out or BENCH_DIR / f"parts-{cand['label']}-f{a.freqs}.json")
    out.write_text(json.dumps(cand, indent=2))
    print(f"\nsaved {out}")
    if a.ref:
        ref = _ref_worktree(a.ref)
        print(f"\nBaseline: {a.ref} at {ref}")
        base = run_parts(ref, parts, a)
        bout = BENCH_DIR / f"parts-{base['label']}-f{a.freqs}.json"
        bout.write_text(json.dumps(base, indent=2))
        print_compare(base, cand)
        print(f"\nsaved {bout}")


def cmd_compare(a):
    print_compare(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.candidate).read_text()))


def _outputs(repo, a, cache: bool):
    import hashlib
    key = hashlib.sha1(f"{_label(repo)}-{a.freqs}-{a.fd}-{a.fd_count}".encode()).hexdigest()[:10]
    out = BENCH_DIR / f"outputs-{_label(repo)}-f{a.freqs}-{key}.npz"
    if cache and out.exists() and "dirty" not in out.name:
        print(f"  using cached {out.name}")
        return out
    extra = ["--freqs", str(a.freqs)] + (["--cpus", str(a.cpus)] if a.cpus else [])
    if a.fd:
        extra += ["--fd", "--fd-count", str(a.fd_count)]
    print(f"  computing outputs for {_label(repo)} ...", flush=True)
    ok, err = _spawn("_worker_outputs", repo, extra, out)
    if not ok:
        sys.exit(f"output computation failed for {repo}:\n{err}")
    return out


def cmd_check(a):
    import numpy as np
    BENCH_DIR.mkdir(exist_ok=True)
    ref_repo = _ref_worktree(a.ref)
    base = np.load(_outputs(ref_repo, a, cache=True))
    cand = np.load(_outputs(REPO, a, cache=False))
    fails, warns = [], []

    def rel(x, y):
        """Norm-wise relative error max|x-y| / max|y| (robust to tiny entries)."""
        x, y = np.asarray(x), np.asarray(y)
        return np.array(np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))

    print(f"\n== correctness: {_label(REPO)} vs {a.ref}, freqs={a.freqs} ==")
    # 1. Official criteria (on this frequency subset).
    s, e = cand["sensitivity"], cand["expected"]
    dlog_ref = np.abs(np.log(s) - np.log(e)).max()
    base_dlog = np.abs(np.log(base["sensitivity"]) - np.log(e)).max()
    print(f"max |dln sens| vs public reference : {dlog_ref:.3e}  (limit 1e-5; baseline has {base_dlog:.3e})")
    if not (np.all(s > 0) and dlog_ref <= 1e-5):
        fails.append("sensitivity vs reference")
    for k in ("carrier", "signal", "noise", "sensitivity", "powers", "loss", "gradient"):
        if not np.isfinite(cand[k]).all():
            fails.append(f"non-finite {k}")
    if not np.any(cand["gradient"] != 0):
        fails.append("gradient identically zero")
    # 2. Agreement with the baseline implementation.
    print(f"{'output':12s} {'max rel err vs baseline':>24s}")
    ds = np.abs(np.log(cand["sensitivity"]) - np.log(base["sensitivity"])).max()
    print(f"{'sensitivity':12s} {ds:24.3e}  (max |dln| per frequency)")
    (fails if ds > 5e-6 else warns if ds > 1e-9 else []).append(f"sensitivity dln vs baseline {ds:.1e}")
    dl = abs(float(cand["loss"]) - float(base["loss"]))
    print(f"{'loss':12s} {dl:24.3e}  (absolute)")
    for k in ("carrier", "signal", "noise", "powers"):
        r = float(rel(cand[k], base[k]))
        flag = "  FAIL" if r > 1e-5 else ("  warn" if r > 1e-8 else "")
        print(f"{k:12s} {r:24.3e}{flag}")
        (fails if r > 1e-5 else warns if r > 1e-8 else []).append(f"{k} rel err {r:.1e}")
    # 3. All 193 gradients.
    g, g0 = cand["gradient"], base["gradient"]
    floor = 1e-6 * np.abs(g0).max()  # components below this are numerically ~0
    rg = np.abs(g - g0) / np.maximum(np.abs(g0), floor)
    worst = np.argsort(rg)[::-1][:5]
    names = [json.loads(x) for x in base["pair_names"]]
    print(f"gradient     {rg.max():24.3e}  (all {g.size}; worst: " +
          ", ".join(f"{names[i] if not isinstance(names[i][0], list) else names[i][0]}={rg[i]:.1e}" for i in worst[:3]) + ")")
    print(f"gradient cosine similarity vs baseline: {float(g @ g0 / np.linalg.norm(g) / np.linalg.norm(g0)):.12f}")
    if rg.max() > 1e-3:
        fails.append(f"gradient rel err {rg.max():.1e}")
    elif rg.max() > 1e-6:
        warns.append(f"gradient rel err {rg.max():.1e}")
    # 4. Optional finite differences.
    if a.fd and "fd_index" in cand.files:
        # Informational: this optical setup is so sensitive that FD is only trustworthy where the
        # two step sizes agree. What matters is that the candidate behaves like the baseline.
        h = cand["fd_steps"]
        print(f"finite differences (central; steps {h[0]:g}, {h[1]:g}; trust FD only where the two agree):")
        print(f"   {'parameter':26s} {'AD cand':>11s} {'AD base':>11s} {'FD h1':>11s} {'FD h2':>11s} {'AD~FD':>6s}")
        for i, (v1, v2) in zip(cand["fd_index"], cand["fd_value"]):
            nm = names[i] if not isinstance(names[i][0], list) else names[i][0]
            conv = abs(v1 - v2) <= 0.01 * max(abs(v2), 1e-6) + 1e-6
            ok = abs(g[i] - v2) <= 0.01 * abs(v2) + 1e-6
            tag = ("yes" if ok else "NO") if conv else "n/a"
            print(f"   {str(nm):26s} {g[i]: .4e} {g0[i]: .4e} {v1: .4e} {v2: .4e} {tag:>6s}")
            if conv and not ok and abs(g0[i] - v2) <= 0.01 * abs(v2) + 1e-6:
                fails.append(f"FD disagrees with candidate AD for {nm} (baseline AD agrees)")
    print()
    for w in warns:
        print("WARN:", w)
    if fails:
        for f in fails:
            print("FAIL:", f)
        sys.exit(1)
    print("PASS" + (" (with warnings)" if warns else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, worker=False):
        p.add_argument("--freqs", type=int, default=10, help="evenly spaced subset of the 101 frequencies (101 = full)")
        p.add_argument("--cpus", type=int, default=1, help="pin to N CPUs like the jury (0 = no pinning)")
        if worker:
            p.add_argument("--repo", required=True)
            p.add_argument("--out", required=True)

    p = sub.add_parser("parts", help="time parts"); common(p)
    p.add_argument("--parts", help=f"comma list (default {','.join(DEFAULT_PARTS)})")
    p.add_argument("--repeats", type=int, default=3, help="warm repetitions inside each process")
    p.add_argument("--trials", type=int, default=1, help="fresh-process trials per part (median reported)")
    p.add_argument("--ref", help=f"also run a baseline git ref (e.g. {DEFAULT_REF}) and print speedups")
    p.add_argument("--out", help="output JSON path")
    p.set_defaults(fn=cmd_parts)

    p = sub.add_parser("check", help="correctness vs baseline ref"); common(p)
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--fd", action="store_true", help="add finite-difference spot checks")
    p.add_argument("--fd-count", type=int, default=4)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("compare", help="speedup between two parts JSONs")
    p.add_argument("baseline"); p.add_argument("candidate")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("_worker_parts"); common(p, True)
    p.add_argument("--part", required=True); p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(fn=worker_parts)

    p = sub.add_parser("_worker_outputs"); common(p, True)
    p.add_argument("--fd", action="store_true"); p.add_argument("--fd-count", type=int, default=4)
    p.set_defaults(fn=worker_outputs)

    a = ap.parse_args()
    if getattr(a, "cpus", None) == 0:
        a.cpus = None
    a.fn(a)


if __name__ == "__main__":
    main()

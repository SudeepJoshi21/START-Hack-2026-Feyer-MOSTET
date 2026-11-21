#!/usr/bin/env python3
"""Compile-size report: what the benchmark's three jit programs consist of, and how long they take.

Mirrors benchmark.py's staged programs (forward = jax.vjp(simulate) returning the tape, sensitivity,
gradient), then reports per program: lower/compile seconds, StableHLO op count, warm run time, and
(with --by-function) the forward/gradient ops attributed to the innermost `differometor` function.
Optionally saves outputs (--save x.npz) so variants can be compared bit-for-bit (--compare a.npz b.npz).

Usage
  python scripts/compile_report.py                       # this checkout, 101 freqs, 1 CPU
  python scripts/compile_report.py --repo .bench/ref-xxx --save .bench/a.npz
  python scripts/compile_report.py --by-function
  python scripts/compile_report.py --compare .bench/a.npz .bench/b.npz
Not the scorer (that is benchmark.py); compile times are noisy (~±5-20 %), repeat runs.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap.add_argument("--freqs", type=int, default=101)
ap.add_argument("--cpus", type=int, default=1)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--by-function", action="store_true")
ap.add_argument("--save")
ap.add_argument("--compare", nargs=2)
args = ap.parse_args()

if args.compare:
    import numpy as np
    a, b = (np.load(p) for p in args.compare)
    for k in ("carrier", "signal", "noise", "sensitivity", "gradient"):
        x, y = a[k], b[k]
        rel = float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))
        print(f"{k:12s} identical={np.array_equal(x, y)!s:5s} max rel diff {rel:.2e}")
    sys.exit()

if args.cpus:
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.cpus])
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
REPO = os.path.abspath(args.repo)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))   # bench.py helpers

import collections, linecache, re, statistics, time
import jax
import jax.numpy as jnp
import numpy as np
import bench
import differometor

assert os.path.dirname(differometor.__file__).startswith(REPO), differometor.__file__
jax.config.update("jax_enable_compilation_cache", False)
W = bench._load_workload(args.freqs)
a, p, ports, ref = W["arrays"], W["params"], W["ports"], W["expected"]


def build(fn, *xs):
    t = time.perf_counter(); low = jax.jit(fn).lower(*xs); tl = time.perf_counter() - t
    t = time.perf_counter(); comp = low.compile(); tc = time.perf_counter() - t
    return low, comp, tl, tc


lowF, F, flo, fco = build(bench._fwd, p, a)
outs, spb = F(p, a)
lowS, S, slo, sco = build(bench._sens, outs, ports, ref)
loss, rpb, (sens, powers) = S(outs, ports, ref)
one = jnp.asarray(1.0)
lowG, G, glo, gco = build(bench._bwd, spb, rpb, one)
grad = G(spb, rpb, one)

fw, gw = [], []
for _ in range(args.repeats):
    del spb, rpb
    t = time.perf_counter(); outs, spb = jax.block_until_ready(F(p, a)); fw.append(time.perf_counter() - t)
    loss, rpb, (sens, powers) = S(outs, ports, ref)
    t = time.perf_counter(); grad = jax.block_until_ready(G(spb, rpb, one)); gw.append(time.perf_counter() - t)

ops = lambda low: low.as_text().count("stablehlo.") + low.as_text().count("chlo.")
print(f"repo {REPO}  ({args.freqs} freqs, {args.cpus} CPU)")
print(f"{'program':12s} {'lower':>7s} {'compile':>8s} {'ops':>6s} {'warm run':>9s}")
print(f"{'forward':12s} {flo:7.2f} {fco:8.2f} {ops(lowF):6d} {statistics.median(fw):9.3f}")
print(f"{'sensitivity':12s} {slo:7.2f} {sco:8.2f} {ops(lowS):6d}")
print(f"{'gradient':12s} {glo:7.2f} {gco:8.2f} {ops(lowG):6d} {statistics.median(gw):9.3f}")
print(f"{'TOTAL compile':12s} {flo + fco + slo + sco + glo + gco:16.2f}")
print(f"max |dln S| vs reference {float(np.max(np.abs(np.log(sens) - np.log(ref)))):.3e}")

if args.save:
    c, s, n = jax.device_get(outs)
    np.savez(args.save, carrier=c, signal=s, noise=n, sensitivity=np.asarray(sens), gradient=np.asarray(grad))
    print("saved", args.save)

if args.by_function:
    def func_of(path, line):
        for ln in range(line, 0, -1):
            m = re.match(r"\s*def (\w+)", linecache.getline(path, ln))
            if m:
                return m.group(1)
        return "?"

    for name, low in (("FORWARD", lowF), ("GRADIENT", lowG)):
        txt = low.as_text(debug_info=True)
        aliases = dict(re.findall(r"^(#loc\d+) = (loc\(.*\))$", txt, re.M))

        def frames(loc, depth=0):
            if depth > 30:
                return []
            out = [(m.group(1), int(m.group(2))) for m in re.finditer(r'"([^"]+\.py)":(\d+)', loc)]
            for al in re.findall(r"#loc\d+", loc):
                out += frames(aliases.get(al, ""), depth + 1)
            return out

        found = re.findall(r"^\s*(?:%\S+ = )?\"?(?:stablehlo|chlo)\.\w+[^\n]*?(loc\(.*\)|#loc\d+)\s*$", txt, re.M)
        by_func = collections.Counter()
        for loc in found:
            fr = [(f, ln) for f, ln in frames(loc) if "/differometor/" in f and f.startswith(REPO)]
            by_func[f"{os.path.basename(fr[0][0])}::{func_of(*fr[0])}" if fr else "(outside differometor)"] += 1
        print(f"\n{name}: {len(found)} ops by innermost differometor function")
        for k, v in by_func.most_common(12):
            print(f"   {v:5d} ({100 * v / max(len(found), 1):4.1f}%)  {k}")

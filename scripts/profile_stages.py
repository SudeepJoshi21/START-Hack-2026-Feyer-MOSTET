#!/usr/bin/env python3
"""Stage-by-stage profile mirroring benchmark.py, with a configurable frequency count.

Usage:
    python scripts/profile_stages.py --freqs 10            # cheap local run (~1/10 memory)
    python scripts/profile_stages.py --freqs 101 --cpus 1  # full workload (grid)
    python scripts/profile_stages.py --freqs 10 --dump-hlo /tmp/hlo   # also dump XLA HLO

Compile time is nearly independent of --freqs; runtime scales ~linearly with it.
Not a substitute for benchmark.py (no validation, not the scored harness).
"""
from time import perf_counter

T0 = perf_counter()
import argparse
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--freqs", type=int, default=10)
p.add_argument("--cpus", type=int)
p.add_argument("--dump-hlo", type=str)
args = p.parse_args()
if args.cpus:
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.cpus])
    for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[n] = str(args.cpus)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
if args.dump_hlo:
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={args.dump_hlo} --xla_dump_hlo_as_text"

t = perf_counter()
import jax
import jax.numpy as jnp
import numpy as np
t_jax = perf_counter() - t
t = perf_counter()
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
t_df = perf_counter() - t
import json
from pathlib import Path

rows = [("import jax/numpy", t_jax), ("import differometor", t_df)]


def timed(label, fn):
    s = perf_counter()
    r = fn()
    jax.block_until_ready(r)
    rows.append((label, perf_counter() - s))
    return r


jax.config.update("jax_enable_compilation_cache", False)
timed("init backend", lambda: jax.devices("cpu"))

repo = Path(__file__).resolve().parents[1]
data = repo / "examples" / "data"
fixture = json.loads((data / "uifo_800_3000.json").read_text())
reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())
idx = np.linspace(0, 100, args.freqs).round().astype(int)

PROPS = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def build():
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPS)
    comps = fixture["nodes"] | fixture["edges"]
    vals = [comps[(pr[0] if isinstance(pr[0], list) else pr)[0]]["properties"][(pr[0] if isinstance(pr[0], list) else pr)[1]] for pr in pairs]
    freqs = jnp.asarray(np.asarray(reference["frequencies"])[idx])
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], freqs, pairs)
    arrays["optimized_parameters"] = jnp.asarray(vals, dtype=jnp.float64)
    return arrays, jnp.asarray(ports), arrays["optimized_parameters"], jnp.asarray(np.asarray(reference["sensitivities"])[idx])


arrays, ports, params, expected = timed("build setup (run_build_step)", build)
one = jnp.asarray(1.0)


def fwd(p, a):
    return jax.vjp(lambda q: df.simulate(**dict(a, optimized_parameters=q)), p)


def readout(outs, ports, ref):
    c, s, n = outs
    sig = df.signal_detector(c, s)[ports].squeeze()
    sens = (n / jnp.abs(sig)).squeeze()
    return jnp.mean(jnp.log10(sens / ref)), (sens, df.power_detector(c))


def sens_pb(outs, ports, ref):
    return jax.vjp(lambda v: readout(v, ports, ref), outs, has_aux=True)


def bwd(sp, rp, ct):
    return sp(rp(ct)[0])[0]


def compile_(label, fn, *a):
    low = timed(f"{label}: trace/lower", lambda: jax.jit(fn).lower(*a))
    return timed(f"{label}: compile", low.compile)


F = compile_("forward", fwd, params, arrays)
outs, spb = timed("forward run", lambda: F(params, arrays))
S = compile_("sensitivity", sens_pb, outs, ports, expected)
loss, rpb, (sens, powers) = timed("sensitivity run", lambda: S(outs, ports, expected))
G = compile_("gradient", bwd, spb, rpb, one)
grad = timed("gradient run", lambda: G(spb, rpb, one))
timed("host transfer", lambda: jax.device_get((outs, sens, powers, loss, grad)))

total = sum(s for _, s in rows)
print(f"\n{'stage':38s} {'sec':>9s} {'%':>6s}   (freqs={args.freqs})")
for label, s in rows:
    print(f"{label:38s} {s:9.3f} {100*s/total:6.1f}")
print(f"{'TOTAL (excl. interpreter start)':38s} {total:9.3f}")
print(f"\nmax |dlog sens| vs reference: {float(np.max(np.abs(np.log(sens) - np.log(expected)))):.2e}")
print(f"gradient finite: {bool(np.isfinite(grad).all())}, nonzero: {int((np.asarray(grad) != 0).sum())}/{grad.size}")
print("system sizes:", {k: tuple(arrays[k].shape) for k in ("carrier_matrix", "signal_matrix", "noise_matrix")})
print("component fn evals per freq:", {k: int(arrays[k].shape[0]) for k in ("carrier_function_indices", "signal_function_indices", "noise_function_indices")})

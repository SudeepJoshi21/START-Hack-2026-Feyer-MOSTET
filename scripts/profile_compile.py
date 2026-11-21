"""
Where does the XLA compile time of the integrated simulator go?   (branch `compile-profiling`)

Compile time hardly depends on the number of frequencies, so this runs at 4 frequencies (override with --freqs).
Every measurement compiles the *benchmark-style* programs (`bench._fwd` forward + pullback residuals, and the
`bench._bwd` gradient) in this one process, pinned to one CPU like the jury.

Sections
  baseline   full forward / gradient compile, optimized-HLO size (instructions, fusions)
  stubs      replace one part at a time by a trivial stand-in and recompile:
               components  every component function -> p[0] * (1, 0, 0, 0)  (cheap, same shapes)
               carrier     the dense 286x286 carrier solve -> its right-hand side
               linking     the linking `lax.switch` -> identity
             (removed cost = full - stubbed; numbers are attributions, not exact sums)
  types      compile of jax.vjp(vmap(f)) for each component type alone, sorted
  opt0       optional (--opt0): the same baseline in a subprocess with XLA_FLAGS=--xla_backend_optimization_level=0
             (diagnostic only; using it in a submission needs organiser approval, FINDINGS §1.7)

Usage:  python scripts/profile_compile.py [--freqs 4] [--opt0]
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import bench  # noqa: E402


def compile_pair(W):
    """lower+compile seconds of the forward and the gradient program, plus optimized-HLO stats of the forward."""
    import jax
    import jax.numpy as jnp

    jax.clear_caches()  # monkeypatched globals are not part of the jit cache key
    a, ports, p, ref = W["arrays"], W["ports"], W["params"], W["expected"]
    t = time.perf_counter()
    forward = jax.jit(bench._fwd).lower(p, a).compile()
    t_forward = time.perf_counter() - t
    hlo = forward.as_text()
    outs, spb = forward(p, a)
    sens = jax.jit(bench._sens).lower(outs, ports, ref).compile()
    _, rpb, _ = sens(outs, ports, ref)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    t = time.perf_counter()
    jax.jit(bench._bwd).lower(spb, rpb, one).compile()
    t_gradient = time.perf_counter() - t
    stats = {
        "instructions": len(re.findall(r"^\s+\S+ = ", hlo, flags=re.M)),
        "fusions": len(re.findall(r"^fused_computation|^%fused_computation|^\S*fused\S* ", hlo, flags=re.M)),
        "hlo_kb": len(hlo) // 1024,
    }
    return t_forward, t_gradient, stats


def section_baseline(W):
    f, g, s = compile_pair(W)
    print(f"[baseline XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}] forward compile {f:6.2f} s | gradient compile {g:6.2f} s | optimized HLO {s}", flush=True)
    return f, g


def section_stubs(W, base):
    import jax.numpy as jnp
    from differometor import assembly, components
    simulate = sys.modules["differometor.simulate"]

    def stub_component(p):
        return jnp.zeros(4, jnp.complex128).at[0].set(p[0])

    stubs = {
        "components": (assembly, "FUNCTIONS", [stub_component] * len(components.FUNCTIONS)),
        "carrier": (simulate, "solve", lambda m, b: b.astype(jnp.complex128)),
        "linking": (simulate, "LINKING_FUNCTIONS", [lambda x: x] * max(1, len(simulate.LINKING_FUNCTIONS))),
    }
    for name, (module, attribute, replacement) in stubs.items():
        original = getattr(module, attribute)
        setattr(module, attribute, replacement)
        try:
            f, g, s = compile_pair(W)
        finally:
            setattr(module, attribute, original)
        print(f"[stub {name:10s}] forward {f:6.2f} s (removed {base[0] - f:+6.2f}) | "
              f"gradient {g:6.2f} s (removed {base[1] - g:+6.2f}) | HLO {s}", flush=True)


def section_types(W):
    import jax
    import numpy as np
    from differometor.components import FUNCTIONS

    plan = W["arrays"]["assembly"]
    params = np.asarray(W["arrays"]["parameters"]).real[0]
    used = {}
    for part in ("carrier", "signal", "noise"):
        for t, inputs in enumerate(plan[part]["groups"]):
            if inputs.shape[0]:
                used.setdefault(t, np.asarray(inputs))
    rows = []
    for t, inputs in used.items():
        x = jax.numpy.asarray(params[inputs])
        fn = lambda x, f=FUNCTIONS[t]: jax.vjp(jax.vmap(f), x)  # noqa: E731
        t0 = time.perf_counter()
        jax.jit(fn).lower(x).compile()
        rows.append((time.perf_counter() - t0, FUNCTIONS[t].__name__, inputs.shape[0]))
    for seconds, name, calls in sorted(rows, reverse=True):
        print(f"[type] {name:40s} calls {calls:4d}  vjp(vmap) compile {seconds:6.3f} s", flush=True)
    print(f"[type] sum {sum(r[0] for r in rows):.2f} s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freqs", type=int, default=4)
    parser.add_argument("--opt0", action="store_true")
    parser.add_argument("--only-baseline", action="store_true")
    parser.add_argument("--stubs-only", action="store_true")
    args = parser.parse_args()

    bench._setup_env(1)
    W = bench._load_workload(args.freqs)
    base = section_baseline(W)
    if args.only_baseline:
        return
    section_stubs(W, base)
    if args.stubs_only:
        return
    section_types(W)
    if args.opt0:
        env = dict(os.environ, XLA_FLAGS="--xla_backend_optimization_level=0")
        subprocess.run([sys.executable, __file__, "--freqs", str(args.freqs), "--only-baseline"], env=env, check=True)


if __name__ == "__main__":
    main()

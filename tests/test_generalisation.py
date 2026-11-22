"""
Generalisation / anti-overfitting checks for the host path (the-prime-model-new2; complements test_robustness.py).

Run:  PYTHONPATH=. python tests/test_generalisation.py      (a few minutes on one CPU)

0. Branch cuts of the NumPy shim (sqrt, arcsin, log10) match jax.numpy, signed zeros included (edge-cases bundle).
1. Input-slot detection (`host._used_slots`) is structural: for every component function it covers every slot with a
   non-zero derivative at random inputs unrelated to any setup.
2. The forward-state cache (`host._LAST_FORWARD`) never leaks between inputs: interleaved forward/backward calls for
   two different parameter vectors give exactly the gradients of fresh, separate calls.
3. Gradients at a design point that is not the benchmark's (all 193 variables perturbed by up to +-5 %) agree with
   central finite differences of the same loss (benchmark.py's tolerances), for several kinds of variable.
4. Process history does not matter: the benchmark-workload outputs after running other setups in the same process
   equal those of a fresh process bit for bit.
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import differometor as df  # noqa: E402
from differometor import host  # noqa: E402
from differometor.components import FUNCTIONS  # noqa: E402
import bench  # noqa: E402


def test_used_slots_structural():
    rng = np.random.default_rng(0)
    functions = host.numpy_functions()
    for t, function in enumerate(functions):
        x = rng.uniform(0.1, 2.0, (7, 6)) + 0j                   # 7 padded slots, 6 calls, generic values
        slots = set(host._used_slots(t, x).tolist())
        jac = host._jacobian(function, x)                        # all 7 slots, numerically
        numeric = set(np.flatnonzero(np.any((jac != 0) | ~np.isfinite(jac), axis=(0, 1))).tolist())
        assert numeric <= slots, f"{FUNCTIONS[t].__name__}: derivative w.r.t. {numeric - slots} missed"
    print("PASS test_used_slots_structural:", {FUNCTIONS[t].__name__: v.tolist() for t, v in host._USED_SLOTS.items()})


def test_branch_cuts_match_jax():
    """Shim functions with branch cuts (sqrt, arcsin, log10) follow jax.numpy's conventions, signed zeros included."""
    shim = host._SHIM
    z = np.array([-4 + 0j, complex(-4, -0.0), complex(-1.025, -0.0), complex(1.025, -0.0), complex(1.5, -0.0),
                  complex(-1.025, 0.0), 0.3 - 0.2j, complex(2.0, -0.0)])
    for name in ("sqrt", "arcsin", "log10"):
        ours = np.asarray(getattr(shim, name)(z))
        theirs = np.asarray(getattr(jnp, name)(jnp.asarray(z)))
        assert np.allclose(ours, theirs, rtol=1e-14, atol=1e-15), f"{name}: {ours} vs {theirs}"
        d_ours = getattr(shim, name)(host.Dual(z, np.ones((1,) + z.shape, complex))).d[0]
        d_theirs = np.asarray([complex(jax.jvp(getattr(jnp, name), (jnp.asarray(v),), (jnp.asarray(1.0 + 0j),))[1]) for v in z])
        assert np.allclose(d_ours, d_theirs, rtol=1e-12, atol=1e-15), f"d {name}: {d_ours} vs {d_theirs}"
    print("PASS test_branch_cuts_match_jax (sqrt, arcsin, log10 values and derivatives, signed zeros)")


def _workload():
    W = bench._load_workload(101)
    return W, W["arrays"]


def test_forward_cache_does_not_leak():
    W, a = _workload()
    p1 = W["params"]
    p2 = p1 * (1 + 0.01 * jnp.asarray(np.random.default_rng(1).uniform(-1, 1, p1.shape)))
    fwd = jax.jit(lambda p, a: jax.vjp(lambda q: df.simulate(**dict(a, optimized_parameters=q)), p))
    rng = np.random.default_rng(2)

    def cotangent(outs):
        return tuple(jnp.asarray(rng.standard_normal(o.shape) + (1j * rng.standard_normal(o.shape) if jnp.iscomplexobj(o) else 0))
                     .astype(o.dtype) for o in outs)

    out1, pb1 = fwd(p1, a)
    ct = cotangent(out1)
    out2, pb2 = fwd(p2, a)                    # cache now holds p2's state
    g1_interleaved = np.asarray(pb1(ct)[0])   # must not use p2's state
    g2_interleaved = np.asarray(pb2(ct)[0])
    _, pb1_fresh = fwd(p1, a)
    g1_fresh = np.asarray(pb1_fresh(ct)[0])
    _, pb2_fresh = fwd(p2, a)
    g2_fresh = np.asarray(pb2_fresh(ct)[0])
    assert np.array_equal(g1_interleaved, g1_fresh) and np.array_equal(g2_interleaved, g2_fresh)
    assert not np.array_equal(g1_fresh, g2_fresh)
    print("PASS test_forward_cache_does_not_leak (interleaved == fresh, bitwise; p1 and p2 gradients differ)")


def test_gradient_finite_differences_off_benchmark_point():
    W, a = _workload()
    rng = np.random.default_rng(3)
    p = W["params"] * (1 + 0.05 * jnp.asarray(rng.uniform(-1, 1, W["params"].shape)))
    ports, expected = W["ports"], W["expected"]

    def loss(q):
        carrier, signal, noise = df.simulate(**dict(a, optimized_parameters=q))
        s = df.signal_detector(carrier, signal)[ports]
        det = s.squeeze() if ports.shape[0] == 1 else s[0] - s[1]
        return jnp.mean(jnp.log10((noise / jnp.abs(det)).squeeze() / expected))

    grad = np.asarray(jax.jit(jax.grad(loss))(p))
    f = jax.jit(loss)
    pairs = W["pairs"]
    names = [("/".join(x[0]) if isinstance(x[0], list) else x[0], x[1]) for x in pairs]
    checked = []
    for prop, step in (("tuning", 1e-3), ("tuning", 1e-3), ("tuning", 1e-3), ("length", 1e-4), ("db", 1e-3)):
        candidates = [i for i, (_, pr) in enumerate(names) if pr == prop and i not in [c[0] for c in checked]]
        if not candidates:
            continue
        i = candidates[len(checked) % len(candidates)]
        estimates = []
        for h in (step, step / 2):
            up = float(f(p.at[i].add(h))); down = float(f(p.at[i].add(-h)))
            estimates.append((up - down) / (2 * h))
        np.testing.assert_allclose(estimates[0], estimates[1], rtol=0.01, atol=1e-6)    # differences converged
        np.testing.assert_allclose(grad[i], estimates[1], rtol=0.01, atol=1e-6)        # benchmark.py's tolerance
        checked.append((i, names[i], grad[i], estimates[1]))
    assert len(checked) >= 3
    for i, name, g, e in checked:
        print(f"   {name}: autodiff {g:+.6e}  central difference {e:+.6e}")
    print(f"PASS test_gradient_finite_differences_off_benchmark_point ({len(checked)} variables, design point perturbed +-5%)")


def _benchmark_outputs_in_fresh_process():
    code = (f"import sys, os; os.environ['JAX_PLATFORMS']='cpu'; os.environ['JAX_ENABLE_X64']='true'; "
            f"sys.path[:0]=[{str(ROOT)!r}, {str(ROOT / 'scripts')!r}]; import numpy as np, jax, bench, differometor as df; "
            f"W = bench._load_workload(101); "
            f"o, pb = jax.jit(bench._fwd)(W['params'], W['arrays']); "
            f"np.savez(sys.argv[1], *[np.asarray(x) for x in o], np.asarray(pb(tuple(jax.numpy.ones_like(x) for x in o))[0]))")
    out = str(ROOT / ".fresh_outputs.npz")
    subprocess.run([sys.executable, "-c", code, out], check=True, env=dict(os.environ))
    data = np.load(out)
    os.remove(out)
    return [data[k] for k in data.files]


def test_process_history_does_not_matter():
    from differometor.setups import aligo, voyager

    sim = sys.modules["differometor.simulate"]
    for setup_fn, mode in ((voyager, "amplitude_modulation"), (aligo, "space_modulation")):  # other setups first
        setup, _ = setup_fn(mode=mode)
        arrays, *_ = sim.run_build_step(setup, [("f", "frequency")], jnp.logspace(1, 3.5, 30))
        jax.jit(lambda a: df.simulate(**a))(arrays)
    W, a = _workload()
    o, pb = jax.jit(bench._fwd)(W["params"], a)
    here = [np.asarray(x) for x in o] + [np.asarray(pb(tuple(jnp.ones_like(x) for x in o))[0])]
    fresh = _benchmark_outputs_in_fresh_process()
    assert all(np.array_equal(x, y) for x, y in zip(here, fresh))
    print("PASS test_process_history_does_not_matter (after Voyager/aLIGO runs == fresh process, bitwise)")


if __name__ == "__main__":
    test_used_slots_structural()
    test_branch_cuts_match_jax()
    test_forward_cache_does_not_leak()
    test_gradient_finite_differences_off_benchmark_point()
    test_process_history_does_not_matter()
    print("ALL PASS")

"""
Host-native path (`differometor.host`) vs the JAX path (WS1-WS4): outputs and gradients.

Run:  python tests/test_host.py
1. Component re-binding: NumPy values and dual-number Jacobians equal JAX's (`jax.jacfwd`) for every used type.
2. Benchmark setup (uifo_800_3000, 8 frequencies): outputs and the gradient of the benchmark loss, host vs JAX.
3. The benchmark's split programs (forward+pullback jit, sensitivity jit, gradient jit) complete and agree (the
   pinned-CPU callback deadlock case).
4. Voyager with its optimisation pairs (reflectivity, tuning, db, angle, power, mass, length, detector phase) and
   aLIGO frequency/db sweeps and a cavity carrier sweep: outputs and gradients, host vs JAX.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import differometor as df  # noqa: E402
from differometor import host  # noqa: E402
from differometor.components import FUNCTIONS  # noqa: E402

sim = sys.modules["differometor.simulate"]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if b.size == 0:
        return 0.0 if a.size == 0 else float("inf")
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def without_host(arrays):
    return {k: v for k, v in arrays.items() if k != "host_plan"}


def test_components(arrays):
    plan = arrays["assembly"]
    params = np.asarray(arrays["parameters"])[0]
    functions = host.numpy_functions()
    worst_v = worst_j = 0.0
    for part in ("carrier", "signal", "noise"):
        for t, inputs in enumerate(plan[part]["groups"]):
            inputs = np.asarray(inputs)
            if inputs.shape[0] == 0:
                continue
            x = params[inputs][:5]                                              # (k, I) complex
            ref_v = np.asarray(jax.vmap(FUNCTIONS[t])(jnp.asarray(x)))         # (k, 4)
            ref_j = np.asarray(jax.vmap(jax.jacfwd(lambda r, f=FUNCTIONS[t]: f(r.astype(jnp.complex128))))(
                jnp.asarray(x.real)))                                           # (k, 4, I) real directions
            v = host._evaluate(functions[t], x.T)
            j = host._jacobian(functions[t], x.T)
            worst_v = max(worst_v, rel(v, ref_v))
            worst_j = max(worst_j, rel(j, ref_j) if np.abs(ref_j).max() > 0 else float(np.abs(j).max()))
    print(f"[components] values max rel diff {worst_v:.2e}, Jacobians {worst_j:.2e}")
    assert worst_v < 1e-12 and worst_j < 1e-10


def test_benchmark_setup():
    import bench

    W = bench._load_workload(8)
    a = W["arrays"]
    assert "host_plan" in a, "host plan missing (precondition failed?)"
    test_components(a)
    ports, ref, p = W["ports"], W["expected"], W["params"]

    def loss(q, arrays):
        c, s, n = df.simulate(**dict(arrays, optimized_parameters=q))
        sens = n / jnp.abs(df.signal_detector(c, s)[ports[0]])
        return jnp.mean(jnp.log10(sens / ref)), (c, s, n)

    (l_h, o_h), g_h = jax.jit(jax.value_and_grad(loss, has_aux=True))(p, a)
    (l_j, o_j), g_j = jax.jit(jax.value_and_grad(loss, has_aux=True))(p, without_host(a))
    for name, x, y in zip(("carrier", "signal", "noise"), o_h, o_j):
        print(f"[uifo] {name:8s} host vs JAX max rel diff {rel(x, y):.2e}")
        assert rel(x, y) < 1e-6
    print(f"[uifo] loss diff {abs(float(l_h - l_j)):.2e}  gradient max rel diff {rel(g_h, g_j):.2e}  "
          f"cosine {float(jnp.dot(g_h, g_j) / jnp.linalg.norm(g_h) / jnp.linalg.norm(g_j)):.12f}")
    assert rel(g_h, g_j) < 1e-5

    T = bench.Timer()
    host_out, _ = bench._run_pipeline(W, T, 1)
    times = {stage: round(t, 3) for stage, _, t in T.rows}
    print(f"[uifo] benchmark-style split pipeline OK: {times}")
    assert rel(host_out[4], g_j) < 1e-5


def _compare(name, setup, pairs, values, optimization_pairs=None, loss_ports=None):
    arrays, ports, *_ = sim.run_build_step(setup, pairs, values, optimization_pairs)
    if arrays["optimized_parameters"] is None:   # callers supply these; use the setup's current values
        q0 = np.zeros(int(np.max(np.asarray(arrays["optimized_value_indices"]))) + 1)
        q0[np.asarray(arrays["optimized_value_indices"])] = np.real(
            np.asarray(arrays["parameters"])[0, np.asarray(arrays["optimized_parameter_indices"])])
        arrays["optimized_parameters"] = jnp.asarray(q0)
    assert "host_plan" in arrays, f"{name}: host plan not built"
    out_h = jax.jit(lambda a: df.simulate(**a))(arrays)
    out_j = jax.jit(lambda a: df.simulate(**a))(without_host(arrays))
    diffs = [rel(x, y) for x, y in zip(out_h, out_j)]
    line = f"[{name}] outputs carrier/signal/noise " + " / ".join(f"{d:.1e}" for d in diffs)
    assert max(diffs) < 1e-6, line
    if optimization_pairs:
        def loss(q, a):
            c, s, n = df.simulate(**dict(a, optimized_parameters=q))
            sig = df.signal_detector(c, s)[ports]
            det = sig[0] - sig[1] if sig.shape[0] > 1 else sig[0]
            return jnp.sum(jnp.log10(n / jnp.abs(det)))

        q0 = arrays["optimized_parameters"]
        g_h = jax.jit(jax.grad(loss))(q0, arrays)
        g_j = jax.jit(jax.grad(loss))(q0, without_host(arrays))
        line += f" | gradient ({q0.shape[0]} params) max rel diff {rel(g_h, g_j):.1e}"
        assert rel(g_h, g_j) < 1e-5, line
    print(line)


def test_other_setups():
    import json
    from differometor.setups import aligo, voyager

    data = json.loads((ROOT / "examples" / "data" / "aligo_test_data.json").read_text())
    _compare("aligo f-sweep", aligo()[0], [("f", "frequency")], jnp.array(data["frequencies"]))
    _compare("aligo db-sweep", aligo()[0], [("sq1", "db")], jnp.array([[0.5, 5.0, 10.0]]))

    setup, property_pairs = voyager()
    wanted = {"reflectivity", "tuning", "db", "angle", "power", "mass", "length", "phase"}
    optimization_pairs = [pair for pair in property_pairs if pair[1] in wanted]
    _compare("voyager f-sweep + gradients", setup, [("f", "frequency")], jnp.logspace(1.3, 3.7, 30), optimization_pairs)

    cav = df.Setup()
    cav.add("laser", "l0", power=1)
    cav.add("mirror", "m0", reflectivity=0.99, loss=0)
    cav.add("mirror", "m1", reflectivity=0.991, loss=0)
    cav.space("l0", "m0", length=1)
    cav.space("m0", "m1", length=1)
    cav.add("detector", "refl", target="m0", port="left", direction="out")
    _compare("cavity tuning (carrier sweep)", cav, [("m0", "tuning")], jnp.linspace(-180, 180, 50))


if __name__ == "__main__":
    test_benchmark_setup()
    test_other_setups()
    print("ALL PASS")

"""
WS3 equivalence tests: grouped component evaluation + pattern assembly (`differometor.assembly`) vs the original code.

Run:  python tests/test_assembly.py      (about 2-4 minutes on one CPU)

1. Benchmark setup (uifo_800_3000, 8 frequencies): the dense signal system and right-hand side built by the original
   steps (update -> carrier scaling -> conjugated blocks) equal `assembly.densify(values)` / `rhs` exactly.
2. Same setup: carrier, signal, noise and the gradient of the mean log sensitivity, with and without the plan.
3. Other setups through `df.run` with and without the plan: aLIGO frequency sweep and squeezer-db sweep,
   voyager frequency sweep, and a cavity with a carrier (tuning) sweep.
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
from differometor import assembly  # noqa: E402

sim = sys.modules["differometor.simulate"]


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def original_signal_system(a, parameters, carrier):
    """The original dense steps of `simulate`, copied verbatim (signal part only)."""
    m = jax.vmap(sim.update, in_axes=(0, None, None, None, None, None))(
        parameters, a["signal_matrix"], a["signal_function_input_indices"], a["signal_output_indices"],
        a["signal_matrix_indices"], a["signal_function_indices"],
    )
    pi, ci = a["signal_carrier_placing_indices"], a["signal_carrier_indices"]
    m = m.at[:, pi[0], pi[1]].set(-m[:, pi[0], pi[1]] * carrier[:, ci.flatten()])
    c = carrier.shape[1]
    m = m.at[:, c:2 * c, c:2 * c].set(jnp.conjugate(m[:, c:2 * c, c:2 * c]))
    m = m.at[:, 2 * c:, :c].set(jnp.conjugate(m[:, 2 * c:, :c]))
    m = m.at[:, c:2 * c, 2 * c:].set(jnp.conjugate(m[:, c:2 * c, 2 * c:]))
    return m[:, :, :-1], m[:, :, -1]


def test_benchmark_setup():
    import bench

    W = bench._load_workload(8)
    a = W["arrays"]
    assert "assembly" in a, "assembly plan missing (precondition failed?)"
    plan = a["assembly"]["signal"]
    print(f"[uifo] signal pattern nnz={plan['rows'].shape[0]} rhs nnz={plan['rhs_rows'].shape[0]} "
          f"call groups={[g.shape[0] for g in plan['groups'] if g.shape[0]]}")

    # 1. matrix level: build parameters and carrier exactly like simulate does (no optimized/linking changes needed:
    #    run_build_step already contains the base values; apply optimized values the same way simulate does)
    p = a["parameters"].at[[0], a["optimized_parameter_indices"]].set(
        a["optimized_parameters"][a["optimized_value_indices"]]
    )
    outs = jax.vmap(lambda i, x: jax.lax.switch(i, sim.LINKING_FUNCTIONS, x))(
        a["linking_function_indices"], p[0, a["linked_indices"]]
    )
    p = p.at[[0], a["indices_to_link"]].set(outs)
    p = sim.expand_parameters(p, a["carrier_changing_parameter_indices"], a["carrier_changing_values"])
    cm = jax.vmap(sim.update, in_axes=(0, None, None, None, None, None))(
        p, a["carrier_matrix"], a["carrier_function_input_indices"], a["carrier_output_indices"],
        a["carrier_matrix_indices"], a["carrier_function_indices"],
    )
    cm_new = jax.vmap(assembly.update, in_axes=(0, None, None, None))(
        p, a["carrier_matrix"], a["assembly"]["carrier"], a["carrier_matrix_indices"]
    )
    print(f"[uifo] carrier matrix  max rel diff {rel(cm_new, cm):.3e}")
    assert rel(cm_new, cm) < 1e-13
    carrier = jax.vmap(sim.solve)(cm[:, :, :-1], cm[:, :, -1])
    p = sim.expand_parameters(p, a["signal_changing_parameter_indices"], a["signal_changing_values"])
    A_old, b_old = original_signal_system(a, p, carrier)
    n = a["signal_matrix"].shape[0]
    values, rhs = assembly.signal_values_and_rhs(p, carrier, plan, n)
    A_new = assembly.densify(values, plan, n)
    print(f"[uifo] signal system   max rel diff {rel(A_new, A_old):.3e}, rhs {rel(rhs, b_old):.3e}")
    assert rel(A_new, A_old) < 1e-13 and rel(rhs, b_old) < 1e-13
    off_pattern = np.asarray(A_old).copy()
    off_pattern[:, np.asarray(plan["rows"]), np.asarray(plan["cols"])] = 0
    assert not off_pattern.any(), "a non-zero of the original matrix lies outside the pattern"

    # 2. outputs and gradient, with and without the plan
    old = {k: v for k, v in a.items() if k != "assembly"}
    ports, ref, params = W["ports"], W["expected"], W["params"]

    def loss(p, arrs):
        c, s, nz = df.simulate(**dict(arrs, optimized_parameters=p))
        sens = nz / jnp.abs(df.signal_detector(c, s)[ports[0]])
        return jnp.mean(jnp.log(sens / ref)), (c, s, nz)

    (l_new, out_new), g_new = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, a)
    (l_old, out_old), g_old = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, old)
    for name, x, y in zip(("carrier", "signal", "noise"), out_new, out_old):
        print(f"[uifo] {name:8s} max rel diff {rel(x, y):.3e}")
        assert rel(x, y) < 1e-6  # cond(A) ~ 1e30: last-bit differences inside jit show up at ~1e-8
    print(f"[uifo] loss diff {abs(float(l_new - l_old)):.3e}   gradient max rel diff {rel(g_new, g_old):.3e}")
    assert rel(g_new, g_old) < 1e-4


def _run_both(setup, pairs, values):
    assembly.ENABLED = True
    new = df.run(setup, pairs, values)[:3]
    assembly.ENABLED = False
    old = df.run(setup, pairs, values)[:3]
    assembly.ENABLED = True
    return new, old


def test_other_setups():
    import json
    from differometor.setups import aligo, voyager

    data = json.loads((ROOT / "examples" / "data" / "aligo_test_data.json").read_text())
    f = jnp.array(data["frequencies"])
    cases = [
        ("aligo f-sweep", aligo()[0], [("f", "frequency")], f),
        ("aligo db-sweep", aligo()[0], [("sq1", "db")], jnp.array([[0.5, 5.0, 10.0]])),
        ("voyager f-sweep", voyager()[0], [("f", "frequency")], jnp.logspace(1, 3.5, 40)),
    ]
    cav = df.Setup()
    cav.add("laser", "l0", power=1)
    cav.add("mirror", "m0", reflectivity=0.99, loss=0)
    cav.add("mirror", "m1", reflectivity=0.991, loss=0)
    cav.space("l0", "m0", length=1)
    cav.space("m0", "m1", length=1)
    cav.add("detector", "refl", target="m0", port="left", direction="out")
    cases.append(("cavity tuning (carrier sweep)", cav, [("m0", "tuning")], jnp.linspace(-180, 180, 50)))

    for name, setup, pairs, values in cases:
        arrays, *_ = sim.run_build_step(setup, pairs, values)
        used = "assembly" in arrays
        new, old = _run_both(setup, pairs, values)
        diffs = [rel(x, y) if np.asarray(y).size else 0.0 for x, y in zip(new, old)]
        finite = all(np.isfinite(np.asarray(x)).all() for x in new)
        print(f"[{name}] plan used={used}  carrier/signal/noise max rel diff "
              + " / ".join(f"{d:.2e}" for d in diffs) + f"  finite={finite}")
        assert finite and max(diffs) < 1e-6


if __name__ == "__main__":
    test_benchmark_setup()
    test_other_setups()
    print("ALL PASS")

"""Componentwise backward error of the unrefined SuperLU solves, per frequency (selective refinement potential)."""
import sys
R = sys.argv[1]; sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
import bench
bench._setup_env(1)
import numpy as np, jax.numpy as jnp
from differometor import host
import scenarios
sim = sys.modules["differometor.simulate"]
u = np.finfo(float).eps / 2
def state(name):
    setup, changing, values, pairs, init, ref = scenarios._scenarios()[name][1]()
    a, *_ = sim.run_build_step(setup, changing, jnp.asarray(values), pairs)
    idx = np.asarray(a["optimized_value_indices"]); q = np.zeros(int(idx.max()) + 1)
    q[idx] = np.real(np.asarray(a["parameters"])[0, np.asarray(a["optimized_parameter_indices"])])
    if init is not None: q = np.asarray(init, float)
    return host._forward(a["host_plan"], q, np.asarray(a["parameters"]), np.asarray(a["carrier_changing_values"]), np.asarray(a["signal_changing_values"]).real)[3]
for name in ("uifo800-benchmark", "uifo20-5000", "voyager", "voyager-ampl", "aligo"):
    st = state(name); fac = st["factors"]
    absfac = type("A", (), {})()
    def omega(b, trans):
        b = b[:, fac.perm] if fac.perm is not None else b
        x = fac._solve(b, trans)
        r = b - fac._apply(x, trans)
        data = fac.data; fac.data = np.abs(data).astype(complex)
        denom = np.abs(fac._apply(np.abs(x).astype(complex), "N" if trans == "N" else "T")).real + np.abs(b)
        fac.data = data
        w = np.abs(r) / np.maximum(denom, 1e-300)
        return w.max(axis=(1, 2)) / u
    wW = omega(np.swapaxes(st["c_rows"], 1, 2).astype(complex), "H")
    rng = np.random.default_rng(0); n = fac.n
    wA = omega(rng.standard_normal((fac.data.shape[0], n, 1)) + 0j, "T")
    print(f"{name:18s} omega/u, noise-weight solve: median {np.median(wW):.1f} max {wW.max():.1f} | adjoint-type solve: median {np.median(wA):.1f} max {wA.max():.1f} "
          f"| frequencies with omega <= 10u: weights {int((wW <= 10).sum())}/{wW.size}, adjoint {int((wA <= 10).sum())}/{wA.size}")

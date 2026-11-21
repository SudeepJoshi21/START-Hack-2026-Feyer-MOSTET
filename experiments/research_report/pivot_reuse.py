import sys
R = sys.argv[1]; sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
import bench
bench._setup_env(1)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from differometor import host, adjoint
import scenarios, jax.numpy as jnp
sim = sys.modules["differometor.simulate"]
def systems(name):
    if name == "uifo800-benchmark":
        W = bench._load_workload(101); a = W["arrays"]; q = np.asarray(W["params"])
    else:
        setup, changing, values, pairs, init, ref = scenarios._scenarios()[name][1]()
        a, *_ = sim.run_build_step(setup, changing, jnp.asarray(values), pairs)
        idx = np.asarray(a["optimized_value_indices"]); q = np.zeros(int(idx.max()) + 1)
        q[idx] = np.real(np.asarray(a["parameters"])[0, np.asarray(a["optimized_parameter_indices"])])
        if init is not None: q = np.asarray(init, float)
    plan = a["host_plan"]
    st = host._forward(plan, q, np.asarray(a["parameters"]), np.asarray(a["carrier_changing_values"]), np.asarray(a["signal_changing_values"]).real)[3]
    fac = st["factors"]; N = fac.n
    indptr = np.concatenate([[0], np.cumsum(np.bincount(fac.columns, minlength=N))])
    return [sp.csc_matrix((fac.data[f], fac.rows, indptr), shape=(N, N)) for f in range(fac.data.shape[0])]
for name in ("uifo800-benchmark", "uifo20-5000", "voyager", "voyager-freqmod", "aligo"):
    mats = systems(name); F = len(mats); ref = F // 2
    lu0 = spla.splu(mats[ref], permc_spec="NATURAL", diag_pivot_thresh=1.0, options=dict(PanelSize=4, Relax=4))
    pr, pc = lu0.perm_r, lu0.perm_c
    inv = np.argsort(pr)
    distinct = len({spla.splu(m, permc_spec="NATURAL", diag_pivot_thresh=1.0, options=dict(PanelSize=4, Relax=4)).perm_r.tobytes() for m in mats})
    maxl = []
    for m in mats:
        B = m[inv][:, pc]                                   # P_r A P_c: reference pivots on the diagonal
        lu = spla.splu(B.tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(PanelSize=4, Relax=4, SymmetricMode=True))
        ok = np.array_equal(lu.perm_r, np.arange(B.shape[0])) and np.array_equal(lu.perm_c, np.arange(B.shape[0]))
        maxl.append(np.abs(lu.L.data).max() if ok else np.inf)
    maxl = np.asarray(maxl)
    print(f"{name:18s} F={F} distinct own pivot orders {distinct:3d} | reference pivots: max|l| per frequency: "
          f"<=1+1e-12: {(maxl <= 1 + 1e-12).sum()}, <=2: {(maxl <= 2).sum()}, <=10: {(maxl <= 10).sum()}, worst {maxl.max():.3g}")

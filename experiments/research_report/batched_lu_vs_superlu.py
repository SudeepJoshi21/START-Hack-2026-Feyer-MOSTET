"""BatchedLU vs SuperLU: accuracy of N/T/H solves, acceptance, timing (benchmark + Voyager systems)."""
import sys, time
R = sys.argv[1]; sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
import bench
bench._setup_env(1)
import numpy as np, jax.numpy as jnp
from differometor import host, adjoint
sys.path.insert(0, R + "/experiments/research_report"); import batched_lu as blu
import scenarios
sim = sys.modules["differometor.simulate"]
def fac_of(name):
    if name == "uifo800-benchmark":
        W = bench._load_workload(101); a = W["arrays"]; q = np.asarray(W["params"])
    else:
        setup, changing, values, pairs, init, ref = scenarios._scenarios()[name][1]()
        a, *_ = sim.run_build_step(setup, changing, jnp.asarray(values), pairs)
        idx = np.asarray(a["optimized_value_indices"]); q = np.zeros(int(idx.max()) + 1)
        q[idx] = np.real(np.asarray(a["parameters"])[0, np.asarray(a["optimized_parameter_indices"])])
    plan = a["host_plan"]
    return host._forward(plan, q, np.asarray(a["parameters"]), np.asarray(a["carrier_changing_values"]), np.asarray(a["signal_changing_values"]).real)[3]["factors"]
for name in ("uifo800-benchmark", "voyager", "aligo"):
    fac = fac_of(name); F, nnz = fac.data.shape; n = fac.n
    indptr = np.concatenate([[0], np.cumsum(np.bincount(fac.columns, minlength=n))])
    ind = fac.rows.astype(np.intc); ptr = indptr.astype(np.intc)
    opts = dict(adjoint.SUPERLU_OPTIONS, ColPerm="NATURAL")
    t = time.perf_counter()
    ref = fac.factors[F // 2]   # SuperLU factor of the reference frequency (pivots only)
    t_ref = time.perf_counter() - t
    import cProfile, pstats; pr = cProfile.Profile(); pr.enable()
    t = time.perf_counter(); B = blu.BatchedLU(fac.data, fac.rows.astype(np.int64), indptr.astype(np.int64), ref.perm_r, ref.perm_c); t_build = time.perf_counter() - t
    pr.disable()
    if name == "uifo800-benchmark": pstats.Stats(pr).sort_stats("tottime").print_stats(6)
    t = time.perf_counter(); B = blu.BatchedLU(fac.data, fac.rows.astype(np.int64), indptr.astype(np.int64), ref.perm_r, ref.perm_c); t_build = time.perf_counter() - t
    t = time.perf_counter(); V, ok, worst = B._factor(fac.data); t_fact = time.perf_counter() - t
    print(f"== {name}: n={n} F={F} | reference gstrf {t_ref*1e3:.2f} ms, program build {t_build*1e3:.1f} ms "
          f"({len(B.steps)} elimination levels; solve levels L {len(B.L_fwd.groups)} U {len(B.U_bwd.groups)} Ut {len(B.Ut_fwd.groups)} Lt {len(B.Lt_bwd.groups)}), "
          f"batched factorisation {t_fact*1e3:.2f} ms | accepted {ok.sum()}/{F}, worst |l| {worst.max():.2f}")
    rng = np.random.default_rng(0)
    for trans, k in (("N", 1), ("T", 1), ("H", 1), ("H", 3)):
        rhs = rng.standard_normal((F, n, k)) + 1j * rng.standard_normal((F, n, k))
        t = time.perf_counter(); xb = B.solve(rhs, trans); tb = time.perf_counter() - t
        t = time.perf_counter(); xs = fac._solve(rhs, trans); ts = time.perf_counter() - t      # SuperLU, same (permuted) system
        acc = np.flatnonzero(ok)
        rel = np.max(np.abs(xb[acc] - xs[acc])) / np.max(np.abs(xs[acc]))
        # residual check (backward error) for both
        def resid(x):
            return np.max(np.abs(rhs - fac._apply(x, trans))) / np.max(np.abs(rhs))
        print(f"   solve {trans} k={k}: batched {tb*1e3:6.2f} ms vs SuperLU x{F} {ts*1e3:6.2f} ms | rel diff {rel:.1e} | "
              f"residual batched {resid(xb[...]) if ok.all() else float('nan'):.1e} SuperLU {resid(xs):.1e}")

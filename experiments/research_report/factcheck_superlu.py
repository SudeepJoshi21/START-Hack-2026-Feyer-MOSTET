import sys, time, subprocess
R = sys.argv[1]; sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
import bench
bench._setup_env(1)
import numpy as np
from differometor import host, adjoint
W = bench._load_workload(101); a = W["arrays"]; plan = a["host_plan"]
args = (np.asarray(W["params"]), np.asarray(a["parameters"]), np.asarray(a["carrier_changing_values"]), np.asarray(a["signal_changing_values"]).real)
c, s, n, st = host._forward(plan, *args)
fac = st["factors"]; F, nnz = fac.data.shape; N = fac.n
import scipy.sparse as sp, scipy.sparse.linalg as spla
indptr = np.concatenate([[0], np.cumsum(np.bincount(fac.columns, minlength=N))])
def csc(f): return sp.csc_matrix((fac.data[f], fac.rows, indptr), shape=(N, N))
# (a) exposed attributes
lu = spla.splu(csc(50), permc_spec="NATURAL", diag_pivot_thresh=1.0, options=dict(PanelSize=4, Relax=4))
print("(a) SuperLU attrs:", [x for x in ("L", "U", "perm_r", "perm_c", "nnz", "solve") if hasattr(lu, x)], "nnz(L)+nnz(U) =", lu.L.nnz + lu.U.nnz)
# (b,c) Fact / MC64 in subprocess
for opt in ("Fact='SamePattern_SameRowPerm'", "Fact='SamePattern'", "RowPerm='LargeDiag_MC64'"):
    code = (f"import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla; A = sp.random(50,50,0.1,format='csc',random_state=0)+sp.eye(50,format='csc');"
            f"lu = spla.splu(A, options=dict({opt})); print('ran, solve err', np.abs(A@lu.solve(np.ones(50))-1).max())")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    print(f"(b) splu options {opt}: exit {p.returncode}: {(p.stdout + p.stderr).strip().splitlines()[-1][:150] if (p.stdout+p.stderr).strip() else ''}")
# (d) per-call overhead
lu_mod = adjoint._superlu(); o = dict(adjoint.SUPERLU_OPTIONS, ColPerm="NATURAL")
tiny = sp.eye(3, format="csc", dtype=complex)
t = time.perf_counter()
for _ in range(200): lu_mod.gstrf(3, 3, tiny.data, tiny.indices.astype(np.intc), tiny.indptr.astype(np.intc), csc_construct_func=adjoint._no_csc, ilu=False, options=o)
print(f"(d) gstrf on 3x3: {(time.perf_counter()-t)/200*1e6:.0f} us/call; on our 751 system: ~{(time.perf_counter()*0)+0:.0f}", end="")
ind, ptr = fac.rows.astype(np.intc), indptr.astype(np.intc)
t = time.perf_counter()
for f in range(F): lu_mod.gstrf(N, nnz, fac.data[f], ind, ptr, csc_construct_func=adjoint._no_csc, ilu=False, options=o)
print(f" {(time.perf_counter()-t)/F*1e6:.0f} us/call")
b = np.ones(N, complex)
t = time.perf_counter()
for f in range(F): fac.factors[f].solve(b)
print(f"(d) solve per call: {(time.perf_counter()-t)/F*1e6:.0f} us")
# (e) levels of the elimination DAG (right-looking steps): level(k) = 1 + max level over k' with L[k,k'] != 0 or U[k',k] != 0
L, U = lu.L.tocsr(), lu.U.tocsc()
level = np.zeros(N, int)
for k in range(N):
    deps = set(L.indices[L.indptr[k]:L.indptr[k+1]]) | set(U.indices[U.indptr[k]:U.indptr[k+1]])
    deps.discard(k)
    level[k] = 1 + max((level[d] for d in deps), default=0)
Lc = lu.L.tocsc(); Ur = lu.U.tocsr()
triples = sum((Lc.indptr[k+1]-Lc.indptr[k]-1) * (Ur.indptr[k+1]-Ur.indptr[k]-1) for k in range(N))
print(f"(e) elimination levels: {level.max()} for n={N}; multiply-add triples: {triples}; level sizes (first 10) {np.bincount(level)[1:11].tolist()}")
# (f) pivot sequence of f=50 reused at every frequency: partial pivoting holds iff all |multipliers| <= 1
pr, pc = lu.perm_r, lu.perm_c
okcount, worst = 0, []
for f in range(F):
    A = csc(f)
    Ap = A[np.argsort(pr)][:, pc] if False else None
    lu_f = spla.splu(csc(f), permc_spec="NATURAL", diag_pivot_thresh=1.0, options=dict(PanelSize=4, Relax=4))
    same = np.array_equal(lu_f.perm_r, pr)
    okcount += same
print(f"(f) frequencies whose own partial-pivoting row order equals the f=50 reference: {okcount}/{F}")
# (g) block-diagonal single gstrf
blocks = sp.block_diag([csc(f) for f in range(F)], format="csc")
bd, bi, bp = blocks.data.astype(complex), blocks.indices.astype(np.intc), blocks.indptr.astype(np.intc)
for rep in range(2):
    t = time.perf_counter(); big = lu_mod.gstrf(F*N, bd.shape[0], bd, bi, bp, csc_construct_func=adjoint._no_csc, ilu=False, options=o); tb = time.perf_counter() - t
    t = time.perf_counter()
    small = [lu_mod.gstrf(N, nnz, fac.data[f], ind, ptr, csc_construct_func=adjoint._no_csc, ilu=False, options=o) for f in range(F)]; ts = time.perf_counter() - t
print(f"(g) factor: block-diagonal single call {tb*1e3:.1f} ms vs 101 calls {ts*1e3:.1f} ms")
B = np.random.default_rng(0).standard_normal((F, N)) + 0j
t = time.perf_counter(); xb = big.solve(B.ravel().copy()); tbs = time.perf_counter() - t
t = time.perf_counter(); xs = np.stack([small[f].solve(B[f].copy()) for f in range(F)]); tss = time.perf_counter() - t
print(f"(g) solve: block-diagonal {tbs*1e3:.2f} ms vs 101 calls {tss*1e3:.2f} ms; identical results: {np.array_equal(xb.reshape(F, N), xs)} (max rel diff {np.max(np.abs(xb.reshape(F,N)-xs))/np.max(np.abs(xs)):.1e})")

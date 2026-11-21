"""Unit tests for differometor/adjoint.py (WS1 + WS2). Run: python tests/test_adjoint.py  (or pytest)

Every test runs for BOTH solvers (SOLVER = "sparse" SuperLU callback, and "dense" XLA LU) and
compares the hand-written custom_vjp against JAX's own autodiff of the ORIGINAL formulation
(dense jnp.linalg.solve, A^H solve, third covariance solve, dense noise matrix) on random,
well-conditioned sparse complex systems with several frequencies and 2 detectors.
(The real fixture has cond(A) ~ 1e30; sparse vs dense agree there only to ~1e-8, which is
checked by `scripts/bench.py check`, not here.)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from differometor import adjoint
from differometor.adjoint import quadratic_noise, signal_pattern, solve_signal_and_noise_weights

SOLVERS = ("sparse", "dense")


def _problem(seed=0, n=48, extra=200, D=2, M=30, F=3):
    rng = np.random.default_rng(seed)
    flat = np.unique(np.r_[np.arange(n) * (n + 1), rng.integers(0, n * n, extra)])
    rows, cols = (flat // n).astype(np.int32), (flat % n).astype(np.int32)
    vals = rng.normal(size=(F, flat.size)) + 1j * rng.normal(size=(F, flat.size))
    vals[:, rows == cols] += 10.0
    b = rng.normal(size=(F, n)) + 1j * rng.normal(size=(F, n))
    C = rng.normal(size=(F, D, n)) + 1j * rng.normal(size=(F, D, n))
    qflat = rng.choice(n * n, M, replace=False)
    q_rows, q_cols = (qflat // n).astype(np.int32), (qflat % n).astype(np.int32)
    q_vals = rng.normal(size=(F, M)) + 1j * rng.normal(size=(F, M))
    return dict(vals=jnp.asarray(vals), rows=jnp.asarray(rows), cols=jnp.asarray(cols), b=jnp.asarray(b),
                C=jnp.asarray(C), q_vals=jnp.asarray(q_vals), q_rows=jnp.asarray(q_rows),
                q_cols=jnp.asarray(q_cols), n=n)


def _reference(vals, b, C, q_vals, P):
    """Original formulation, per frequency: dense solves, A^H solve, THIRD covariance solve."""
    n = P["n"]

    def one(v, bb, CC, q):
        A = jnp.zeros((n, n), complex).at[P["rows"], P["cols"]].set(v)
        Q = jnp.zeros((n, n), complex).at[P["q_rows"], P["q_cols"]].set(q)
        x = jnp.linalg.solve(A, bb)
        W = jax.vmap(lambda c: jnp.linalg.solve(jnp.conj(A).T, c))(CC)
        cov = jax.vmap(lambda w: jnp.linalg.solve(A, Q @ w))(W)
        return x, jnp.sum(jnp.real(cov * jnp.conj(CC)), axis=-1)

    return jax.vmap(one)(vals, b, C, q_vals)


def _candidate(vals, b, C, q_vals, P):
    x, W = solve_signal_and_noise_weights(vals, P["rows"], P["cols"], b, C)
    S = jax.vmap(quadratic_noise, in_axes=(0, 0, None, None))(W, q_vals, P["q_rows"], P["q_cols"])
    return x, S


def _loss(f, P):
    # Real loss touching every output like readout(): |signal|, log of noise, real/imag parts.
    def L(vals, b, C, q_vals):
        x, S = f(vals, b, C, q_vals, P)
        return (jnp.sum(jnp.log(jnp.abs(x[:, :7]) ** 2)) + jnp.sum(jnp.real(x) * jnp.imag(x))
                + jnp.sum(jnp.log(jnp.abs(S))))
    return L


def _args(P):
    return P["vals"], P["b"], P["C"], P["q_vals"]


def test_forward_matches():
    P = _problem()
    x0, S0 = _reference(*_args(P), P)
    for solver in SOLVERS:
        adjoint.SOLVER = solver
        x1, S1 = _candidate(*_args(P), P)
        assert np.allclose(x0, x1, rtol=1e-11, atol=1e-13), (solver, float(np.abs(x0 - x1).max()))
        assert np.allclose(S0, S1, rtol=1e-10), (solver, float(np.abs(S0 - S1).max()))
    adjoint.SOLVER = "sparse"


def test_gradients_match_autodiff():
    for seed in range(3):
        P = _problem(seed)
        g0 = jax.grad(_loss(_reference, P), argnums=(0, 1, 2, 3))(*_args(P))
        for solver in SOLVERS:
            adjoint.SOLVER = solver
            g1 = jax.grad(_loss(_candidate, P), argnums=(0, 1, 2, 3))(*_args(P))
            for name, a, c in zip(("vals", "b", "C", "q_vals"), g0, g1):
                err = float(jnp.max(jnp.abs(a - c)) / jnp.max(jnp.abs(a)))
                assert err < 1e-10, f"{solver} seed {seed}: grad wrt {name} rel err {err:.2e}"
    adjoint.SOLVER = "sparse"


def test_sparse_equals_dense():
    P = _problem(seed=7, F=4)
    out = {}
    for solver in SOLVERS:
        adjoint.SOLVER = solver
        out[solver] = (_candidate(*_args(P), P), jax.grad(_loss(_candidate, P))(*_args(P)))
    (xs, Ss), gs = out["sparse"]
    (xd, Sd), gd = out["dense"]
    assert float(jnp.max(jnp.abs(xs - xd)) / jnp.max(jnp.abs(xd))) < 1e-11
    assert float(jnp.max(jnp.abs(gs - gd)) / jnp.max(jnp.abs(gd))) < 1e-10
    adjoint.SOLVER = "sparse"


def test_jit_vjp_like_benchmark():
    """Under jit + jax.vjp with the pullback returned and called in a second jit (benchmark style)."""
    P = _problem()
    loss = _loss(_candidate, P)
    g_ref = jax.grad(_loss(_reference, P))(*_args(P))
    for solver in SOLVERS:
        adjoint.SOLVER = solver
        out, pullback = jax.jit(lambda v: jax.vjp(lambda w: loss(w, *_args(P)[1:]), v))(P["vals"])
        g = jax.jit(lambda pb: pb(jnp.ones(()))[0])(pullback)
        assert np.allclose(out, loss(*_args(P)))
        assert float(jnp.max(jnp.abs(g - g_ref)) / jnp.max(jnp.abs(g_ref))) < 1e-10, solver
        if solver == "sparse":   # tape holds no LU factors: nothing of size F*n*n
            sizes = [l.size for l in jax.tree_util.tree_leaves(pullback)]
            assert max(sizes) < P["vals"].shape[0] * P["n"] ** 2, max(sizes)
    adjoint.SOLVER = "sparse"


def test_signal_pattern_wraps_indices():
    base = np.zeros((4, 5), complex); base[0, 0] = base[1, 1] = 1
    # (2,-1) is the RHS column -> excluded; (0,0) duplicate; (-1,2) == (3,2) after row wrap.
    idx = np.array([[2, 3, 0, -1], [-1, 2, 0, 2]])
    r, c = signal_pattern(base, idx)
    assert list(zip(r.tolist(), c.tolist())) == [(0, 0), (1, 1), (3, 2)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)

"""Unit tests for differometor/adjoint.py (WS2). Run: python tests/test_adjoint.py  (or pytest)

Each test compares the hand-written custom_vjp against JAX's own autodiff of the ORIGINAL
formulation (dense jnp.linalg.solve, three solves, dense noise matrix) on a random sparse
complex system with 2 detectors and a non-trivial real loss.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from differometor.adjoint import quadratic_noise, signal_pattern, solve_signal_and_noise_weights


def _problem(seed=0, n=48, extra=200, D=2, M=30):
    rng = np.random.default_rng(seed)
    flat = np.unique(np.r_[np.arange(n) * (n + 1), rng.integers(0, n * n, extra)])
    rows, cols = (flat // n).astype(np.int32), (flat % n).astype(np.int32)
    vals = rng.normal(size=flat.size) + 1j * rng.normal(size=flat.size)
    vals[rows == cols] += 10.0
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    C = rng.normal(size=(D, n)) + 1j * rng.normal(size=(D, n))
    qflat = rng.choice(n * n, M, replace=False)
    q_rows, q_cols = (qflat // n).astype(np.int32), (qflat % n).astype(np.int32)
    q_vals = rng.normal(size=M) + 1j * rng.normal(size=M)
    q_base = np.zeros((n, n), complex)
    q_base[rng.integers(0, n, 5), rng.integers(0, n, 5)] = 0.3 + 0.1j
    q_base[q_rows, q_cols] = 0.0                     # simulate() masks updated positions
    return dict(vals=jnp.asarray(vals), rows=jnp.asarray(rows), cols=jnp.asarray(cols), b=jnp.asarray(b),
                C=jnp.asarray(C), q_vals=jnp.asarray(q_vals), q_rows=jnp.asarray(q_rows),
                q_cols=jnp.asarray(q_cols), q_base=jnp.asarray(q_base), n=n)


def _reference(vals, b, C, q_vals, P):
    """Original formulation: dense solves, w via A^H, covariance via a THIRD solve."""
    n = P["n"]
    A = jnp.zeros((n, n), complex).at[P["rows"], P["cols"]].set(vals)
    Q = P["q_base"].at[P["q_rows"], P["q_cols"]].set(q_vals)
    x = jnp.linalg.solve(A, b)
    W = jax.vmap(lambda c: jnp.linalg.solve(jnp.conj(A).T, c))(C)
    cov = jax.vmap(lambda w: jnp.linalg.solve(A, Q @ w))(W)
    S = jnp.sum(jnp.real(cov * jnp.conj(C)), axis=-1)
    return x, S


def _candidate(vals, b, C, q_vals, P):
    x, W = solve_signal_and_noise_weights(vals, P["rows"], P["cols"], b, C)
    S = quadratic_noise(W, q_vals, P["q_rows"], P["q_cols"], P["q_base"])
    return x, S


def _loss(f, P):
    # Real loss touching every output like readout(): |signal|, log of noise, phases via real/imag.
    def L(vals, b, C, q_vals):
        x, S = f(vals, b, C, q_vals, P)
        return (jnp.sum(jnp.log(jnp.abs(x[:7]) ** 2)) + jnp.sum(jnp.real(x) * jnp.imag(x))
                + jnp.sum(jnp.log(jnp.abs(S))))
    return L


def test_forward_matches():
    P = _problem()
    args = (P["vals"], P["b"], P["C"], P["q_vals"])
    (x0, S0), (x1, S1) = _reference(*args, P), _candidate(*args, P)
    assert np.allclose(x0, x1, rtol=1e-12, atol=1e-14), np.abs(x0 - x1).max()
    assert np.allclose(S0, S1, rtol=1e-10), np.abs(S0 - S1).max()


def test_gradients_match_autodiff():
    for seed in range(3):
        P = _problem(seed)
        args = (P["vals"], P["b"], P["C"], P["q_vals"])
        g0 = jax.grad(_loss(_reference, P), argnums=(0, 1, 2, 3))(*args)
        g1 = jax.grad(_loss(_candidate, P), argnums=(0, 1, 2, 3))(*args)
        for name, a, c in zip(("vals", "b", "C", "q_vals"), g0, g1):
            err = float(jnp.max(jnp.abs(a - c)) / jnp.max(jnp.abs(a)))
            assert err < 1e-10, f"seed {seed}: grad wrt {name} rel err {err:.2e}"


def test_jit_vmap_vjp_like_benchmark():
    """Batched over frequencies, under jit + jax.vjp with the pullback returned (benchmark style)."""
    P = _problem()
    F = 3
    vals = jnp.stack([P["vals"] * (1 + 0.01 * k) for k in range(F)])
    bs, Cs = jnp.stack([P["b"]] * F), jnp.stack([P["C"]] * F)

    def f(vals):
        x, W = jax.vmap(solve_signal_and_noise_weights, in_axes=(0, None, None, 0, 0))(vals, P["rows"], P["cols"], bs, Cs)
        return jnp.sum(jnp.abs(x) ** 2) + jnp.sum(jnp.abs(W))

    def ref(vals):
        A = jax.vmap(lambda v: jnp.zeros((P["n"],) * 2, complex).at[P["rows"], P["cols"]].set(v))(vals)
        x = jax.vmap(jnp.linalg.solve)(A, bs)
        W = jax.vmap(lambda a, C: jax.vmap(lambda c: jnp.linalg.solve(jnp.conj(a).T, c))(C))(A, Cs)
        return jnp.sum(jnp.abs(x) ** 2) + jnp.sum(jnp.abs(W))

    out, pullback = jax.jit(lambda v: jax.vjp(f, v))(vals)
    g = jax.jit(lambda pb: pb(jnp.ones(()))[0])(pullback)
    g_ref = jax.grad(ref)(vals)
    assert np.allclose(out, ref(vals))
    assert float(jnp.max(jnp.abs(g - g_ref)) / jnp.max(jnp.abs(g_ref))) < 1e-10


def test_signal_pattern_normalises_rhs_column():
    base = np.zeros((4, 5), complex); base[0, 0] = base[1, 1] = 1
    idx = np.array([[2, 3, 0], [-1, 2, 0]])       # (2,-1) is the RHS column -> excluded; (0,0) dup
    r, c = signal_pattern(base, idx)
    assert list(zip(r.tolist(), c.tolist())) == [(0, 0), (1, 1), (3, 2)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)

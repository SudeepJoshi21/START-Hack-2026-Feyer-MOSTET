"""
Hand-written reverse-mode gradient (adjoint method) for the signal and noise solves.

Workstreams WS2 (gradient) + WS1 (sparse solver) of OPTIMIZATION_PLAN.md.
Background: docs/WS2_GRADIENT_EXPLAINED.md. Change log: docs/WS2_CHANGELOG.md.
Merge plan: docs/WS1_WS2_INTEGRATION.md.

Why this exists
---------------
`benchmark.py` differentiates `simulate` with `jax.vjp`, which stores every intermediate
array for the backward pass (the "tape") and builds the derivative w.r.t. the signal
matrix as a dense 751x751 array per frequency. Here the two signal-matrix solves get a
`jax.custom_vjp`, over all frequencies at once:

    forward : per frequency ONE factorisation of A (sparse SuperLU by default), then
              x = A^-1 b        (signal)
              W = A^-H C        (noise weights, one row per detector)
    tape    : sparse values of A + x + W  (sparse solver: no factors kept)
    backward: refactorise, then
              lam_x = A^-T xbar,  lam_u = A^-1 conj(Wbar)
              Abar[k] = -lam_x[r_k] x[c_k] - sum_d u_d[r_k] lam_u,d[c_k],  u = conj(W)
              bbar = lam_x,  Cbar = conj(lam_u)
    The matrix cotangent exists only on the non-zero pattern (2,597 entries, not 564,001).

The third solve of the original code (covariances = A^-1 Q w) is not needed:
    noise sum  = Re(c^H A^-1 Q w) = Re(w^H Q w)   because  c^H A^-1 = (A^-H c)^H = w^H.

Conventions (verified in tests/test_adjoint.py): JAX propagates cotangents of holomorphic
maps with the plain TRANSPOSE, so the adjoint of x = A^-1 b uses A^-T, not A^-H.

Solvers (module switch `SOLVER`)
--------------------------------
"sparse": SciPy SuperLU (COLAMD ordering) on the host via `jax.pure_callback`; ~0.6 ms per
          factorisation of the 751 system. Ported from WS1 (`ws1` @ 095df6e).
"dense" : `jax.scipy.linalg.lu_factor` / `lu_solve` inside XLA; the LU factors are kept on
          the tape (~0.9 GB at 101 frequencies). For A/B tests and as a fallback.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import lu_factor, lu_solve

# Set to False to force the original dense-autodiff path in `simulate` (for A/B tests).
ENABLED = True
# "sparse" (SciPy SuperLU host callback) or "dense" (XLA LU). See module docstring.
SOLVER = "sparse"


# ---------------------------------------------------------------------------------------
# Build time (NumPy only)
# ---------------------------------------------------------------------------------------

def signal_pattern(signal_matrix, signal_matrix_indices) -> tuple[np.ndarray, np.ndarray]:
    """
    Positions (rows, cols) that can be non-zero in the signal system matrix A.

    A = signal_matrix[:, :n] after `update` writes into it, so its non-zeros are a subset
    of: non-zeros of the base matrix  U  positions written by `update`. (The carrier
    scaling and the conjugate-block updates only rescale existing entries.)
    `update` uses column -1 for the right-hand side (augmented column n); indices are
    wrapped (rows mod n, cols mod n+1) and the RHS column is excluded (b is dense).

    Returns sorted, unique int32 arrays of equal length nnz.
    """
    base = np.asarray(signal_matrix)
    n = base.shape[0]
    rows, cols = (np.asarray(x) for x in signal_matrix_indices)
    rows = np.mod(rows, n)                          # guard: negative rows would duplicate entries
    cols = np.mod(cols, n + 1)                      # -1 -> n (right-hand side column)
    in_matrix = cols < n
    flat = np.concatenate([np.flatnonzero(base[:, :n]), rows[in_matrix] * n + cols[in_matrix]])
    flat = np.unique(flat)                          # unique => scatter-set == scatter-add
    return (flat // n).astype(np.int32), (flat % n).astype(np.int32)


def pattern_arrays(simulation_arrays: dict) -> dict:
    """
    Extra keyword arguments for `simulate` that switch on the adjoint path.

    Called by `run_build_step`. Returns {} (=> original dense path) if a precondition of
    the adjoint path does not hold, so unusual setups keep working unchanged:
      - noise matrix and signal system have the same size,
      - noise-matrix update indices are non-negative and unique (sparse sum == `.at[].set`).

    Keys:
      signal_pattern_rows/cols : (nnz,) int32   sparsity pattern of A
      noise_base_rows/cols     : (K,) int32     non-zeros of the base noise matrix that
                                                `update` does not overwrite (K = 0 here)
      noise_parameter_rows     : (1,) or (R,)   [0] if no swept parameter feeds a noise
                                                function (noise values are then computed
                                                once and broadcast), else all R rows
    """
    signal_matrix = np.asarray(simulation_arrays["signal_matrix"])
    noise_matrix = np.asarray(simulation_arrays["noise_matrix"])
    n = signal_matrix.shape[0]
    if noise_matrix.shape != (n, n):
        return {}
    q_rows, q_cols = (np.asarray(x) for x in simulation_arrays["noise_system_matrix_indices"])
    if q_rows.size and (q_rows.min() < 0 or q_cols.min() < 0):     # segment_sum needs plain indices
        return {}
    if np.unique(q_rows * n + q_cols).size != q_rows.size:
        return {}
    rows, cols = signal_pattern(signal_matrix, simulation_arrays["signal_matrix_indices"])

    # Base noise matrix with the update positions removed (exactly what `.at[].set` leaves).
    base = noise_matrix.copy()
    base[q_rows, q_cols] = 0
    base_rows, base_cols = np.nonzero(base)

    # Noise functions (vacuum, loss, squeezer) do not depend on frequency. Unless a swept
    # parameter is one of their inputs, all expanded parameter rows give the same values.
    # (Idea and check from WS1, `_structure_arrays` on `ws1`.)
    swept = set(np.asarray(simulation_arrays["carrier_changing_parameter_indices"]).ravel().tolist())
    swept |= set(np.asarray(simulation_arrays["signal_changing_parameter_indices"]).ravel().tolist())
    noise_inputs = set(np.asarray(simulation_arrays["noise_function_input_indices"]).ravel().tolist())
    if swept & noise_inputs:
        row_count = (np.asarray(simulation_arrays["carrier_changing_values"]).shape[1]
                     * np.asarray(simulation_arrays["signal_changing_values"]).shape[1])
        noise_parameter_rows = np.arange(row_count, dtype=np.int32)
    else:
        noise_parameter_rows = np.zeros(1, dtype=np.int32)

    return {
        "signal_pattern_rows": jnp.asarray(rows),
        "signal_pattern_cols": jnp.asarray(cols),
        "noise_base_rows": jnp.asarray(base_rows.astype(np.int32)),
        "noise_base_cols": jnp.asarray(base_cols.astype(np.int32)),
        "noise_parameter_rows": jnp.asarray(noise_parameter_rows),
    }


# ---------------------------------------------------------------------------------------
# Solver kernels: forward (x, W) and adjoint (lam_x, lam_u), batched over frequencies
#   vals (F, nnz), b (F, n), C (F, D, n), xbar (F, n), Wbar (F, D, n)
# ---------------------------------------------------------------------------------------

def _csc_structure(rows, cols, n):
    """Column-major order of the pattern: (order, row indices, column pointers). From WS1."""
    order = np.lexsort((rows, cols))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, cols + 1, 1)
    return order, rows[order], np.cumsum(indptr)


REFINEMENT_STEPS = 1
"""
Iterative-refinement steps after every SuperLU solve (x += A^-1 (b - A x)).

Without it the SuperLU noise weights of Voyager (an example setup, not scored) differ from the dense LU by up to
3.6e-3 (sensitivity |d ln S| 4.2e-3); one step brings that to 2.8e-13, and the scored setup from 3.0e-8 to 4.5e-9.
Cost: one sparse product + one triangular solve per frequency.
"""


class _Refined:
    """SuperLU factorisation of A plus the matrix itself, with refined solves for op(A) in {A, A^T, A^H}."""

    def __init__(self, matrix):
        from scipy.sparse.linalg import splu

        self.matrix = matrix
        self.lu = splu(matrix, permc_spec="COLAMD")

    def _apply(self, x, trans):
        # A^T x = (x^T A)^T and A^H x = conj(A^T conj(x)): no transposed copies of A are built.
        if trans == "N":
            return self.matrix @ x
        if trans == "T":
            return (self.matrix.T @ x)
        return np.conj(self.matrix.T @ np.conj(x))

    def solve(self, rhs, trans):
        x = self.lu.solve(rhs, trans=trans)
        for _ in range(REFINEMENT_STEPS):
            x = x + self.lu.solve(rhs - self._apply(x, trans), trans=trans)
        return x


def _host_factorisations(vals, rows, cols, n):
    """Yield one SuperLU factorisation per frequency (COLAMD ordering, as validated by WS1), with refined solves."""
    from scipy.sparse import csc_matrix

    order, indices, indptr = _csc_structure(np.asarray(rows), np.asarray(cols), n)
    vals = np.asarray(vals)
    for f in range(vals.shape[0]):
        yield f, _Refined(csc_matrix((vals[f][order], indices, indptr), shape=(n, n)))


def _host_forward(vals, rows, cols, b, C):
    b, C = np.asarray(b), np.asarray(C)
    n = b.shape[-1]
    x = np.empty(b.shape, dtype=np.complex128)
    W = np.empty(C.shape, dtype=np.complex128)
    for f, lu in _host_factorisations(vals, rows, cols, n):
        x[f] = lu.solve(b[f], trans="N")                 # A x = b
        if C.shape[1]:
            W[f] = lu.solve(C[f].T.copy(), trans="H").T  # A^H w_d = c_d, all detectors at once
    return x, W


def _host_adjoint(vals, rows, cols, xbar, ubar):
    xbar, ubar = np.asarray(xbar), np.asarray(ubar)
    n = xbar.shape[-1]
    lam_x = np.empty(xbar.shape, dtype=np.complex128)
    lam_u = np.empty(ubar.shape, dtype=np.complex128)
    for f, lu in _host_factorisations(vals, rows, cols, n):   # refactorise: ~0.6 ms
        lam_x[f] = lu.solve(xbar[f], trans="T")               # A^T lam_x = xbar
        if ubar.shape[1]:
            lam_u[f] = lu.solve(ubar[f].T.copy(), trans="N").T  # A lam_u = ubar
    return lam_x, lam_u


def _callback(fn, out_like, vals, rows, cols, rhs1, rhs2):
    """
    Run `fn(vals, rows, cols, rhs1, rhs2)` on the host.

    Deadlock guard (found at 101 frequencies with the benchmark's split forward/pullback programs and
    the process pinned to one CPU): in JAX 0.9, `pure_callback` hands the callback `jax.Array`s, and
    `np.asarray` on an operand that is (an alias of) an *input buffer of the running program* waits on
    that program, which is waiting for the callback. So every operand is passed packed into freshly
    computed buffers (`concatenate` always materialises a new buffer) and unpacked in NumPy.
    """
    shapes = tuple(jax.ShapeDtypeStruct(a.shape, jnp.complex128) for a in out_like)
    frequencies, nnz = vals.shape
    shape1, shape2 = rhs1.shape, rhs2.shape
    packed = jnp.concatenate(
        [vals, rhs1.reshape(frequencies, -1), rhs2.reshape(frequencies, -1)], axis=1
    )                                                               # (F, nnz + |rhs1| + |rhs2|) complex
    index = jnp.concatenate([rows.astype(jnp.int32), cols.astype(jnp.int32)])   # (2 nnz,) int32
    split1 = nnz + int(np.prod(shape1[1:]))

    def host(packed, index):
        packed, index = np.asarray(packed), np.asarray(index)
        return fn(
            packed[:, :nnz],
            index[:nnz],
            index[nnz:],
            packed[:, nnz:split1].reshape(shape1),
            packed[:, split1:].reshape(shape2),
        )

    return jax.pure_callback(host, shapes, packed, index, vmap_method="sequential")


def _sparse_forward(vals, rows, cols, b, C):
    x, W = _callback(_host_forward, (b, C), vals, rows, cols, b, C)
    return x, W, vals                                   # tape: values only


def _sparse_adjoint(saved, rows, cols, xbar, ubar):
    vals = saved
    return _callback(_host_adjoint, (xbar, ubar), vals, rows, cols, xbar, ubar)


def _dense_forward(vals, rows, cols, b, C):
    n = b.shape[-1]
    A = vmap(lambda v: jnp.zeros((n, n), jnp.complex128).at[rows, cols].set(v))(vals)
    lu, piv = vmap(lu_factor)(A)
    x = vmap(lambda l, p, bb: lu_solve((l, p), bb, trans=0))(lu, piv, b)
    W = vmap(lambda l, p, CC: vmap(lambda c: lu_solve((l, p), c, trans=2))(CC))(lu, piv, C)
    return x, W, (lu, piv)                              # tape: LU factors


def _dense_adjoint(saved, rows, cols, xbar, ubar):
    lu, piv = saved
    lam_x = vmap(lambda l, p, xb: lu_solve((l, p), xb, trans=1))(lu, piv, xbar)
    lam_u = vmap(lambda l, p, U: vmap(lambda u: lu_solve((l, p), u, trans=0))(U))(lu, piv, ubar)
    return lam_x, lam_u


def _kernels():
    if SOLVER == "sparse":
        return _sparse_forward, _sparse_adjoint
    if SOLVER == "dense":
        return _dense_forward, _dense_adjoint
    raise ValueError(f"adjoint.SOLVER must be 'sparse' or 'dense', got {SOLVER!r}")


# ---------------------------------------------------------------------------------------
# The custom-gradient solve (all frequencies at once)
# ---------------------------------------------------------------------------------------

@jax.custom_vjp
def solve_signal_and_noise_weights(vals, rows, cols, b, C):
    """
    x[f] = A_f^-1 b[f] and W[f, d] = A_f^-H C[f, d], with A_f[rows, cols] = vals[f].

    vals : (F, nnz) complex   values of A on the static pattern (rows, cols)
    rows, cols : (nnz,) int   pattern (not differentiated)
    b    : (F, n) complex     right-hand sides of the signal system
    C    : (F, D, n) complex  detector selection vectors
    returns x (F, n), W (F, D, n)
    """
    forward, _ = _kernels()
    x, W, _ = forward(*_cast(vals, b, C, rows, cols))
    return x, W


def _cast(vals, b, C, rows, cols):
    return vals.astype(jnp.complex128), rows, cols, b.astype(jnp.complex128), C.astype(jnp.complex128)


def _fwd(vals, rows, cols, b, C):
    forward, _ = _kernels()
    vals, rows, cols, b, C = _cast(vals, b, C, rows, cols)
    x, W, saved = forward(vals, rows, cols, b, C)
    return (x, W), (saved, rows, cols, x, W)


def _bwd(residuals, cotangents):
    _, adjoint = _kernels()
    saved, rows, cols, x, W = residuals
    xbar, Wbar = cotangents
    # Signal x = A^-1 b:            lam_x = A^-T xbar,  Abar -= lam_x x^T
    # Weights W = A^-H C, u = conj(W) = A^-T conj(C):  lam_u = A^-1 conj(Wbar),  Abar -= u lam_u^T
    lam_x, lam_u = adjoint(saved, rows, cols, xbar.astype(jnp.complex128), jnp.conj(Wbar).astype(jnp.complex128))
    u = jnp.conj(W)
    vals_bar = -lam_x[:, rows] * x[:, cols] - jnp.sum(u[:, :, rows] * lam_u[:, :, cols], axis=1)
    return vals_bar, None, None, lam_x, jnp.conj(lam_u)


solve_signal_and_noise_weights.defvjp(_fwd, _bwd)


# ---------------------------------------------------------------------------------------
# Noise from the weights: S = Re(w^H Q w), Q sparse
# ---------------------------------------------------------------------------------------

def quadratic_noise(W, q_vals, q_rows, q_cols):
    """
    S[d] = Re( W[d]^H Q W[d] )  for one frequency, Q given by its non-zeros.

    W : (D, n); q_vals : (M,) values at unique positions (q_rows, q_cols).
    """
    n = W.shape[-1]
    Qw = vmap(lambda w: jax.ops.segment_sum(q_vals * w[q_cols], q_rows, num_segments=n))(W)
    return jnp.real(jnp.sum(jnp.conj(W) * Qw, axis=-1))

"""
Hand-written reverse-mode gradient (adjoint method) for the signal and noise solves.

Workstream WS2 of OPTIMIZATION_PLAN.md. Background: docs/WS2_GRADIENT_EXPLAINED.md.
Change log: docs/WS2_CHANGELOG.md.

Why this exists
---------------
`benchmark.py` differentiates `simulate` with `jax.vjp`, which stores every intermediate
array for the backward pass (the "tape", ~9 GiB at 101 frequencies) and builds the
derivative w.r.t. the signal matrix as a dense 751x751 array per frequency. Here the
two signal-matrix solves get a `jax.custom_vjp` instead:

    forward : one LU factorisation of A per frequency, then
              x = A^-1 b        (signal)
              W = A^-H C        (noise weights, one row per detector)
    tape    : the LU factors + x + W (no dense matrix copies, no dense cotangents)
    backward: two triangular solves with the SAME LU, and the matrix cotangent only on
              the non-zero pattern of A:
              lam_x = A^-T xbar,  lam_u = A^-1 conj(Wbar)
              Abar[k] = -lam_x[r_k] x[c_k] - sum_d u_d[r_k] lam_u,d[c_k],  u = conj(W)
              bbar = lam_x,  Cbar = conj(lam_u)

The third solve of the original code (covariances = A^-1 Q w) is not needed:
    noise sum  = Re(c^H A^-1 Q w) = Re(w^H Q w)   because  c^H A^-1 = (A^-H c)^H = w^H.
(identity from SIMULATOR_EXPLAINED.md section 7; plan finding 3: 2e-8 relative change.)

Conventions (verified numerically in tests/test_adjoint.py):
JAX propagates cotangents of holomorphic maps with the plain TRANSPOSE, so the adjoint
solve for x = A^-1 b uses A^-T (trans=1), not A^-H.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import lu_factor, lu_solve

# Set to False to force the original dense-autodiff path in `simulate` (for A/B tests).
ENABLED = True


# ---------------------------------------------------------------------------------------
# Build time (NumPy only): static sparsity pattern of the signal matrix
# ---------------------------------------------------------------------------------------

def signal_pattern(signal_matrix, signal_matrix_indices) -> tuple[np.ndarray, np.ndarray]:
    """
    Positions (rows, cols) that can be non-zero in the signal system matrix A.

    A = signal_matrix[:, :n] after `update` writes into it, so its non-zeros are a subset
    of: non-zeros of the base matrix  U  positions written by `update`.
    `update` uses column -1 for the right-hand side (augmented column n); those are
    normalised and excluded here (b is handled separately as a dense vector).

    Returns sorted, unique int32 arrays of equal length nnz.
    """
    base = np.asarray(signal_matrix)
    n = base.shape[0]
    rows, cols = (np.asarray(x) for x in signal_matrix_indices)
    cols = np.mod(cols, n + 1)                     # -1 -> n (right-hand side column)
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
      - noise-matrix update positions are unique (so a sparse sum equals `.at[].set`).
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
    return {"signal_pattern_rows": jnp.asarray(rows), "signal_pattern_cols": jnp.asarray(cols)}


# ---------------------------------------------------------------------------------------
# The custom-gradient solve (one frequency; vmapped by the caller)
# ---------------------------------------------------------------------------------------

def _dense(vals, rows, cols, n):
    return jnp.zeros((n, n), dtype=jnp.complex128).at[rows, cols].set(vals)


@jax.custom_vjp
def solve_signal_and_noise_weights(vals, rows, cols, b, C):
    """
    x = A^-1 b and W[d] = A^-H C[d] for the matrix A with A[rows, cols] = vals.

    vals : (nnz,) complex   values of A on the static pattern (rows, cols)
    rows, cols : (nnz,) int  pattern (not differentiated)
    b    : (n,) complex      right-hand side of the signal system
    C    : (D, n) complex    detector selection vectors (one per detector)
    returns x (n,), W (D, n)
    """
    x, W, _ = _forward(vals, rows, cols, b, C)
    return x, W


def _forward(vals, rows, cols, b, C):
    n = b.shape[0]
    factors = lu_factor(_dense(vals.astype(jnp.complex128), rows, cols, n))   # the ONLY factorisation
    x = lu_solve(factors, b.astype(jnp.complex128), trans=0)                  # A x = b
    W = vmap(lambda c: lu_solve(factors, c, trans=2))(C.astype(jnp.complex128))  # A^H w = c
    return x, W, factors


def _fwd(vals, rows, cols, b, C):
    x, W, factors = _forward(vals, rows, cols, b, C)
    return (x, W), (factors, rows, cols, x, W)


def _bwd(residuals, cotangents):
    (lu, piv), rows, cols, x, W = residuals
    xbar, Wbar = cotangents
    # Signal: x = A^-1 b.
    lam_x = lu_solve((lu, piv), xbar, trans=1)                    # A^-T xbar
    vals_bar = -lam_x[rows] * x[cols]
    # Noise weights: W = A^-H C  <=>  u = conj(W) = A^-T conj(C).
    u = jnp.conj(W)
    lam_u = vmap(lambda ub: lu_solve((lu, piv), ub, trans=0))(jnp.conj(Wbar))   # A^-1 ubar
    vals_bar = vals_bar - jnp.sum(u[:, rows] * lam_u[:, cols], axis=0)
    C_bar = jnp.conj(lam_u)
    return vals_bar, None, None, lam_x, C_bar


solve_signal_and_noise_weights.defvjp(_fwd, _bwd)


# ---------------------------------------------------------------------------------------
# Noise from the weights: S = Re(w^H Q w), Q sparse
# ---------------------------------------------------------------------------------------

def quadratic_noise(W, q_vals, q_rows, q_cols, q_base):
    """
    S[d] = Re( W[d]^H Q W[d] )  for one frequency.

    Q = q_base with positions (q_rows, q_cols) overwritten by q_vals (exactly what
    `update` builds densely). Positions are unique (checked at build time).
    W : (D, n); q_vals : (M,); q_base : (n, n) dense, usually all zeros.
    """
    n = W.shape[-1]
    Qw = vmap(lambda w: jax.ops.segment_sum(q_vals * w[q_cols], q_rows, num_segments=n))(W)
    Qw = Qw + W @ q_base.T                      # base part (zeros in the public fixture)
    return jnp.real(jnp.sum(jnp.conj(W) * Qw, axis=-1))

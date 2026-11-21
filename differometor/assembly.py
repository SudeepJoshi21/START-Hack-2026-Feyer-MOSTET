"""
Matrix assembly without `vmap(lax.switch)` and without dense intermediate copies (workstream WS3).

Why
---
`simulate.update` evaluates the component functions with `vmap(lax.switch)` over a *batched* function index. JAX
lowers that to "evaluate all 19 branches for every call, then select", so each matrix fill traces, compiles and runs
every component function (carrier: 77 calls, signal: 679, noise: 66; only 3/11/3 distinct types). The signal matrix is
then built densely, (F, 751, 752) complex = 0.9 GB at 101 frequencies, and rewritten three more times: carrier scaling
of the signal connectors plus three conjugated blocks.

What this module does
---------------------
1. **Grouped evaluation.** At build time (NumPy) the calls are sorted by function type. At run time each type's
   function is vmapped over its own calls only, and the per-type results are concatenated. The type → function mapping is
   fixed by the *position* in a tuple, and the call count by the array *shape*, so both are static under `jit`
   although the index values are traced arguments.
2. **Signal system on its sparsity pattern.** Every position that can be non-zero in the augmented signal matrix
   is known at build time: base non-zeros ∪ `update` writes ∪ carrier-scaling positions. For each position the
   sequence "base value or component output → optional scaling by −carrier → optional conjugation" is
   recorded as four small arrays. At run time the system becomes `values (F, nnz)` on the pattern (row-major,
   matching `adjoint.signal_pattern` on `ws2-adjoint`) and a right-hand side `rhs (F, N)`.
   No dense intermediate is built. `simulate` only densifies once, for the dense solver it still uses.

`assembly_arrays` returns `{}` if a precondition fails (duplicate write positions, unexpected output width). `simulate`
then runs the original code unchanged, so unusual setups keep working.

Interfaces for WS1/WS2: `signal_values_and_rhs` returns exactly the `(values, rhs)` that WS2's
`solve_signal_and_noise_weights(values, rows, cols, rhs, C)` and WS1's sparse SuperLU kernel consume.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import vmap

from differometor.components import FUNCTIONS

ENABLED = True
"""Set to False (before building) to make `run_build_step` emit no assembly plan, i.e. use the original code."""

OUTPUT_WIDTH = 4
"""Every component function returns `standardize_output(...)`, a length-4 complex vector."""


# ---------------------------------------------------------------------------------------
# Build time (NumPy only)
# ---------------------------------------------------------------------------------------

def _group_calls(function_input_indices, function_indices, output_indices) -> tuple[tuple, np.ndarray]:
    """
    Sort component calls by function type.

    Returns
    -------
    groups : tuple of len(FUNCTIONS) int arrays, entry t has shape (K_t, I)
        Parameter indices of the inputs of every call of function type t.
    flat_outputs : (M,) int array
        For each output used by the original `update` (`output_matrix[output_indices]`), its index into the
        flattened grouped output vector of length K * OUTPUT_WIDTH.
    """
    function_indices = np.asarray(function_indices).astype(np.int64).reshape(-1)
    function_input_indices = np.asarray(function_input_indices)
    call_count = function_indices.shape[0]
    if call_count and (function_indices.min() < 0 or function_indices.max() >= len(FUNCTIONS)):
        raise ValueError("function index out of range")

    order = np.argsort(function_indices, kind="stable")
    position_of_call = np.empty(call_count, dtype=np.int64)
    position_of_call[order] = np.arange(call_count)

    counts = np.bincount(function_indices, minlength=len(FUNCTIONS))
    sorted_inputs = function_input_indices[order]
    groups = tuple(np.split(sorted_inputs, np.cumsum(counts)[:-1]))

    rows, columns = (x.astype(np.int64).reshape(-1) for x in np.asarray(output_indices))  # NumPy first: iterating a jax array compiles jit(_unstack)
    rows = np.mod(rows, max(call_count, 1))
    columns = np.mod(columns, OUTPUT_WIDTH)
    flat_outputs = position_of_call[rows] * OUTPUT_WIDTH + columns
    return groups, flat_outputs


def _has_duplicates(rows, columns, width) -> bool:
    flat = np.asarray(rows).astype(np.int64) * width + np.asarray(columns).astype(np.int64)
    return np.unique(flat).size != flat.size


def _signal_plan(signal_matrix, matrix_indices, flat_outputs, carrier_indices, placing_indices, carrier_size):
    """
    Per-position recipe for the augmented signal matrix (N, N+1), reproducing `simulate`'s dense steps:
    base → `.at[matrix_indices].set(outputs)` → scale `placing_indices` by −carrier → conjugate three blocks.
    """
    base = np.asarray(signal_matrix)
    size = base.shape[0]
    width = size + 1

    write_rows, write_columns = (x.astype(np.int64).reshape(-1) for x in np.asarray(matrix_indices))
    write_rows, write_columns = np.mod(write_rows, size), np.mod(write_columns, width)
    place_rows, place_columns = (x.astype(np.int64).reshape(-1) for x in np.asarray(placing_indices))
    place_rows, place_columns = np.mod(place_rows, size), np.mod(place_columns, width)
    carrier_indices = np.asarray(carrier_indices).astype(np.int64).reshape(-1)

    # With duplicate positions, the result of `.at[].set` depends on scatter order. Keep the original path then.
    if _has_duplicates(write_rows, write_columns, width) or _has_duplicates(place_rows, place_columns, width):
        return None
    if carrier_indices.shape[0] != place_rows.shape[0]:
        return None

    source = np.full((size, width), -1, dtype=np.int64)
    source[write_rows, write_columns] = flat_outputs
    scale = np.full((size, width), -1, dtype=np.int64)
    scale[place_rows, place_columns] = carrier_indices
    conjugate = np.zeros((size, width), dtype=bool)
    c = carrier_size
    conjugate[c : 2 * c, c : 2 * c] = True
    conjugate[2 * c :, :c] = True
    conjugate[c : 2 * c, 2 * c :] = True

    candidate = (base != 0) | (source >= 0) | (scale >= 0)
    rows, columns = np.nonzero(candidate)  # row-major order
    in_matrix = columns < size
    order = np.concatenate([np.flatnonzero(in_matrix), np.flatnonzero(~in_matrix)])
    rows, columns = rows[order], columns[order]

    return {
        "rows": rows[: in_matrix.sum()].astype(np.int32),
        "cols": columns[: in_matrix.sum()].astype(np.int32),
        "rhs_rows": rows[in_matrix.sum() :].astype(np.int32),
        "source": source[rows, columns].astype(np.int32),
        "base": base[rows, columns].astype(np.complex128),
        "scale": scale[rows, columns].astype(np.int32),
        "conjugate": conjugate[rows, columns],
    }


def assembly_arrays(simulation_arrays: dict, as_jax: bool = True) -> dict:
    """
    Build-time assembly plan for `simulate(..., assembly=...)`. NumPy only.
    `as_jax=False` returns the NumPy arrays (for the host plan) instead of JAX arrays.

    Returns {} if assembly is disabled or a precondition fails; `simulate` then uses the original code.
    """
    if not ENABLED:
        return {}
    a = simulation_arrays
    try:
        plan = {}
        for name, prefix in (("carrier", "carrier"), ("noise", "noise")):
            groups, flat_outputs = _group_calls(
                a[f"{prefix}_function_input_indices"], a[f"{prefix}_function_indices"], a[f"{prefix}_output_indices"]
            )
            plan[name] = {"groups": groups, "outputs": flat_outputs.astype(np.int32)}

        groups, flat_outputs = _group_calls(
            a["signal_function_input_indices"], a["signal_function_indices"], a["signal_output_indices"]
        )
        signal = _signal_plan(
            a["signal_matrix"],
            a["signal_matrix_indices"],
            flat_outputs,
            a["signal_carrier_indices"],
            a["signal_carrier_placing_indices"],
            carrier_size=np.asarray(a["carrier_matrix"]).shape[0],
        )
        if signal is None:
            return {}
        signal["groups"] = groups
    except ValueError:
        return {}
    plan["signal"] = signal
    return _to_jax(plan) if as_jax else plan


def _to_jax(tree):
    if isinstance(tree, dict):
        return {k: _to_jax(v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_jax(v) for v in tree)
    return jnp.asarray(tree)


# ---------------------------------------------------------------------------------------
# Run time
# ---------------------------------------------------------------------------------------

def component_outputs(parameters: jnp.ndarray, groups: tuple) -> jnp.ndarray:
    """
    Evaluate every component call of one parameter row, grouped by type.

    parameters : (P,) one parameter row
    groups     : tuple of len(FUNCTIONS) int arrays (K_t, I); empty types are skipped at trace time
    returns    : (K * OUTPUT_WIDTH,) complex, calls in grouped order
    """
    parts = [
        vmap(FUNCTIONS[function_type])(parameters[inputs])
        for function_type, inputs in enumerate(groups)
        if inputs.shape[0]
    ]
    if not parts:
        return jnp.zeros((0,), dtype=jnp.complex128)
    return jnp.concatenate(parts).reshape(-1)


def update(parameters, matrix, plan, matrix_indices):
    """Drop-in for `simulate.update` (one parameter row): same scatter, grouped evaluation."""
    outputs = component_outputs(parameters, plan["groups"])[plan["outputs"]]
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def signal_values_and_rhs(
    parameters: jnp.ndarray, carrier: jnp.ndarray, plan: dict, size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Signal system on its sparsity pattern.

    parameters : (R, P) expanded parameter rows (R = carrier values x signal values)
    carrier    : (CV, CN) carrier solution, broadcast against the rows exactly as in `simulate`
    plan       : `assembly_arrays(...)["signal"]`
    size       : N, the signal system size (static: `signal_matrix.shape[0]`)
    returns    : values (R, nnz) on (plan["rows"], plan["cols"]) and rhs (R, N)
    """
    outputs = vmap(component_outputs, in_axes=(0, None))(parameters, plan["groups"])  # (R, K*4)
    source = plan["source"]
    has_source = source >= 0
    entries = jnp.where(has_source, outputs[:, jnp.where(has_source, source, 0)], plan["base"])

    scale = plan["scale"]
    has_scale = scale >= 0
    entries = entries * jnp.where(has_scale, -carrier[:, jnp.where(has_scale, scale, 0)], 1.0)
    entries = jnp.where(plan["conjugate"], jnp.conjugate(entries), entries)

    nnz = plan["rows"].shape[0]
    rhs = jnp.zeros((entries.shape[0], size), dtype=entries.dtype)
    rhs = rhs.at[:, plan["rhs_rows"]].set(entries[:, nnz:])
    return entries[:, :nnz], rhs


def densify(values: jnp.ndarray, plan: dict, size: int) -> jnp.ndarray:
    """(R, nnz) pattern values → dense (R, N, N) system matrix, for the dense solvers."""
    dense = jnp.zeros((values.shape[0], size, size), dtype=values.dtype)
    return dense.at[:, plan["rows"], plan["cols"]].set(values)

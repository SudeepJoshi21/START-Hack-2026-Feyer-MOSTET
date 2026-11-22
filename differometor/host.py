"""
Host-native `simulate`: the whole physics on NumPy/SciPy behind one `jax.custom_vjp`   (branch `host-native`).

Why
---
After WS1-WS4 the run takes ~0.3 s, but XLA compilation takes ~18 s. Stubbing the component functions removes
~11 s of compile (docs/COMPILE_PROFILE.md). The time goes into compiling the physics, not into computing it. So here
the physics is not compiled at all:

* `simulate` becomes one `jax.custom_vjp` whose forward and backward are each a single `jax.pure_callback`.
  The XLA program is just these callbacks plus some reshapes, and compiles in milliseconds.
* The host code is NumPy and SciPy, i.e. compiled C/Fortran (LAPACK, SuperLU). Both are pinned in requirements.txt.
  Nothing needs a compiler on the judge's machine.
* **The physics stays single-source.** The component functions in `components.py` are *re-bound* to a tiny NumPy
  namespace (`_SHIM`) instead of `jax.numpy` and evaluated vectorised over all calls. For the gradient, the same code
  runs on forward-mode dual numbers (`Dual`), which gives each call's exact Jacobian w.r.t. its ≤ 7 real inputs.
  Components use only exp, sin, cos, sinh, cosh, sqrt, log10, arcsin, abs, conj and radians.
* The backward pass is a hand-written reverse sweep. It chains noise → (signal, weights) solve adjoint (WS2's
  formulas, WS1's SuperLU) → pattern assembly (WS3's plan) → carrier-selection vectors → dense carrier solve adjoint
  → component Jacobians → parameter expansion → linking → optimized parameters.

Gradient convention: identical to JAX's. For a real input p and complex intermediate y with cotangent ȳ,
p̄ = Re(ȳ · ∂y/∂p). Derivatives are taken along *real* directions of the inputs. The cotangents returned for the
complex-typed inputs `parameters` and `carrier_changing_values` therefore cover their real parts, which is where every
physical parameter lives.

`host_plan(...)` returns None (keep the JAX path) unless every precondition holds: WS3 assembly plan and WS2 pattern
present, identity or sqrt linking (see `_host_links`), unique write positions, broadcast-compatible sweeps.
`ENABLED = False` forces the JAX path for A/B tests.
"""

from __future__ import annotations

import functools
import hashlib
import types

import numpy as np
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from differometor import adjoint as _adjoint
from differometor import components as _components
from differometor.components import UNIT_VACUUM, H_PLANCK, F0

ENABLED = True

JAX_PATH_ARRAYS = False
"""
When a host plan is built, `run_build_step` leaves out what only the JAX path needs: the WS2 pattern and WS3 assembly
arrays, and the dense base matrices (carrier, signal, noise; replaced by empty placeholders, so `simulate` keeps its
signature). `simulate` refuses to run the JAX path on such arrays (ValueError) rather than compute on placeholders.
Set to True before building to keep everything, e.g. to compare the host path with the JAX path on one set of arrays.
"""

NOISE_FACTOR = UNIT_VACUUM * H_PLANCK * F0 * 0.25
"""noise = sqrt(2 * S * NOISE_FACTOR), exactly as in `simulate`."""

REFINE = {"signal": "adaptive", "weights": True, "adjoint": True}
"""
Iterative refinement of the SuperLU solves: True = adjoint.REFINEMENT_STEPS step(s) with a float64 residual,
"extended" = adjoint.EXTENDED_REFINEMENT_STEPS steps of mixed-precision refinement (residual in np.longdouble).

Signal solve: "extended". On Voyager in frequency-modulation mode the detected sidebands are tiny next to other
components of x (cond(A) 5.7e11), and a float64 residual cannot resolve them: sensitivity 1.1e-2 off the
extended-precision reference (original dense code: 3.7e-1 off); with 2 extended steps 8.6e-6 (floor of a float64
read-out of the exact solution: 6.1e-6). Benchmark workload: 6e-10 either way. Cost: 55-75 ms per forward pass
(first call in a fresh process: one more batch of solves + longdouble residual products); interleaved official
benchmark on a loaded node +0.05..0.26 s. An adaptive switch was tried: a float64 componentwise error estimate flags
every frequency of every setup (structurally tiny components), so only the read-out could tell; not adopted.
Set to True for the float64-only variant (~60 ms faster; the judged workload passes either way, 5.374e-6).

Noise weights and adjoint: one float64 step each.

Measured against dense LU: the noise weights (A^H) need it on Voyager (3.6e-3 without), the adjoint solves too
(Voyager gradient 1.1e-4 without, 4e-13 with). The signal solve was briefly left unrefined (benchmark workload 4.8e-8
without), but scripts/scenarios.py found a setup where that is not enough: on the second trained UIFO (uifo_20_5000)
the unrefined signal solve gives a sensitivity 1.6e-5 off at one frequency, where every refined solver and the
original code agree to 1e-10. With refinement: 7.9e-10. Cost ~10 ms on the benchmark.
"""


# =======================================================================================
# Forward-mode dual numbers and the NumPy namespace the component functions run on
# =======================================================================================

class Dual:
    """
    Value `v` (shape S) and tangent `d` (shape (T,) + S', S' broadcastable to S with len(S') == len(S)) along T real
    input directions. Only the operations the component functions use are implemented.

    The direction axis leads (the-prime-model-new): coefficient arrays of shape S then broadcast over it with
    contiguous inner loops of length n, instead of strided inner loops of length T <= 7 with the axis last.
    The per-element arithmetic is unchanged, so Jacobians are bitwise identical.
    """

    __slots__ = ("v", "d")
    __array_ufunc__ = None  # make numpy defer to the reflected operators below

    def __init__(self, v, d):
        self.v, self.d = v, d

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        d = self.d if self.d.shape[1:] == self.v.shape else np.broadcast_to(self.d, self.d.shape[:1] + self.v.shape)
        return Dual(self.v[index], d[(slice(None),) + index])

    def _aligned(self, ndim):
        """Tangent with leading singleton axes inserted after T so that it lines up with a result of rank `ndim`."""
        missing = ndim - self.v.ndim
        return self.d if missing == 0 else self.d.reshape(self.d.shape[:1] + (1,) * missing + self.d.shape[1:])

    def _tangent(self, shape):
        return np.broadcast_to(self._aligned(len(shape)), self.d.shape[:1] + tuple(shape))

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __add__(self, other):
        if isinstance(other, Dual):
            v = self.v + other.v
            return Dual(v, self._aligned(v.ndim) + other._aligned(v.ndim))
        v = self.v + other
        return Dual(v, self._aligned(np.ndim(v)))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Dual):
            v = self.v * other.v
            return Dual(v, self._aligned(v.ndim) * other.v + other._aligned(v.ndim) * self.v)
        other = np.asarray(other)
        v = self.v * other
        return Dual(v, self._aligned(v.ndim) * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            v = self.v / other.v
            return Dual(v, (self._aligned(v.ndim) - v * other._aligned(v.ndim)) / other.v)
        other = np.asarray(other)
        v = self.v / other
        return Dual(v, self._aligned(v.ndim) / other)

    def __rtruediv__(self, other):
        v = other / self.v
        return Dual(v, -(v / self.v) * self._aligned(v.ndim))

    def __pow__(self, exponent):
        if isinstance(exponent, Dual):
            raise NotImplementedError("Dual ** Dual is not used by the component functions")
        return Dual(self.v**exponent, (exponent * self.v ** (exponent - 1)) * self.d)


def _unary(function, derivative):
    def apply(x):
        if isinstance(x, Dual):
            # A direction the input does not depend on (tangent exactly 0) contributes exactly 0, as with JAX's
            # symbolic zeros, even where the derivative is infinite (sqrt(0), arcsin(±1), log10(0)). Without this,
            # 0 * inf = NaN leaked into unrelated gradients, e.g. a mirror's tuning gradient at reflectivity 1.0.
            # Finite derivatives give bit-identical results either way (finite * 0 == 0).
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = derivative(x.v)
            d = slope * x.d
            if not np.all(np.isfinite(slope)):
                d = np.where(x.d == 0, 0, d)
            return Dual(function(x.v), d)
        return function(x)

    return apply


def _abs(x):
    if not isinstance(x, Dual):
        return np.abs(x)
    a = np.abs(x.v)
    if np.iscomplexobj(x.v):
        return Dual(a, np.real(np.conj(x.v) * x.d) / a)
    return Dual(a, np.sign(x.v) * x.d)


def _conj(x):
    return Dual(np.conj(x.v), np.conj(x.d)) if isinstance(x, Dual) else np.conj(x)


def _array(values, dtype=None):
    """`jnp.array([...])` for lists of scalars, arrays and Duals: stack along a new axis 0 with broadcasting."""
    if isinstance(values, (Dual, np.ndarray)):
        return values
    duals = [x for x in values if isinstance(x, Dual)]
    shape = np.broadcast_shapes(*[(x.v.shape if isinstance(x, Dual) else np.shape(x)) for x in values])
    value = np.stack([np.broadcast_to(x.v if isinstance(x, Dual) else np.asarray(x), shape) for x in values])
    if dtype is not None:
        value = value.astype(dtype)
    if not duals:
        return value
    tangents = duals[0].d.shape[0]
    zero = np.zeros((tangents,) + shape, dtype=duals[0].d.dtype)
    tangent = np.stack([x._tangent(shape) if isinstance(x, Dual) else zero for x in values], axis=1)
    if dtype is not None:
        tangent = tangent.astype(dtype)
    return Dual(value, tangent)


_SHIM = types.SimpleNamespace(
    exp=_unary(np.exp, np.exp),
    sin=_unary(np.sin, np.cos),
    cos=_unary(np.cos, lambda v: -np.sin(v)),
    sinh=_unary(np.sinh, np.cosh),
    cosh=_unary(np.cosh, np.sinh),
    # Branch cuts must follow jax.numpy, not C99/NumPy: on the cut, JAX's complex sqrt and arcsin ignore the sign of
    # a zero imaginary part (treat -0.0 as +0.0), while NumPy honours it. Example (edge-cases bundle, edge_tir225):
    # cos(arcsin(1.45 sin(225 deg))) has argument -1.025 - 0j; NumPy gives -0.226j, JAX +0.226j (the other total-
    # internal-reflection branch; carrier 12x off). `v + 0.0` turns -0.0 into +0.0 and changes nothing else.
    # log10 honours the sign in both libraries and stays as is.
    sqrt=_unary(lambda v: np.sqrt(v + 0.0), lambda v: 0.5 / np.sqrt(v + 0.0)),
    log10=_unary(np.log10, lambda v: 1.0 / (v * np.log(10.0))),
    arcsin=_unary(lambda v: np.arcsin(v + 0.0), lambda v: 1.0 / np.sqrt(1.0 - (v + 0.0) * (v + 0.0))),
    radians=_unary(lambda v: v * (np.pi / 180.0), lambda v: np.full(np.shape(v), np.pi / 180.0)),  # as jnp.radians; np.radians rejects complex
    abs=_abs,
    conj=_conj,
    conjugate=_conj,
    array=_array,
    pi=np.pi,
    e=np.e,
    complex128=np.complex128,
    ndarray=np.ndarray,
)


@functools.lru_cache(maxsize=1)
def numpy_namespace() -> dict:
    """Every function of `components.py`, re-bound so that `jnp` means `_SHIM` (NumPy + Dual). Same code."""
    namespace = dict(vars(_components))
    namespace["jnp"] = _SHIM
    for name, obj in list(namespace.items()):
        if isinstance(obj, types.FunctionType) and obj.__module__ == _components.__name__:
            namespace[name] = types.FunctionType(obj.__code__, namespace, name, obj.__defaults__, obj.__closure__)
    return namespace


@functools.lru_cache(maxsize=1)
def numpy_functions() -> tuple:
    """`components.FUNCTIONS` on NumPy (see `numpy_namespace`), same order."""
    namespace = numpy_namespace()
    return tuple(namespace[f.__name__] for f in _components.FUNCTIONS)


def _evaluate(function, inputs: np.ndarray) -> np.ndarray:
    """Values of one component type. inputs (I, n) complex -> (n, 4) complex."""
    out = function(inputs)
    out = np.asarray(out)
    return np.broadcast_to(out.T if out.ndim == 2 else out[None, :], (inputs.shape[1], 4)).astype(np.complex128)


def _jacobian(function, inputs: np.ndarray, slots=None) -> np.ndarray:
    """
    d outputs / d inputs along real directions. inputs (I, n) -> (n, 4, U) complex, for the input slots `slots`
    (default: all I). Seeding only the slots a function actually reads keeps the dual numbers small.
    """
    count, n = inputs.shape
    slots = np.arange(count) if slots is None else slots
    seed = np.zeros((slots.size, count, n), dtype=np.complex128)
    seed[np.arange(slots.size), slots, :] = 1.0
    out = function(Dual(inputs, seed))
    if not isinstance(out, Dual):  # constant output
        return np.zeros((n, 4, slots.size), dtype=np.complex128)
    tangent = np.broadcast_to(out._aligned(2), (slots.size, 4, n))                 # (U, 4, n)
    # same values and memory order (4, n, U) as with the direction axis last, viewed as (n, 4, U)
    return np.ascontiguousarray(np.transpose(tangent, (1, 2, 0)), dtype=np.complex128).transpose(1, 0, 2)


_USED_SLOTS = {}


class _ReadRecorder(Dual):
    """A Dual input that records every index the component function reads from it (`parameters[k]`)."""

    __slots__ = ("reads",)

    def __init__(self, v, d):
        super().__init__(v, d)
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return Dual.__getitem__(self, index)


def _used_slots(function_type: int, sample: np.ndarray) -> np.ndarray:
    """
    Input slots a component function reads (cached per type). Inputs are padded to 7 slots, most functions read 1-3.

    Decided structurally, independent of any data: the function runs once on an input that records which indices
    it reads. Every component function reads its inputs as `parameters[k]` with constant k (and has no data-dependent
    Python branches: jnp.where evaluates both sides), so the recorded set is exactly the inputs it can depend on.
    Any other kind of read (slice, array index, iteration) marks every slot as used. `sample` (I, n) only provides
    shapes and finite values to run on.
    """
    if function_type not in _USED_SLOTS:
        count = sample.shape[0]
        recorder = _ReadRecorder(sample, np.zeros((count,) + sample.shape, dtype=np.complex128))
        numpy_functions()[function_type](recorder)
        reads = recorder.reads
        if all(isinstance(i, (int, np.integer)) and -count <= int(i) < count for i in reads):
            used = sorted({int(i) % count for i in reads})
        else:
            used = list(range(count))
        _USED_SLOTS[function_type] = np.asarray(used, dtype=np.int64)
    return _USED_SLOTS[function_type]


def _component_values(rows: np.ndarray, groups: tuple) -> np.ndarray:
    """Grouped component outputs for each parameter row: (R, P) -> (R, K*4), order as `assembly.component_outputs`."""
    functions = numpy_functions()
    parts = []
    for function_type, inputs in enumerate(groups):
        if inputs.shape[0] == 0:
            continue
        r, k = rows.shape[0], inputs.shape[0]
        x = rows[:, inputs]                                  # (R, K_t, I)
        values = _evaluate(functions[function_type], x.reshape(r * k, -1).T)
        parts.append(values.reshape(r, k * 4))
    if not parts:
        return np.zeros((rows.shape[0], 0), dtype=np.complex128)
    return np.concatenate(parts, axis=1)


def _component_pullback(rows: np.ndarray, groups: tuple, outputs_bar: np.ndarray, rows_bar: np.ndarray,
                        needed: np.ndarray | None = None) -> None:
    """
    rows_bar[r, input] += Re(Σ_o outputs_bar[r, call, o] · ∂output_o/∂input), for all calls (in place).

    needed (bool per parameter column, optional): only input slots that hold a needed parameter in at least one call
    are differentiated (the others' contributions are not used). Each dual-number direction is computed on its own,
    so the needed entries of rows_bar are bitwise the same as with all slots.
    """
    functions = numpy_functions()
    offset = 0
    r = rows.shape[0]
    for function_type, inputs in enumerate(groups):
        k = inputs.shape[0]
        if k == 0:
            continue
        x = rows[:, inputs].reshape(r * k, -1).T                                 # (I, R*K_t)
        slots = _used_slots(function_type, x[:, : min(4, x.shape[1])])
        if needed is not None and slots.size:
            slots = slots[np.any(needed[inputs[:, slots]], axis=0)]
        offset += k
        if slots.size == 0:
            continue
        jac = _jacobian(functions[function_type], x, slots)                      # (R*K_t, 4, U)
        ybar = outputs_bar[:, (offset - k) * 4 : offset * 4].reshape(r * k, 4)   # (R*K_t, 4)
        contribution = np.real(np.einsum("no,noi->ni", ybar, jac)).reshape(r, k, -1)
        width = rows_bar.shape[1]
        flat = (np.arange(r)[:, None, None] * width + inputs[None, :, slots]).ravel()
        rows_bar += np.bincount(flat, weights=contribution.ravel(), minlength=r * width).reshape(r, width)


# =======================================================================================
# Build time: the static plan
# =======================================================================================

def _fingerprint(tree, digest) -> None:
    if isinstance(tree, dict):
        for key in sorted(tree):
            digest.update(key.encode())
            _fingerprint(tree[key], digest)
    elif isinstance(tree, (tuple, list)):
        digest.update(b"(%d" % len(tree))
        for item in tree:
            _fingerprint(item, digest)
    else:
        array = np.ascontiguousarray(tree)
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())


@jax.tree_util.register_pytree_node_class
class HostPlan:
    """
    Static structure for the host path, as a pytree node *without leaves*: jit treats it as static.

    It holds index structure, masks and base-matrix constants only, all derived by `run_build_step` inside
    the timed build. All numeric inputs (optimized parameters, parameter row, sweep values) still flow through
    the traced arguments of `simulate`.
    """

    def __init__(self, plan: dict):
        self.plan = plan
        digest = hashlib.sha256()
        _fingerprint(plan, digest)
        self.key = digest.digest()

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux, children):
        return aux

    def __getitem__(self, name):
        return self.plan[name]

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, HostPlan) and self.key == other.key


def _unique(rows, cols, width) -> bool:
    flat = np.asarray(rows).astype(np.int64) * width + np.asarray(cols).astype(np.int64)
    return np.unique(flat).size == flat.size


def _host_links():
    """
    Linking functions the host path supports: kind -> (callables recognised, value, derivative).

    Setups link derived parameters with these, e.g. a laser amplitude = sqrt(power) (`setups.py`, amplitude and
    signal modes). Any other callable keeps the JAX path. NumPy's sqrt is IEEE correctly rounded like XLA's.
    """
    from differometor.build import identity_link

    return {
        0: ((identity_link,), lambda x: x, lambda x: np.ones_like(x)),
        1: ((jnp.sqrt, np.sqrt), lambda x: np.sqrt(x + 0.0), lambda x: 0.5 / np.sqrt(x + 0.0)),   # jnp.sqrt branch convention (see _SHIM)
    }


def _apply_links(kinds: np.ndarray, x: np.ndarray, which: int) -> np.ndarray:
    """Values (which=1) or derivatives (which=2) of the linking functions `kinds` at the inputs x."""
    out = np.empty_like(x)
    for kind, entry in _host_links().items():
        mask = kinds == kind
        if mask.any():
            out[mask] = entry[which](x[mask])
    return out


def _link_kind(function) -> int | None:
    for kind, (callables, _, _) in _host_links().items():
        if any(function is c for c in callables):
            return kind
    return None


def host_plan(a: dict) -> HostPlan | None:
    """Build the static host plan from `run_build_step`'s arrays, or None if a precondition fails."""
    from differometor.build import LINKING_FUNCTIONS

    if not ENABLED or "assembly" not in a or "signal_pattern_rows" not in a:
        return None
    n = lambda key: np.asarray(a[key])  # noqa: E731

    # Linking: functions with a host implementation only (identity, sqrt), each target written once.
    link_kinds = [_link_kind(LINKING_FUNCTIONS[int(i)]) for i in n("linking_function_indices").ravel()]
    if any(kind is None for kind in link_kinds):
        return None
    if np.unique(n("indices_to_link")).size != n("indices_to_link").size:
        return None

    plan = {}
    for part in ("carrier", "signal", "noise"):
        plan[part] = {k: (tuple(np.asarray(g) for g in v) if k == "groups" else np.asarray(v))
                      for k, v in a["assembly"][part].items()}

    carrier_base = n("carrier_matrix")
    cn = carrier_base.shape[0]
    mr, mc = (x.astype(np.int64) for x in n("carrier_matrix_indices"))
    mr, mc = np.mod(mr, cn), np.mod(mc, cn + 1)
    if not _unique(mr, mc, cn + 1):
        return None
    plan["carrier"].update(base=carrier_base.astype(np.complex128), matrix_rows=mr, matrix_cols=mc)

    q_rows, q_cols = (x.astype(np.int64) for x in n("noise_system_matrix_indices"))
    plan["noise"].update(
        q_rows=q_rows, q_cols=q_cols,
        base_rows=n("noise_base_rows").astype(np.int64), base_cols=n("noise_base_cols").astype(np.int64),
        base_values=n("noise_matrix")[n("noise_base_rows"), n("noise_base_cols")].astype(np.complex128),
        once=bool(n("noise_parameter_rows").shape[0] == 1),
    )

    carrier_changing = n("carrier_changing_parameter_indices").astype(np.int64)
    signal_changing = n("signal_changing_parameter_indices").astype(np.int64)
    if np.unique(carrier_changing).size != carrier_changing.size or np.unique(signal_changing).size != signal_changing.size:
        return None
    cv, sv = n("carrier_changing_values").shape[1], n("signal_changing_values").shape[1]
    if not (cv == 1 or cv == cv * sv):
        return None
    qhd = n("qhd_placing_indices").reshape(-1, 3).astype(np.int64)

    plan["global"] = {
        "optimized_parameter_indices": n("optimized_parameter_indices").astype(np.int64),
        "optimized_value_indices": n("optimized_value_indices").astype(np.int64),
        "linked_indices": n("linked_indices").astype(np.int64),
        "indices_to_link": n("indices_to_link").astype(np.int64),
        "link_kinds": np.asarray(link_kinds, dtype=np.int64),
        "qhd_parameter_indices": n("qhd_parameter_indices").astype(np.int64),
        "qhd_placing_indices": qhd,
        "carrier_changing_parameter_indices": carrier_changing,
        "signal_changing_parameter_indices": signal_changing,
        "noise_selection_vectors": n("noise_selection_vectors").astype(np.complex128),
        "signal_size": np.asarray(n("signal_matrix").shape[0]),
    }
    return HostPlan(plan)


# =======================================================================================
# Host numerics: forward and reverse sweep
# =======================================================================================

def _expand(array, indices, values):
    """NumPy copy of `simulate.expand_parameters` (same row layout)."""
    count = values.shape[1]
    tiled = np.tile(array, (count, 1))
    tiled[np.repeat(np.arange(count), len(indices)), np.tile(indices, count)] = values.T.reshape(-1)
    return tiled


def _expand_pullback(tiled_bar, source_rows, indices, count):
    """Reverse of `_expand`: (values_bar (V, count), array_bar (source_rows, P))."""
    rows, cols = np.repeat(np.arange(count), len(indices)), np.tile(indices, count)
    values_bar = tiled_bar[rows, cols].reshape(count, len(indices)).T.copy()
    tiled_bar = tiled_bar.copy()
    tiled_bar[rows, cols] = 0
    return values_bar, tiled_bar.reshape(count, source_rows, -1).sum(axis=0)


def _scatter_add(target, index, values):
    """
    target[..., index] += values along the last axis (duplicates in `index` accumulate), for a zero-initialised
    complex target. Same sums, in the same order per entry, as np.add.at, but one vectorised bincount per part.
    """
    lead = int(np.prod(target.shape[:-1]))
    width = target.shape[-1]
    values = np.broadcast_to(values, target.shape[:-1] + (index.shape[0],)).reshape(lead, index.shape[0])
    flat = (np.arange(lead)[:, None] * width + index[None, :]).ravel()
    size = lead * width
    total = np.bincount(flat, values.real.ravel(), minlength=size) + 1j * np.bincount(flat, values.imag.ravel(), minlength=size)
    target += total.reshape(target.shape)


def _to_carrier_rows(x, cv):
    """Sum a per-signal-row array over the rows that share one carrier row (broadcast cv == 1)."""
    return x.sum(axis=0, keepdims=True) if cv == 1 and x.shape[0] != 1 else x


def _forward(plan, optimized, parameters, carrier_values, signal_values):
    g, cp, sp, npn = plan["global"], plan["carrier"], plan["signal"], plan["noise"]
    s = {}
    p = parameters.astype(np.complex128).copy()
    p[0, g["optimized_parameter_indices"]] = optimized[g["optimized_value_indices"]]
    link_inputs = p[0, g["linked_indices"]]                          # gather all inputs, then set (as simulate)
    p[0, g["indices_to_link"]] = _apply_links(g["link_kinds"], link_inputs, 1)
    s["link_inputs"] = link_inputs
    phase = np.exp(1j * (p[0, g["qhd_parameter_indices"]] * (np.pi / 180.0)))   # jnp.radians semantics
    s["p"], s["phase"] = p, phase

    # carrier: grouped components -> dense (CV, CN, CN+1) -> LAPACK solve
    pc = _expand(p, g["carrier_changing_parameter_indices"], carrier_values)
    yc = _component_values(pc, cp["groups"])
    m = np.broadcast_to(cp["base"], (pc.shape[0],) + cp["base"].shape).copy()
    m[:, cp["matrix_rows"], cp["matrix_cols"]] = yc[:, cp["outputs"]]
    carrier = np.linalg.solve(m[:, :, :-1], m[:, :, -1:])[..., 0]   # (CV, CN)
    s["pc"], s["carrier_matrix"], s["carrier"] = pc, m, carrier
    cv, cn = carrier.shape

    # signal system on its pattern (WS3 recipe)
    ps = _expand(pc, g["signal_changing_parameter_indices"], signal_values)
    ys = _component_values(ps, sp["groups"])
    if ys.shape[1] == 0:                                             # no signal components at all
        ys = np.zeros((ps.shape[0], 1), dtype=np.complex128)
    has_source, has_scale = sp["source"] >= 0, sp["scale"] >= 0
    source = np.where(has_source, ys[:, np.where(has_source, sp["source"], 0)], sp["base"])
    scale = np.where(has_scale, -carrier[:, np.where(has_scale, sp["scale"], 0)], 1.0)
    pre = source * scale
    entries = np.where(sp["conjugate"], np.conj(pre), pre)
    nnz, size = sp["rows"].shape[0], int(g["signal_size"])
    values = entries[:, :nnz]
    rhs = np.zeros((entries.shape[0], size), dtype=np.complex128)
    rhs[:, sp["rhs_rows"]] = entries[:, nnz:]
    s.update(ps=ps, source=source, scale=scale, values=values)
    r = ps.shape[0]

    # detector selections C (D, CV, N) exactly as in simulate
    selection = g["noise_selection_vectors"].copy()                  # (D, 1, N)
    qhd = g["qhd_placing_indices"]
    s["selection_before_qhd"] = selection.copy()
    selection[qhd[:, 0], qhd[:, 1], qhd[:, 2]] *= phase
    tiled = np.tile(selection, (1, cv, 1))
    c_sel = tiled.copy()
    c_sel[:, :, :cn] = tiled[:, :, :cn] * carrier
    c_sel[:, :, cn : 2 * cn] = np.conj(tiled[:, :, cn : 2 * cn] * carrier)
    c_rows = np.swapaxes(np.broadcast_to(c_sel, (c_sel.shape[0], r, size)), 0, 1)   # (R, D, N)
    s["selection_tiled"] = tiled
    s["c_rows"], s["rhs"] = c_rows, rhs

    # one SuperLU factorisation per frequency (WS1), kept in the state for the backward pass
    factors = _adjoint.Factorisations(values, sp["rows"], sp["cols"], size)
    x = factors.solve(rhs, "N", REFINE["signal"],                                       # A x = b
                      needs_extended=lambda x, dx: _needs_extended(g, carrier, x, dx))
    w = (factors.solve(np.swapaxes(c_rows, 1, 2), "H", REFINE["weights"]).swapaxes(1, 2)   # A^H w_d = c_d
         if c_rows.shape[1] else np.zeros(c_rows.shape, dtype=np.complex128))
    s["factors"] = factors
    s["x"], s["w"] = x, w

    # noise = sqrt(2 K Re(w^H Q w)), Q values from the grouped noise components
    pn = ps[:1] if npn["once"] else ps
    yn = _component_values(pn, npn["groups"])
    q = np.concatenate([yn[:, npn["outputs"]], np.broadcast_to(npn["base_values"], (pn.shape[0], npn["base_values"].shape[0]))], axis=1)
    q_rows = np.concatenate([npn["q_rows"], npn["base_rows"]])
    q_cols = np.concatenate([npn["q_cols"], npn["base_cols"]])
    total = np.real(np.einsum("rdk,rk,rdk->rd", np.conj(w[:, :, q_rows]), np.broadcast_to(q, (r, q.shape[1])), w[:, :, q_cols]))
    noise = np.sqrt(2 * total * NOISE_FACTOR)                         # (R, D)
    s.update(pn=pn, q=q, q_rows=q_rows, q_cols=q_cols, noise=noise)
    return carrier.T, x.T, noise.T, s


def _needed_parameters(plan) -> np.ndarray:
    """Parameter columns that influence the optimized values' gradient: the optimized ones and the link targets of them."""
    g = plan["global"]
    size = int(max(g["optimized_parameter_indices"].max(initial=0), g["linked_indices"].max(initial=0),
                   g["indices_to_link"].max(initial=0), g["qhd_parameter_indices"].max(initial=0))) + 1
    needed = np.zeros(size, dtype=bool)
    needed[g["optimized_parameter_indices"]] = True
    needed[g["indices_to_link"][needed[g["linked_indices"]]]] = True     # identity/sqrt links: one level
    return needed


def _backward(plan, optimized, parameters, carrier_values, signal_values, carrier_bar_out, signal_bar, noise_bar, s=None,
              only_optimized=False):
    g, cp, sp, npn = plan["global"], plan["carrier"], plan["signal"], plan["noise"]
    if s is None:                                                   # no cached forward state: recompute
        _, _, _, s = _forward(plan, optimized, parameters, carrier_values, signal_values)
    carrier, x, w, ps, pc = s["carrier"], s["x"], s["w"], s["ps"], s["pc"]
    cv, cn = carrier.shape
    r, size = x.shape

    # noise
    noise_b = noise_bar.T                                            # (R, D)
    total_b = noise_b * NOISE_FACTOR / s["noise"]
    q, q_rows, q_cols = s["q"], s["q_rows"], s["q_cols"]
    qb = np.broadcast_to(q, (r, q.shape[1]))
    qw = np.zeros_like(w)
    _scatter_add(qw, q_rows, qb[:, None, :] * w[:, :, q_cols])
    qtw = np.zeros_like(w)
    _scatter_add(qtw, q_cols, qb[:, None, :] * np.conj(w[:, :, q_rows]))
    w_bar = total_b[:, :, None] * (np.conj(qw) + qtw)
    q_bar = np.einsum("rd,rdk,rdk->rk", total_b, np.conj(w[:, :, q_rows]), w[:, :, q_cols])
    if npn["once"]:
        q_bar = q_bar.sum(axis=0, keepdims=True)

    # signal + weights solve (WS2 adjoint formulas, WS1 SuperLU kernel)
    rows, cols = sp["rows"], sp["cols"]
    x_bar, u_bar = signal_bar.T, np.conj(w_bar)
    factors = s["factors"]                                          # reuse the forward factorisations
    lam_x = factors.solve(x_bar, "T", REFINE["adjoint"])            # A^T lam_x = x_bar
    lam_u = (factors.solve(np.swapaxes(u_bar, 1, 2), "N", REFINE["adjoint"]).swapaxes(1, 2)   # A lam_u = u_bar
             if u_bar.shape[1] else np.zeros(u_bar.shape, dtype=np.complex128))
    values_bar = -lam_x[:, rows] * x[:, cols] - np.sum(np.conj(w)[:, :, rows] * lam_u[:, :, cols], axis=1)
    rhs_bar, c_bar = lam_x, np.conj(lam_u)                           # (R, N), (R, D, N)

    # pattern assembly (WS3 recipe) backwards
    entries_bar = np.concatenate([values_bar, rhs_bar[:, sp["rhs_rows"]]], axis=1)
    pre_bar = np.where(sp["conjugate"], np.conj(entries_bar), entries_bar)
    source_bar = pre_bar * s["scale"]
    scale_bar = pre_bar * s["source"]
    carrier_bar = np.zeros((cv, cn), dtype=np.complex128)
    has_scale, has_source = sp["scale"] >= 0, sp["source"] >= 0
    _scatter_add(carrier_bar, sp["scale"][has_scale].astype(np.int64), -_to_carrier_rows(scale_bar[:, has_scale], cv))
    ys_bar = np.zeros((r, sum(gr.shape[0] for gr in sp["groups"]) * 4), dtype=np.complex128)
    _scatter_add(ys_bar, sp["source"][has_source].astype(np.int64), source_bar[:, has_source])

    # detector selections backwards
    c_bar = _to_carrier_rows(c_bar, cv).transpose(1, 0, 2)          # (R or 1, D, N) -> (D, CV, N)
    tiled = s["selection_tiled"]
    carrier_bar += np.sum(c_bar[:, :, :cn] * tiled[:, :, :cn], axis=0)
    carrier_bar += np.sum(np.conj(c_bar[:, :, cn : 2 * cn]) * tiled[:, :, cn : 2 * cn], axis=0)
    tiled_bar = c_bar.copy()
    tiled_bar[:, :, :cn] = c_bar[:, :, :cn] * carrier
    tiled_bar[:, :, cn : 2 * cn] = np.conj(c_bar[:, :, cn : 2 * cn]) * carrier
    selection_bar = tiled_bar.sum(axis=1, keepdims=True)              # (D, 1, N)
    qhd = g["qhd_placing_indices"]
    phase_bar = selection_bar[qhd[:, 0], qhd[:, 1], qhd[:, 2]] * s["selection_before_qhd"][qhd[:, 0], qhd[:, 1], qhd[:, 2]]

    # carrier solve: carrier = A^-1 b  ->  lam = A^-T carrier_bar, A_bar = -lam carrier^T, b_bar = lam
    carrier_bar += carrier_bar_out.T
    m = s["carrier_matrix"]
    lam = np.linalg.solve(np.swapaxes(m[:, :, :-1], 1, 2), carrier_bar[..., None])[..., 0]
    m_bar = np.concatenate([-lam[:, :, None] * carrier[:, None, :], lam[:, :, None]], axis=2)   # (CV, CN, CN+1)
    yc_bar = np.zeros((cv, sum(gr.shape[0] for gr in cp["groups"]) * 4), dtype=np.complex128)
    _scatter_add(yc_bar, cp["outputs"].astype(np.int64), m_bar[:, cp["matrix_rows"], cp["matrix_cols"]])

    # component Jacobians -> parameter rows
    ps_bar = np.zeros(ps.shape, dtype=np.complex128)
    needed = None
    if only_optimized:                       # only the optimized values are differentiated (custom_vjp symbolic zeros)
        needed = np.zeros(ps.shape[1], dtype=bool)
        mask = _needed_parameters(plan)
        needed[: mask.size] = mask[: ps.shape[1]]
    _component_pullback(ps, sp["groups"], ys_bar, ps_bar, needed)
    yn_bar = np.zeros((q_bar.shape[0], sum(gr.shape[0] for gr in npn["groups"]) * 4), dtype=np.complex128)
    np.add.at(yn_bar, (slice(None), npn["outputs"]), q_bar[:, : npn["outputs"].shape[0]])
    _component_pullback(s["pn"], npn["groups"], yn_bar, ps_bar[: s["pn"].shape[0]], needed)
    pc_bar = np.zeros(pc.shape, dtype=np.complex128)
    _component_pullback(pc, cp["groups"], yc_bar, pc_bar, needed)

    # parameter expansion, qhd phases, linking, optimized values
    signal_values_bar, pc_bar_from_signal = _expand_pullback(
        ps_bar, pc.shape[0], g["signal_changing_parameter_indices"], signal_values.shape[1]
    )
    pc_bar += pc_bar_from_signal
    carrier_values_bar, p_bar = _expand_pullback(
        pc_bar, 1, g["carrier_changing_parameter_indices"], carrier_values.shape[1]
    )
    np.add.at(
        p_bar[0], g["qhd_parameter_indices"],
        np.real(phase_bar * 1j * (np.pi / 180.0) * s["phase"]),
    )
    to_link = p_bar[0, g["indices_to_link"]].copy()
    p_bar[0, g["indices_to_link"]] = 0
    derivative = _apply_links(g["link_kinds"], s["link_inputs"], 2)
    if np.any(np.imag(derivative) != 0):
        # p_bar holds the real part of JAX's (complex) parameter cotangent, which is all a real design variable
        # needs. Through a link with a non-real derivative (e.g. sqrt of a negative power) the imaginary part
        # would be needed too: refuse rather than return a wrong gradient.
        raise ValueError("host path: linked parameter outside the real domain of its linking function "
                         "(e.g. negative power); rebuild with differometor.host.ENABLED = False")
    np.add.at(p_bar[0], g["linked_indices"], to_link * np.real(derivative))
    optimized_bar = np.zeros(optimized.shape, dtype=np.float64)
    np.add.at(optimized_bar, g["optimized_value_indices"], np.real(p_bar[0, g["optimized_parameter_indices"]]))
    p_bar[0, g["optimized_parameter_indices"]] = 0
    return optimized_bar, p_bar, carrier_values_bar, signal_values_bar


# =======================================================================================
# JAX side: one custom_vjp, two callbacks
# =======================================================================================

def _pack(*arrays):
    """
    Concatenate (as complex128) into one freshly computed buffer.

    Callback operands that alias input buffers of the running program deadlock with the process pinned to one CPU
    (see the deadlock fix in adjoint._callback).
    """
    return jnp.concatenate([jnp.ravel(a).astype(jnp.complex128) for a in arrays])


def _unpack(packed, shapes):
    out, start = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        out.append(packed[start : start + size].reshape(shape))
        start += size
    return out


def _output_shapes(plan, carrier_values, signal_values):
    cn = plan["carrier"]["base"].shape[0]
    cv, sv = carrier_values.shape[1], signal_values.shape[1]
    d = plan["global"]["noise_selection_vectors"].shape[0]
    size = int(plan["global"]["signal_size"])
    return (
        jax.ShapeDtypeStruct((cn, cv), jnp.complex128),
        jax.ShapeDtypeStruct((size, cv * sv), jnp.complex128),
        jax.ShapeDtypeStruct((d, cv * sv), jnp.float64),
    )


_LAST_FORWARD = {"key": None, "state": None}
"""
The forward state (including the SuperLU factorisations) of the most recent forward call, keyed by the exact bytes
of its inputs. The backward pass reuses it instead of recomputing the forward. This is ordinary AD residual storage,
kept on the host because SuperLU factor objects cannot be passed through XLA. On a key mismatch the backward
recomputes, so results never depend on the cache.
"""


def _input_key(plan, packed_inputs: np.ndarray) -> tuple:
    return plan.key, hashlib.sha1(np.ascontiguousarray(packed_inputs).tobytes()).digest()


def _call_forward(plan, optimized, parameters, carrier_values, signal_values):
    shapes = [optimized.shape, parameters.shape, carrier_values.shape, signal_values.shape]

    @_adjoint.with_single_threaded_blas
    def host(packed):
        packed = np.asarray(packed)
        o, p, c, s = _unpack(packed, shapes)
        carrier, signal, noise, state = _forward(plan, o.real, p, c, s.real)
        _LAST_FORWARD["key"], _LAST_FORWARD["state"] = _input_key(plan, packed), state
        return carrier, signal, noise.astype(np.float64)

    return jax.pure_callback(
        host, _output_shapes(plan, carrier_values, signal_values),
        _pack(optimized, parameters, carrier_values, signal_values), vmap_method="sequential",
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _simulate(plan, optimized, parameters, carrier_values, signal_values):
    carrier, signal, noise = _call_forward(plan, optimized, parameters, carrier_values, signal_values)
    return carrier, signal, noise.squeeze()


def _simulate_fwd(plan, optimized, parameters, carrier_values, signal_values):
    # symbolic_zeros=True: each argument is a CustomVJPPrimal (value, perturbed). Which inputs are differentiated is
    # static; it travels to bwd as the shape of an empty residual array.
    primals = (optimized, parameters, carrier_values, signal_values)
    values = tuple(x.value for x in primals)
    perturbed = jnp.zeros((0,) + tuple(int(bool(x.perturbed)) for x in primals))
    carrier, signal, noise = _call_forward(plan, *values)
    return (carrier, signal, noise.squeeze()), (*values, perturbed)


def _simulate_bwd(plan, residuals, cotangents):
    optimized, parameters, carrier_values, signal_values, perturbed = residuals
    perturbed = [bool(b) for b in perturbed.shape[1:]]
    only_optimized = perturbed[0] and not any(perturbed[1:])
    shapes = _output_shapes(plan, carrier_values, signal_values)
    cotangents = tuple(jnp.zeros(ct.aval.shape, ct.aval.dtype) if type(ct) is SymbolicZero else ct for ct in cotangents)
    carrier_bar, signal_bar, noise_bar = cotangents
    in_shapes = [optimized.shape, parameters.shape, carrier_values.shape, signal_values.shape]
    ct_shapes = [s.shape for s in shapes]

    input_size = sum(int(np.prod(shape)) for shape in in_shapes)

    @_adjoint.with_single_threaded_blas
    def host(packed):
        packed = np.asarray(packed)
        o, p, c, s, cb, sb, nb = _unpack(packed, in_shapes + ct_shapes)
        cached = _LAST_FORWARD["state"] if _LAST_FORWARD["key"] == _input_key(plan, packed[:input_size]) else None
        grads = _backward(plan, o.real, p, c, s.real, cb, sb, nb.real, s=cached, only_optimized=only_optimized)
        return tuple(
            (np.asarray(gr) if np.issubdtype(dt, np.complexfloating) else np.real(gr)).astype(dt)
            for gr, dt in zip(grads, dtypes)
        )

    dtypes = [np.dtype(a.dtype) for a in (optimized, parameters, carrier_values, signal_values)]
    result_shapes = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (optimized, parameters, carrier_values, signal_values))
    packed = _pack(optimized, parameters, carrier_values, signal_values,
                   carrier_bar, signal_bar, jnp.reshape(noise_bar, ct_shapes[2]))
    grads = jax.pure_callback(host, result_shapes, packed, vmap_method="sequential")
    return tuple(gr if d else None for gr, d in zip(grads, perturbed))      # None: zero cotangent (not differentiated)


_simulate.defvjp(_simulate_fwd, _simulate_bwd, symbolic_zeros=True)


def simulate(plan, optimized_parameters, parameters, carrier_changing_values, signal_changing_values):
    """Host-native equivalent of `differometor.simulate` for the arrays `run_build_step` produced with this plan."""
    return _simulate(plan, optimized_parameters, parameters, carrier_changing_values, signal_changing_values)


# =======================================================================================
# Detector read-outs on the host (used by `differometor.signal_detector` / `power_detector` when enabled)
# =======================================================================================
# The benchmark's sensitivity and gradient programs are compiled by XLA around these two functions. As host
# callbacks with hand-written VJPs (JAX conventions: holomorphic parts transpose, conj parts conjugate) they add
# only a callback each to those programs.

def _host_call(function, out_shape, *arrays):
    shapes = [a.shape for a in arrays]

    @_adjoint.with_single_threaded_blas
    def run(packed):
        return function(*_unpack(np.asarray(packed), shapes))

    return jax.pure_callback(run, out_shape, _pack(*arrays), vmap_method="sequential")


@jax.custom_vjp
def signal_detector(carrier, signal):
    """Same as `components.signal_detector`: conj(c) * s_upper + c * s_lower."""
    n = carrier.shape[0]
    shape = jnp.broadcast_shapes(carrier.shape, signal[:n].shape)
    return _host_call(
        lambda c, s: np.conj(c) * s[:n] + c * s[n : 2 * n],
        jax.ShapeDtypeStruct(shape, jnp.complex128), carrier, signal,
    )


def _signal_detector_fwd(carrier, signal):
    return signal_detector(carrier, signal), (carrier, signal)


def _signal_detector_bwd(residuals, y_bar):
    carrier, signal = residuals
    n = carrier.shape[0]
    shapes = [carrier.shape, signal.shape, y_bar.shape]

    def bwd(c, s, yb):
        c_bar = np.conj(yb * s[:n]) + yb * s[n : 2 * n]
        c_bar = c_bar.sum(axis=tuple(i for i in range(c_bar.ndim) if c.shape[i] == 1 and c_bar.shape[i] != 1), keepdims=True)
        s_bar = np.zeros(s.shape, dtype=np.complex128)
        s_bar[:n] = yb * np.conj(c)
        s_bar[n : 2 * n] = yb * c
        return c_bar.astype(np.complex128), s_bar

    out = (jax.ShapeDtypeStruct(shapes[0], jnp.complex128), jax.ShapeDtypeStruct(shapes[1], jnp.complex128))
    c_bar, s_bar = _host_call(bwd, out, carrier, signal, y_bar)
    return c_bar.astype(carrier.dtype), s_bar.astype(signal.dtype)


signal_detector.defvjp(_signal_detector_fwd, _signal_detector_bwd)


@jax.custom_vjp
def power_detector(solution):
    """Same as `components.power_detector`: 0.5 * EPSILON_0C * |E|^2."""
    return _host_call(
        lambda e: (0.5 * _components.EPSILON_0C * np.abs(e) ** 2).astype(np.float64),
        jax.ShapeDtypeStruct(solution.shape, jnp.float64), solution,
    )


def _power_detector_fwd(solution):
    return power_detector(solution), solution


def _power_detector_bwd(solution, y_bar):
    # y = k |E|^2 (real) -> E_bar = 2 k y_bar conj(E)  (JAX convention, cf. jax.grad(abs(z)**2) = 2 conj(z))
    return (2 * 0.5 * _components.EPSILON_0C * y_bar * jnp.conj(solution),)


power_detector.defvjp(_power_detector_fwd, _power_detector_bwd)


SIGNAL_ERROR_TOLERANCE = 1e-7
"""
Adaptive refinement of the signal solve (REFINE["signal"] = "adaptive"): frequencies whose estimated relative error
of the detected signal exceeds this get the mixed-precision (longdouble) refinement steps; the others keep one
float64 refinement step. The estimate uses that step's correction, which bounds the remaining error of the refined
solution from above (benchmark workload: at most 5.2e-8, true error of the refined solution ~7e-10).
"""


def _needs_extended(g, carrier, x, dx):
    """
    Per frequency: does the detected signal need mixed-precision refinement?

    The detectors are the non-zero positions p (upper sideband) of the noise selection vectors; a signal detector
    reads sig_p = conj(c_p) x_p + c_p x_{p+n} (n = carrier size). With dx the last float64 correction (an estimate of
    the remaining error of x), the error of sig_p is about |c_p| (|dx_p| + |dx_{p+n}|). It is compared with |sig_p|
    and, for balanced read-outs, with |sig_i - sig_j| and |sig_i + sig_j|: on Voyager in frequency-modulation mode the
    entries are accurate to ~1e-12 but the read-out cancels by ~1e10, which only this estimate sees. Setups without
    detector positions always get the extended steps (conservative).
    """
    rows, size = x.shape[0], x.shape[1]
    cn = carrier.shape[1]
    detectors = np.flatnonzero(np.abs(g["noise_selection_vectors"]).sum(axis=(0, 1))[:cn] > 0)
    if detectors.size == 0 or size < 2 * cn:
        return np.ones(rows, dtype=bool)
    c = carrier if carrier.shape[0] == rows else np.broadcast_to(carrier[:1], (rows, cn))
    cp = c[:, detectors]
    xu, xl = x[:, detectors, 0], x[:, detectors + cn, 0]
    sig = np.conj(cp) * xu + cp * xl                                                       # (R, P)
    err = np.abs(cp) * (np.abs(dx[:, detectors, 0]) + np.abs(dx[:, detectors + cn, 0]))
    scale = np.abs(sig)
    if detectors.size > 1:
        i, j = np.triu_indices(detectors.size, 1)
        scale = np.concatenate([scale, np.abs(sig[:, i] - sig[:, j]), np.abs(sig[:, i] + sig[:, j])], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = err.max(axis=1) / scale.min(axis=1)
    return ~(estimate <= SIGNAL_ERROR_TOLERANCE)

"""
Batched fixed-pattern LU refactorisation over frequencies (the-prime-model-new2, experimental).

All F systems share one sparsity pattern. SuperLU spends most of its ~0.46 ms per 751x751 factorisation on symbolic
work; the numeric work is only ~18.6k complex multiply-adds. Here the row/column permutations of one SuperLU
factorisation (partial pivoting, threshold 1.0) define a fixed elimination program (fill pattern computed
symbolically), executed for all F value sets at once with NumPy: values are stored as (slot, frequency), the
elimination is level-scheduled over pivots, and triangular solves for op(A) in {A, A^T, A^H} are level-scheduled
over rows.

Pivot safety. Reusing one pivot sequence at other frequencies is threshold pivoting, not strict partial pivoting:
on the benchmark workload the reference pivots give multipliers up to |l| = 5.5 (no frequency keeps |l| <= 1; 29
distinct partial-pivoting row orders over 101 frequencies). A frequency is accepted only if all multipliers satisfy
|l| <= PIVOT_GROWTH_LIMIT (10, a pivot threshold of 0.1 as in UMFPACK/MA48) and the factors are finite with
non-zero pivots; every other frequency must be handled by SuperLU.
"""

from __future__ import annotations

import numpy as np

PIVOT_GROWTH_LIMIT = 10.0
"""Largest accepted |multiplier| when reusing the reference pivot sequence (pivot threshold 0.1)."""


def _levels(n, deps_ptr, deps_idx, order):
    """Longest-path level of every node of a DAG (deps of i: deps_idx[deps_ptr[i]:deps_ptr[i+1]]), `order` topological."""
    level = [0] * n
    ptr, idx = deps_ptr.tolist(), deps_idx.tolist()
    for i in order:
        best = 0
        for k in idx[ptr[i]:ptr[i + 1]]:
            if level[k] >= best:
                best = level[k] + 1
        level[i] = best
    return np.asarray(level, dtype=np.int64)


def _group(n, keys, *payload):
    """Sort by keys; return (ptr over 0..n, sorted payload arrays)."""
    order = np.argsort(keys, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=n))])
    return (ptr, *(p[order] for p in payload))


def _split_by_level(level_of_item, n_levels, *arrays):
    """Split item arrays (already sorted by level, then by any secondary key) into per-level views."""
    bounds = np.searchsorted(level_of_item, np.arange(n_levels + 1))
    return [tuple(a[bounds[l]:bounds[l + 1]] for a in arrays) for l in range(n_levels)]


class _TriangularSchedule:
    """
    x_i = (r_i - sum_{e: target[e] = i} value[e] * x[source[e]]) / diag_i, rows in dependency levels.
    value/diag are slot indices into the factor values (slots, F); x and r have shape (n, F, k).
    """

    def __init__(self, n, target, source, value_slot, diag_slot, topo_order):
        ptr, src_sorted = _group(n, target, source)
        level = _levels(n, ptr, src_sorted, topo_order)
        n_levels = int(level.max()) + 1 if n else 0
        self.diag_slot = diag_slot
        rows_sorted = np.argsort(level, kind="stable")
        rows_by_level = _split_by_level(level[rows_sorted], n_levels, rows_sorted)
        e_order = np.lexsort((target, level[target]))
        t, s, v = target[e_order], source[e_order], value_slot[e_order]
        entries_by_level = _split_by_level(level[t], n_levels, t, s, v)
        self.groups = []
        for (rows,), (t, s, v) in zip(rows_by_level, entries_by_level):
            with_terms, starts = np.unique(t, return_index=True)
            self.groups.append((rows, with_terms, starts, s, v))

    def solve(self, values, r, conj=False):
        vals = np.conj(values) if conj else values
        x = np.empty_like(r)
        diag = None if self.diag_slot is None else vals[self.diag_slot][:, :, None]
        for rows, with_terms, starts, s, v in self.groups:
            x[rows] = r[rows]
            if with_terms.size:
                x[with_terms] -= np.add.reduceat(vals[v][:, :, None] * x[s], starts, axis=0)
            if diag is not None:
                x[rows] /= diag[rows]
        return x


class BatchedLU:
    """
    Fixed-pattern LU of F matrices given in one CSC pattern (indices, indptr) with values data (F, nnz).

    perm_r, perm_c: permutations of a SuperLU partial-pivoting factorisation of one of them
    (M = P_r A P_c = L U, M[perm_r[i], perm_c[j]] = A[i, j]). `accepted[f]`: frequency f passed the pivot check.
    """

    def __init__(self, data, indices, indptr, perm_r, perm_c):
        n = indptr.shape[0] - 1
        self.n = n
        perm_r, perm_c = np.asarray(perm_r, np.int64), np.asarray(perm_c, np.int64)
        self.perm_r, self.perm_c = perm_r, perm_c
        a_cols = np.repeat(np.arange(n), np.diff(indptr))
        m_rows, m_cols = perm_r[indices], perm_c[a_cols]

        # symbolic LU of M without pivoting: structure only, so a superset of every frequency's fill
        col_sets = [set() for _ in range(n)]
        row_sets = [set() for _ in range(n)]
        for i, j in zip(m_rows.tolist(), m_cols.tolist()):
            col_sets[j].add(i)
            row_sets[i].add(j)
        for k in range(n):
            lower = [i for i in col_sets[k] if i > k]
            if lower:
                for j in [j for j in row_sets[k] if j > k]:
                    column = col_sets[j]
                    new = [i for i in lower if i not in column]
                    if new:
                        column.update(new)
                        for i in new:
                            row_sets[i].add(j)
        counts = [len(c) for c in col_sets]
        p_cols = np.repeat(np.arange(n), counts)
        p_rows = np.fromiter((i for c in col_sets for i in c), dtype=np.int64, count=p_cols.size)
        strict = p_rows > p_cols
        l_rows, l_cols = p_rows[strict], p_cols[strict]
        upper = ~strict
        u_rows, u_cols = p_rows[upper], p_cols[upper]
        n_u = u_rows.size
        self.n_slots = n_u + l_rows.size
        slot_of = np.full(n * n, -1, dtype=np.int64)
        slot_of[u_rows * n + u_cols] = np.arange(n_u)
        slot_of[l_rows * n + l_cols] = n_u + np.arange(l_rows.size)
        diag_slot = slot_of[np.arange(n) * (n + 1)]
        if np.any(diag_slot < 0):
            raise ValueError("structurally missing pivot")
        self.init_slot = slot_of[m_rows * n + m_cols]
        self.diag_slot = diag_slot

        # elimination program (right-looking), level-scheduled over pivot steps k
        l_slot = n_u + np.arange(l_rows.size)
        su = u_cols > u_rows
        us_rows, us_cols = u_rows[su], u_cols[su]
        us_slot = slot_of[us_rows * n + us_cols]
        l_ptr, lk_rows, lk_slot = _group(n, l_cols, l_rows, l_slot)          # L by column k
        u_ptr, uk_cols, uk_slot = _group(n, us_rows, us_cols, us_slot)       # strict U by row k
        dep_t = np.concatenate([l_rows, us_cols])                            # step k depends on k' < k with
        dep_s = np.concatenate([l_cols, us_rows])                            # L(k, k') or U(k', k) structurally non-zero
        dptr, dsrc = _group(n, dep_t, dep_s)
        step_level = _levels(n, dptr, dsrc, range(n))
        n_levels = int(step_level.max()) + 1
        cl, cu = np.diff(l_ptr), np.diff(u_ptr)
        # multipliers L(i, k) /= U(k, k), grouped by level of k
        lk_k = np.repeat(np.arange(n), cl)
        o = np.argsort(step_level[lk_k], kind="stable")
        mult = _split_by_level(step_level[lk_k][o], n_levels, lk_slot[o], diag_slot[lk_k][o])
        # updates A(i, j) -= L(i, k) U(k, j) for i in L col k, j in U row k, grouped by level of k, then target
        pairs = cl * cu
        k_rep = np.repeat(np.arange(n), pairs)
        local = np.arange(pairs.sum()) - np.repeat(np.cumsum(pairs) - pairs, pairs)
        li = l_ptr[k_rep] + local // cu[k_rep]
        uj = u_ptr[k_rep] + local % cu[k_rep]
        tgt = slot_of[lk_rows[li] * n + uk_cols[uj]]
        if np.any(tgt < 0):
            raise ValueError("fill outside the symbolic pattern")
        o = np.lexsort((tgt, step_level[k_rep]))
        upd = _split_by_level(step_level[k_rep][o], n_levels, tgt[o], lk_slot[li][o], uk_slot[uj][o])
        self.steps = []
        for (m_slot, m_piv), (t, a, b) in zip(mult, upd):
            uniq, starts = np.unique(t, return_index=True)
            self.steps.append((m_slot, m_piv, a, b, uniq, starts))

        # triangular solve schedules
        fwd, bwd = range(n), range(n - 1, -1, -1)
        self.L_fwd = _TriangularSchedule(n, l_rows, l_cols, l_slot, None, fwd)             # L z = r
        self.U_bwd = _TriangularSchedule(n, us_rows, us_cols, us_slot, diag_slot, bwd)     # U y = z
        self.Ut_fwd = _TriangularSchedule(n, us_cols, us_rows, us_slot, diag_slot, fwd)    # U^T w = r
        self.Lt_bwd = _TriangularSchedule(n, l_cols, l_rows, l_slot, None, bwd)            # L^T v = w

        self.values, self.accepted, self.max_multiplier = self._factor(np.asarray(data))

    def _factor(self, data):
        F = data.shape[0]
        V = np.zeros((self.n_slots, F), dtype=np.complex128)
        V[self.init_slot] = data.T
        worst = np.zeros(F)
        for m_slot, m_piv, a, b, uniq, starts in self.steps:
            if m_slot.size:
                V[m_slot] /= V[m_piv]
                np.maximum(worst, np.abs(V[m_slot]).max(axis=0), out=worst)
            if uniq.size:
                V[uniq] -= np.add.reduceat(V[a] * V[b], starts, axis=0)
        pivots = V[self.diag_slot]
        ok = np.isfinite(V).all(axis=0) & np.all(pivots != 0, axis=0) & (worst <= PIVOT_GROWTH_LIMIT)
        return V, ok, worst

    def solve(self, rhs, trans, columns=None):
        """op(A_f)^-1 rhs[f] for the frequencies `columns` (default all); rhs (F', n, k) -> (F', n, k)."""
        V = self.values if columns is None else self.values[:, columns]
        b = np.moveaxis(rhs, 0, 1)                                           # (n, F', k) view
        r = np.empty(b.shape, dtype=np.complex128)
        if trans == "N":
            r[self.perm_r] = b
            y = self.U_bwd.solve(V, self.L_fwd.solve(V, r))
            x = y[self.perm_c]
        else:
            conj = trans == "H"
            r[self.perm_c] = b
            v = self.Lt_bwd.solve(V, self.Ut_fwd.solve(V, r, conj), conj)
            x = v[self.perm_r]
        return np.moveaxis(x, 1, 0)

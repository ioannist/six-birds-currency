"""Kernel generators for reversible and driven Markov chains."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np

from currencymorphism.markov import normalize_rows


def reversible_kernel(
    G: nx.Graph,
    w: Mapping[Any, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a reversible row-stochastic kernel from an undirected graph."""
    nodes = list(G.nodes())
    n_states = len(nodes)
    if n_states == 0:
        msg = "G must contain at least one node."
        raise ValueError(msg)

    index = {node: i for i, node in enumerate(nodes)}
    S = np.zeros((n_states, n_states), dtype=np.float64)

    for u, v in G.edges():
        if u == v:
            continue

        if w is None:
            weight = 1.0 if rng is None else float(rng.uniform(0.5, 1.5))
        else:
            weight = _get_symmetric_weight(w, u, v)

        if weight <= 0.0:
            msg = f"Edge weight must be positive for edge ({u}, {v}); got {weight}."
            raise ValueError(msg)

        i = index[u]
        j = index[v]
        S[i, j] += weight
        S[j, i] += weight

    row_sums = S.sum(axis=1)
    isolated = row_sums == 0.0
    S[isolated, isolated] = 1.0

    pi = S.sum(axis=1)
    return S / pi[:, None]


def ring_drift_kernel(n: int, affinity_A: float, stay_prob: float = 0.0) -> np.ndarray:
    """Build an n-state ring kernel with calibrated net cycle affinity."""
    if n < 3:
        msg = f"n must be at least 3, got {n}."
        raise ValueError(msg)
    if not (0.0 <= stay_prob < 1.0):
        msg = f"stay_prob must satisfy 0 <= stay_prob < 1, got {stay_prob}."
        raise ValueError(msg)

    a = float(affinity_A) / float(n)
    base = (1.0 - stay_prob) / (2.0 * np.cosh(a / 2.0))
    cw = base * np.exp(a / 2.0)
    ccw = base * np.exp(-a / 2.0)

    P = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    P[idx, idx] = stay_prob
    P[idx, (idx + 1) % n] = cw
    P[idx, (idx - 1) % n] = ccw
    return P


def module_rings_chain(
    M: int,
    L: int,
    affinities: np.ndarray | list[float],
    bridge_weight: float,
) -> np.ndarray:
    """Build M drifted rings of size L connected by undirected bridges."""
    if M < 1:
        msg = f"M must be at least 1, got {M}."
        raise ValueError(msg)
    if L < 3:
        msg = f"L must be at least 3, got {L}."
        raise ValueError(msg)
    if bridge_weight < 0.0:
        msg = f"bridge_weight must be nonnegative, got {bridge_weight}."
        raise ValueError(msg)

    A = np.asarray(affinities, dtype=np.float64)
    if A.shape != (M,):
        msg = f"affinities must have length {M}, got shape {A.shape}."
        raise ValueError(msg)

    n_states = M * L
    W = np.zeros((n_states, n_states), dtype=np.float64)

    for m in range(M):
        a = A[m] / (2.0 * L)
        cw = float(np.exp(+a))
        ccw = float(np.exp(-a))
        base = m * L
        for i in range(L):
            s = base + i
            W[s, base + ((i + 1) % L)] += cw
            W[s, base + ((i - 1) % L)] += ccw

    for m in range(M - 1):
        u = m * L
        v = (m + 1) * L
        W[u, v] += bridge_weight
        W[v, u] += bridge_weight

    return normalize_rows(W, eps=0.0)


def _get_symmetric_weight(w: Mapping[Any, float], u: Any, v: Any) -> float:
    """Resolve symmetric edge weights from tuple or frozenset keys."""
    if (u, v) in w:
        return float(w[(u, v)])
    if (v, u) in w:
        return float(w[(v, u)])

    key = frozenset((u, v))
    if key in w:
        return float(w[key])

    msg = f"Missing weight for edge ({u}, {v})."
    raise ValueError(msg)

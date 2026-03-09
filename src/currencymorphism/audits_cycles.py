"""Cycle-based audits for Markov kernels."""

from __future__ import annotations

import networkx as nx
import numpy as np


def edge_log_ratio(P: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Compute antisymmetric edge log-ratios log(P_ij / P_ji) on bidirectional edges."""
    K = np.asarray(P, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be square, got shape {K.shape}."
        raise ValueError(msg)

    n = K.shape[0]
    a = np.zeros_like(K)
    mask = (K > eps) & (K.T > eps)
    mask[np.diag_indices(n)] = False
    a[mask] = np.log(K[mask]) - np.log(K.T[mask])
    return a


def undirected_support_graph(P: np.ndarray, thresh: float = 0.0) -> nx.Graph:
    """Build undirected bidirectional support graph from a transition matrix."""
    K = np.asarray(P, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be square, got shape {K.shape}."
        raise ValueError(msg)

    n = K.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] > thresh and K[j, i] > thresh:
                G.add_edge(i, j)
    return G


def cycle_basis(G: nx.Graph) -> list[list[int]]:
    """Return a fundamental cycle basis of an undirected graph."""
    return nx.cycle_basis(G)


def cycle_rank(G: nx.Graph) -> int:
    """Return cycle rank beta_1 = m - n + c."""
    m = G.number_of_edges()
    n = G.number_of_nodes()
    c = nx.number_connected_components(G)
    return int(m - n + c)


def cycle_affinities(
    P: np.ndarray,
    cycles: list[list[int]],
    orientation: str = "canonical",
) -> np.ndarray:
    """Compute cycle affinities from edge log-ratios along each cycle."""
    a = edge_log_ratio(P)
    out = np.zeros(len(cycles), dtype=np.float64)

    for idx, cyc in enumerate(cycles):
        if len(cyc) < 2:
            msg = "Each cycle must contain at least two distinct nodes."
            raise ValueError(msg)

        seq = list(cyc)
        if orientation == "canonical":
            seq = _canonical_cycle_orientation(seq)
        elif orientation != "given":
            msg = f"Unknown orientation: {orientation!r}."
            raise ValueError(msg)

        total = 0.0
        for t in range(len(seq)):
            u = seq[t]
            v = seq[(t + 1) % len(seq)]
            total += float(a[u, v])
        out[idx] = total

    return out


def _canonical_cycle_orientation(cycle: list[int]) -> list[int]:
    """Choose deterministic cycle orientation via min-rotation lexicographic rule."""
    fwd = _rotate_to_min(cycle)
    rev = _rotate_to_min(list(reversed(cycle)))
    return fwd if tuple(fwd) <= tuple(rev) else rev


def _rotate_to_min(seq: list[int]) -> list[int]:
    """Rotate sequence so the smallest node label appears first."""
    arr = list(seq)
    min_pos = int(np.argmin(arr))
    return arr[min_pos:] + arr[:min_pos]

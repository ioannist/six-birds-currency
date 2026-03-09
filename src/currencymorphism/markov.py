"""Utilities for finite-state Markov chains."""

from __future__ import annotations

import numpy as np


def normalize_rows(A: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Return a row-stochastic matrix from a numeric 2D array-like input."""
    P = np.asarray(A, dtype=np.float64)
    if P.ndim != 2:
        msg = f"A must be 2D, got shape {P.shape}."
        raise ValueError(msg)
    if eps < 0:
        msg = f"eps must be nonnegative, got {eps}."
        raise ValueError(msg)

    if eps > 0:
        P = P + eps

    P = np.clip(P, 0.0, None)
    row_sums = P.sum(axis=1)
    if np.any(row_sums == 0.0):
        zero_rows = np.where(row_sums == 0.0)[0]
        msg = (
            "Cannot normalize rows with zero mass after clipping/smoothing. "
            f"Zero-sum row indices: {zero_rows.tolist()}."
        )
        raise ValueError(msg)

    return P / row_sums[:, None]


def stationary_dist(
    P: np.ndarray,
    method: str = "power",
    tol: float = 1e-12,
    max_iter: int = 1_000_000,
) -> np.ndarray:
    """Compute a stationary distribution pi satisfying pi @ P ~= pi."""
    K = np.asarray(P, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be a square 2D array, got shape {K.shape}."
        raise ValueError(msg)

    if np.any(K < -1e-14):
        msg = "P has negative entries beyond numerical tolerance."
        raise ValueError(msg)

    row_sums = K.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        msg = "P must be row-stochastic (rows sum to 1 within tolerance)."
        raise ValueError(msg)

    n_states = K.shape[0]

    if method == "power":
        pi = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        for _ in range(max_iter):
            nxt = pi @ K
            nxt = np.clip(nxt, 0.0, None)
            total = float(nxt.sum())
            if total <= 0.0:
                msg = "Power iteration produced a zero vector after clipping."
                raise RuntimeError(msg)
            nxt /= total
            if np.linalg.norm(nxt - pi, ord=1) < tol:
                return nxt
            pi = nxt

        msg = f"Power iteration failed to converge within {max_iter} iterations."
        raise RuntimeError(msg)

    if method == "eigs":
        vals, vecs = np.linalg.eig(K.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        vec = np.real(vecs[:, idx])
        if float(vec.sum()) < 0.0:
            vec = -vec
        vec = np.clip(vec, 0.0, None)
        if float(vec.sum()) <= 0.0:
            vec = np.abs(np.real(vecs[:, idx]))
        total = float(vec.sum())
        if total <= 0.0:
            msg = "Eigenvector method failed to extract a nonzero stationary vector."
            raise RuntimeError(msg)
        return vec / total

    msg = f"Unknown method: {method!r}. Use 'power' or 'eigs'."
    raise ValueError(msg)


def simulate(P: np.ndarray, x0: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate one trajectory of length T+1 including the initial state."""
    K = np.asarray(P, dtype=np.float64)
    n_states = K.shape[0]
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be square, got shape {K.shape}."
        raise ValueError(msg)
    if T < 0:
        msg = f"T must be nonnegative, got {T}."
        raise ValueError(msg)
    if not (0 <= x0 < n_states):
        msg = f"x0 must be in [0, {n_states}), got {x0}."
        raise ValueError(msg)

    path = np.empty(T + 1, dtype=np.int64)
    path[0] = int(x0)
    for t in range(T):
        s = int(path[t])
        path[t + 1] = int(rng.choice(n_states, p=K[s]))
    return path


def simulate_many(
    P: np.ndarray,
    x0_dist: np.ndarray,
    T: int,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate N independent trajectories with vectorized stepping across chains."""
    K = np.asarray(P, dtype=np.float64)
    x0 = np.asarray(x0_dist, dtype=np.float64)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be square, got shape {K.shape}."
        raise ValueError(msg)
    n_states = K.shape[0]
    if x0.ndim != 1 or x0.shape[0] != n_states:
        msg = f"x0_dist must have shape ({n_states},), got {x0.shape}."
        raise ValueError(msg)
    if T < 0:
        msg = f"T must be nonnegative, got {T}."
        raise ValueError(msg)
    if N <= 0:
        msg = f"N must be positive, got {N}."
        raise ValueError(msg)
    if np.any(x0 < 0.0) or not np.isclose(x0.sum(), 1.0, atol=1e-12):
        msg = "x0_dist must be a valid probability distribution."
        raise ValueError(msg)

    paths = np.empty((N, T + 1), dtype=np.int64)
    paths[:, 0] = rng.choice(n_states, size=N, p=x0)

    for t in range(T):
        current = paths[:, t]
        probs = K[current]
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(N)
        nxt = (cdf < u[:, None]).sum(axis=1)
        paths[:, t + 1] = np.minimum(nxt, n_states - 1)

    return paths


def empirical_counts(seq: np.ndarray, n_states: int) -> np.ndarray:
    """Compute transition counts from one path or a batch of paths."""
    arr = np.asarray(seq)
    if n_states <= 0:
        msg = f"n_states must be positive, got {n_states}."
        raise ValueError(msg)

    counts = np.zeros((n_states, n_states), dtype=np.int64)

    if arr.ndim == 1:
        if arr.size < 2:
            return counts
        src = arr[:-1].astype(np.int64, copy=False)
        dst = arr[1:].astype(np.int64, copy=False)
    elif arr.ndim == 2:
        if arr.shape[1] < 2:
            return counts
        src = arr[:, :-1].astype(np.int64, copy=False).ravel()
        dst = arr[:, 1:].astype(np.int64, copy=False).ravel()
    else:
        msg = f"seq must be 1D or 2D, got ndim={arr.ndim}."
        raise ValueError(msg)

    invalid_src = np.any(src < 0) or np.any(src >= n_states)
    invalid_dst = np.any(dst < 0) or np.any(dst >= n_states)
    if invalid_src or invalid_dst:
        msg = "State indices in seq must be within [0, n_states)."
        raise ValueError(msg)

    np.add.at(counts, (src, dst), 1)
    return counts

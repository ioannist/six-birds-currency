"""Coarse-graining utilities for Markov-chain state partitions."""

from __future__ import annotations

import numpy as np

from currencymorphism.markov import stationary_dist


def _validate_part(part: np.ndarray, n: int | None = None) -> np.ndarray:
    """Validate partition labels as contiguous integers 0..k-1."""
    p = np.asarray(part, dtype=np.int64)
    if p.ndim != 1:
        msg = f"part must be a 1D array, got shape {p.shape}."
        raise ValueError(msg)
    if p.size == 0:
        msg = "part must be non-empty."
        raise ValueError(msg)
    if n is not None and p.shape[0] != n:
        msg = f"part must have length {n}, got {p.shape[0]}."
        raise ValueError(msg)
    if np.any(p < 0):
        msg = "part labels must be nonnegative integers."
        raise ValueError(msg)

    unique = np.unique(p)
    if unique[0] != 0:
        msg = "part labels must start at 0."
        raise ValueError(msg)
    expected = np.arange(unique.size, dtype=np.int64)
    if not np.array_equal(unique, expected):
        msg = "part labels must be contiguous 0..k-1."
        raise ValueError(msg)

    return p


def random_partition(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random partition with labels 0..k-1 and nonempty blocks."""
    if not (1 <= k <= n):
        msg = f"k must satisfy 1 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)

    part = np.empty(n, dtype=np.int64)
    part[:k] = np.arange(k, dtype=np.int64)
    if n > k:
        part[k:] = rng.integers(0, k, size=n - k, dtype=np.int64)
    rng.shuffle(part)
    return part


def block_partition(n: int, k: int) -> np.ndarray:
    """Generate contiguous index blocks with labels 0..k-1."""
    if not (1 <= k <= n):
        msg = f"k must satisfy 1 <= k <= n, got n={n}, k={k}."
        raise ValueError(msg)

    idx = np.arange(n, dtype=np.int64)
    return (idx * k) // n


def pushforward_dist(rho: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Aggregate a micro distribution into macro blocks."""
    rho_arr = np.asarray(rho, dtype=np.float64)
    p = _validate_part(part)

    if rho_arr.ndim != 1 or rho_arr.shape[0] != p.shape[0]:
        msg = f"rho must have shape ({p.shape[0]},), got {rho_arr.shape}."
        raise ValueError(msg)
    if np.any(rho_arr < 0.0):
        msg = "rho must be nonnegative."
        raise ValueError(msg)

    k = int(p.max()) + 1
    rho_macro = np.bincount(p, weights=rho_arr, minlength=k).astype(np.float64)

    if not np.isclose(rho_macro.sum(), rho_arr.sum(), atol=1e-12):
        msg = "pushforward_dist failed to preserve total mass within tolerance."
        raise RuntimeError(msg)

    return rho_macro


def lift_dist(
    rho_macro: np.ndarray,
    part: np.ndarray,
    scheme: str = "uniform",
    pi_micro: np.ndarray | None = None,
) -> np.ndarray:
    """Lift a macro distribution to micro states using a chosen within-block scheme."""
    p = _validate_part(part)
    n = p.shape[0]
    k = int(p.max()) + 1

    rho_m = np.asarray(rho_macro, dtype=np.float64)
    if rho_m.ndim != 1 or rho_m.shape[0] != k:
        msg = f"rho_macro must have shape ({k},), got {rho_m.shape}."
        raise ValueError(msg)
    if np.any(rho_m < 0.0):
        msg = "rho_macro must be nonnegative."
        raise ValueError(msg)

    rho_micro = np.zeros(n, dtype=np.float64)

    if scheme == "uniform":
        counts = np.bincount(p, minlength=k).astype(np.float64)
        rho_micro = rho_m[p] / counts[p]
    elif scheme == "stationary":
        if pi_micro is None:
            msg = "pi_micro is required when scheme='stationary'."
            raise ValueError(msg)
        pi = np.asarray(pi_micro, dtype=np.float64)
        if pi.ndim != 1 or pi.shape[0] != n:
            msg = f"pi_micro must have shape ({n},), got {pi.shape}."
            raise ValueError(msg)
        if np.any(pi < 0.0):
            msg = "pi_micro must be nonnegative."
            raise ValueError(msg)

        pi_block = np.bincount(p, weights=pi, minlength=k).astype(np.float64)
        if np.any(pi_block == 0.0):
            msg = "Each macro block must have positive pi_micro mass."
            raise ValueError(msg)
        rho_micro = rho_m[p] * (pi / pi_block[p])
    else:
        msg = f"Unknown scheme: {scheme!r}. Use 'uniform' or 'stationary'."
        raise ValueError(msg)

    if not np.isclose(rho_micro.sum(), rho_m.sum(), atol=1e-12):
        msg = "lift_dist failed to preserve total mass within tolerance."
        raise RuntimeError(msg)

    return rho_micro


def lumped_kernel(
    P: np.ndarray,
    part: np.ndarray,
    pi: np.ndarray | None = None,
) -> np.ndarray:
    """Compute macro transition kernel via stationary-conditional averaging."""
    K = np.asarray(P, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be square, got shape {K.shape}."
        raise ValueError(msg)
    if np.any(K < -1e-14):
        msg = "P must be nonnegative up to numerical tolerance."
        raise ValueError(msg)
    if not np.allclose(K.sum(axis=1), 1.0, atol=1e-10):
        msg = "P must be row-stochastic (rows sum to 1)."
        raise ValueError(msg)

    n = K.shape[0]
    p = _validate_part(part, n=n)
    k = int(p.max()) + 1

    if pi is None:
        pi_micro = stationary_dist(K, method="power", tol=1e-14)
    else:
        pi_micro = np.asarray(pi, dtype=np.float64)
        if pi_micro.ndim != 1 or pi_micro.shape[0] != n:
            msg = f"pi must have shape ({n},), got {pi_micro.shape}."
            raise ValueError(msg)
        if np.any(pi_micro < 0.0):
            msg = "pi must be nonnegative."
            raise ValueError(msg)
        total = float(pi_micro.sum())
        if total <= 0.0:
            msg = "pi must have positive total mass."
            raise ValueError(msg)
        pi_micro = pi_micro / total

    pi_macro = np.bincount(p, weights=pi_micro, minlength=k).astype(np.float64)
    if np.any(pi_macro == 0.0):
        msg = "Cannot condition on macro blocks with zero stationary mass."
        raise ValueError(msg)

    M = np.zeros((n, k), dtype=np.float64)
    M[np.arange(n), p] = 1.0

    C = np.zeros((k, n), dtype=np.float64)
    C[p, np.arange(n)] = pi_micro / pi_macro[p]

    Q = C @ K @ M
    Q = np.clip(Q, 0.0, None)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q


def coarse_path(seq_micro: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Map micro state trajectories to macro labels using the partition."""
    p = _validate_part(part)
    seq = np.asarray(seq_micro)

    if seq.ndim not in (1, 2):
        msg = f"seq_micro must be 1D or 2D, got shape {seq.shape}."
        raise ValueError(msg)
    if not np.issubdtype(seq.dtype, np.integer):
        msg = "seq_micro must have integer dtype."
        raise ValueError(msg)

    n = p.shape[0]
    if np.any(seq < 0) or np.any(seq >= n):
        msg = f"seq_micro entries must be in [0, {n - 1}]."
        raise ValueError(msg)

    return p[seq]

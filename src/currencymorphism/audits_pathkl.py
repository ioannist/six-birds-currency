"""Path-reversal KL audits for Markov processes and trajectory data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SigmaEstimate:
    """KL value (possibly infinite) and raw reverse-support diagnostics."""

    value: float
    infinite_risk_mass: float
    n_windows: int


def sigma_T_markov(P: np.ndarray, rho0: np.ndarray, T: int) -> SigmaEstimate:
    """Compute path-reversal KL for a Markov chain over horizon T."""
    if not isinstance(T, (int, np.integer)) or T < 1:
        msg = f"T must be at least 1, got {T}."
        raise ValueError(msg)

    K = np.asarray(P, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        msg = f"P must be square, got shape {K.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(K)) or np.any(K < 0.0):
        msg = "P must be finite and nonnegative."
        raise ValueError(msg)
    if not np.allclose(K.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        msg = "P must be row-stochastic (rows sum to 1 within tolerance)."
        raise ValueError(msg)

    n = K.shape[0]
    rho = np.asarray(rho0, dtype=np.float64)
    if rho.ndim != 1 or rho.shape[0] != n:
        msg = f"rho0 must have shape ({n},), got {rho.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(rho)) or np.any(rho < 0.0):
        msg = "rho0 must be nonnegative."
        raise ValueError(msg)
    rho_sum = float(rho.sum())
    if not np.isfinite(rho_sum) or rho_sum <= 0.0:
        msg = "rho0 must have positive finite total mass."
        raise ValueError(msg)
    rho = rho / rho_sum

    rho_init = rho.copy()
    one_way = (K > 0.0) & (K.T == 0.0)

    two_way = (K > 0.0) & (K.T > 0.0)
    log_ratio = np.zeros_like(K)
    log_ratio[two_way] = np.log(K[two_way]) - np.log(K.T[two_way])

    # Probability of paths with bidirectional edges and a supported endpoint.
    safe_rho = rho.copy()
    safe_kernel = np.where(one_way, 0.0, K)
    singular = False
    step_sum = 0.0
    for _ in range(T):
        singular |= bool(np.any((rho[:, None] * K)[one_way] > 0.0))
        safe_rho = safe_rho @ safe_kernel
        flow = rho[:, None] * K
        step_sum += float(np.sum(flow[two_way] * log_ratio[two_way]))
        rho = rho @ K

    rho_T = rho
    support_violation = (rho_T > 0.0) & (rho_init == 0.0)
    if singular or np.any(support_violation):
        safe_mass = float(safe_rho[rho_init > 0.0].sum())
        return SigmaEstimate(
            value=np.inf,
            infinite_risk_mass=float(np.clip(1.0 - safe_mass, 0.0, 1.0)),
            n_windows=0,
        )

    init_support = rho_init > 0.0
    term_start = float(np.sum(rho_init[init_support] * np.log(rho_init[init_support])))
    log_rho_init = np.zeros_like(rho_init)
    np.log(rho_init, out=log_rho_init, where=rho_init > 0.0)
    term_end = float(np.sum(rho_T * log_rho_init))

    return SigmaEstimate(
        value=term_start - term_end + step_sum,
        infinite_risk_mass=0.0,
        n_windows=0,
    )


def sigma_T_empirical(
    paths: np.ndarray, T: int, *, reverse_mix: float = 0.0
) -> SigmaEstimate:
    """KL of the empirical window law against its reversal.

    Missing reverses give infinity when reverse_mix=0. For 0<reverse_mix<1/2,
    use p_mix=(1-reverse_mix)*p + reverse_mix*R*p and its reversal. This
    regularization commutes with deterministic pointwise coarse observation,
    so DPI applies on the same sample. Risk mass always describes the raw law.
    """
    if not np.isfinite(reverse_mix) or not 0.0 <= reverse_mix < 0.5:
        raise ValueError("reverse_mix must be finite and in [0, 0.5).")
    if not isinstance(T, (int, np.integer)) or T < 1:
        msg = f"T must be at least 1, got {T}."
        raise ValueError(msg)

    arr = np.asarray(paths)
    if arr.ndim not in (1, 2):
        msg = f"paths must be 1D or 2D, got shape {arr.shape}."
        raise ValueError(msg)
    if not np.issubdtype(arr.dtype, np.integer):
        msg = "paths must have integer dtype."
        raise ValueError(msg)

    windows = _extract_windows(arr, T)
    n_windows = windows.shape[0]
    if n_windows == 0:
        raise ValueError("No windows: the empirical path law is undefined.")

    windows = np.ascontiguousarray(windows)
    itemsize = windows.dtype.itemsize
    key_bytes = itemsize * (T + 1)
    key_dtype = np.dtype((np.void, key_bytes))
    keys = windows.view(key_dtype).ravel()

    uniq, counts = np.unique(keys, return_counts=True)
    count_map: dict[bytes, int] = {}
    for idx in range(uniq.shape[0]):
        count_map[bytes(uniq[idx])] = int(counts[idx])

    total = float(n_windows)
    inf_mass = sum(
        c / total
        for key, c in count_map.items()
        if _reverse_key_bytes(key, chunk_size=itemsize) not in count_map
    )
    if reverse_mix == 0.0 and inf_mass > 0.0:
        return SigmaEstimate(np.inf, float(inf_mass), n_windows)

    # Include the reversal of every observed key, even when it was unobserved.
    support = set(count_map)
    support.update(_reverse_key_bytes(key, itemsize) for key in count_map)
    value = 0.0
    for key in support:
        c = count_map.get(key, 0)
        c_rev = count_map.get(_reverse_key_bytes(key, itemsize), 0)
        p = ((1.0 - reverse_mix) * c + reverse_mix * c_rev) / total
        q = ((1.0 - reverse_mix) * c_rev + reverse_mix * c) / total
        if p > 0.0:
            value += p * (np.log(p) - np.log(q))
    return SigmaEstimate(float(value), float(inf_mass), n_windows)


def sigma_T_empirical_from_single_path(seq: np.ndarray, T: int) -> SigmaEstimate:
    """Convenience wrapper to estimate from a single long trajectory."""
    return sigma_T_empirical(seq, T)


def _extract_windows(paths: np.ndarray, T: int) -> np.ndarray:
    """Return stacked contiguous windows of length T+1 from 1D or 2D paths."""
    win_len = T + 1
    if paths.ndim == 1:
        if paths.shape[0] <= T:
            return np.empty((0, win_len), dtype=paths.dtype)
        return np.lib.stride_tricks.sliding_window_view(paths, win_len)

    if paths.shape[1] <= T:
        return np.empty((0, win_len), dtype=paths.dtype)
    w = np.lib.stride_tricks.sliding_window_view(paths, win_len, axis=1)
    return w.reshape(-1, win_len)


def _reverse_key_bytes(key: bytes, chunk_size: int) -> bytes:
    """Reverse fixed-width chunks in a serialized integer-window key."""
    n_chunks = len(key) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start = (n_chunks - 1 - i) * chunk_size
        end = start + chunk_size
        chunks.append(key[start:end])
    return b"".join(chunks)

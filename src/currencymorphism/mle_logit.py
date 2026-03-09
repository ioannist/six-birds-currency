"""Conditional-logit MLE for single-parameter transition pricing."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def fit_lambda(
    C: np.ndarray,
    u: np.ndarray,
    *,
    lam_min: float = 0.0,
    lam_max: float = 50.0,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """Fit lambda by conditional-logit MLE and return (lambda_hat, se_hat)."""
    counts = np.asarray(C, dtype=np.float64)
    cost = np.asarray(u, dtype=np.float64)

    if counts.ndim != 2 or cost.ndim != 2 or counts.shape != cost.shape:
        msg = (
            "C and u must be 2D and shape-matched, "
            f"got {counts.shape} and {cost.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(cost)):
        msg = "C and u must contain only finite values."
        raise ValueError(msg)
    if np.any(counts < 0.0):
        msg = "C must be nonnegative."
        raise ValueError(msg)
    if np.any(cost < -1e-15):
        msg = "u must be nonnegative up to numerical tolerance."
        raise ValueError(msg)
    if counts.sum() <= 0.0:
        msg = "C must contain positive total count mass."
        raise ValueError(msg)

    cost = np.clip(cost, 0.0, None)

    if lam_min < 0.0:
        msg = f"lam_min must be nonnegative, got {lam_min}."
        raise ValueError(msg)
    if lam_max <= lam_min:
        msg = f"lam_max must be greater than lam_min, got {lam_max} <= {lam_min}."
        raise ValueError(msg)
    if tol <= 0.0:
        msg = f"tol must be positive, got {tol}."
        raise ValueError(msg)

    score_lo = _score(lam_min, counts, cost)
    if score_lo <= 0.0:
        lam_hat = float(lam_min)
        info_hat = _info(lam_hat, counts, cost)
        if _is_non_identifiable(lam_hat, counts, cost) or info_hat <= 0.0:
            return lam_hat, float(np.inf)
        return lam_hat, float(np.sqrt(1.0 / info_hat))

    hi = max(lam_max, lam_min + 1.0)
    score_hi = _score(hi, counts, cost)
    while score_hi > 0.0:
        hi *= 2.0
        if hi > 1e6:
            msg = "Failed to bracket root before lambda exceeded 1e6."
            raise RuntimeError(msg)
        score_hi = _score(hi, counts, cost)

    lam_hat = float(
        brentq(lambda x: _score(x, counts, cost), lam_min, hi, xtol=tol, rtol=tol)
    )
    info_hat = _info(lam_hat, counts, cost)
    if _is_non_identifiable(lam_hat, counts, cost) or info_hat <= 0.0:
        return lam_hat, float(np.inf)

    se_hat = float(np.sqrt(1.0 / info_hat))
    return lam_hat, se_hat


def _logq(lam: float, u: np.ndarray) -> np.ndarray:
    """Compute row-wise log-probabilities under q_ij(lam) ∝ exp(-lam*u_ij)."""
    x = -lam * u
    return x - logsumexp(x, axis=1, keepdims=True)


def _row_moments(lam: float, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row E[u] and E[u^2] under q(lam)."""
    logq = _logq(lam, u)
    q = np.exp(logq)
    eu = np.sum(q * u, axis=1)
    eu2 = np.sum(q * (u * u), axis=1)
    return eu, eu2


def _score(lam: float, C: np.ndarray, u: np.ndarray) -> float:
    """Compute d/dlam log-likelihood for transition counts C."""
    row_counts = C.sum(axis=1)
    eu, _ = _row_moments(lam, u)
    observed = np.sum(C * u, axis=1)
    return float(-np.sum(observed - row_counts * eu))


def _info(lam: float, C: np.ndarray, u: np.ndarray) -> float:
    """Compute observed Fisher information I = -ell''(lam)."""
    row_counts = C.sum(axis=1)
    eu, eu2 = _row_moments(lam, u)
    var = np.maximum(eu2 - eu * eu, 0.0)
    return float(np.sum(row_counts * var))


def _is_non_identifiable(lam: float, C: np.ndarray, u: np.ndarray) -> bool:
    """Return True when all informative rows have near-zero conditional variance."""
    row_counts = C.sum(axis=1)
    informative = row_counts > 0.0
    if not np.any(informative):
        return True

    eu, eu2 = _row_moments(lam, u)
    var = np.maximum(eu2 - eu * eu, 0.0)
    return bool(np.all(var[informative] <= 1e-15))

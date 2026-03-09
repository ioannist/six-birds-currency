"""Single-constraint maximum-caliber transition kernels."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import brentq


def maxent_kernel(u: np.ndarray, lam: float) -> np.ndarray:
    """Return per-row softmax kernel q_ij(lam) proportional to exp(-lam*u_ij)."""
    if lam < 0.0:
        msg = f"lam must be nonnegative, got {lam}."
        raise ValueError(msg)

    U = np.asarray(u, dtype=np.float64)
    if U.ndim != 2:
        msg = f"u must be 2D, got shape {U.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(U)):
        msg = "u must contain only finite entries."
        raise ValueError(msg)
    if np.any(U < -1e-15):
        msg = "u must be nonnegative up to numerical tolerance."
        raise ValueError(msg)
    U = np.clip(U, 0.0, None)

    x = -lam * U
    row_max = np.max(x, axis=1, keepdims=True)
    if not np.all(np.isfinite(row_max)):
        msg = "Encountered invalid row maxima during softmax stabilization."
        raise ValueError(msg)
    x = x - row_max

    ex = np.exp(x)
    row_sums = ex.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
        msg = "Encountered invalid row normalization in softmax."
        raise ValueError(msg)

    q = ex / row_sums
    if np.any(~np.isfinite(q)):
        msg = "maxent_kernel produced non-finite probabilities."
        raise ValueError(msg)
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-12):
        msg = "maxent_kernel row sums are not 1 within tolerance."
        raise ValueError(msg)

    return q


def expected_cost(q: np.ndarray, u: np.ndarray, mu: np.ndarray | None = None) -> float:
    """Compute weighted expected transition cost under row kernel q."""
    Q = np.asarray(q, dtype=np.float64)
    U = np.asarray(u, dtype=np.float64)

    if Q.ndim != 2 or U.ndim != 2 or Q.shape != U.shape:
        msg = f"q and u must be 2D with identical shape, got {Q.shape} and {U.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(U)):
        msg = "q and u must be finite."
        raise ValueError(msg)
    if np.any(Q < -1e-12):
        msg = "q must be nonnegative up to numerical tolerance."
        raise ValueError(msg)
    if np.any(U < -1e-15):
        msg = "u must be nonnegative up to numerical tolerance."
        raise ValueError(msg)
    if not np.allclose(Q.sum(axis=1), 1.0, atol=1e-10):
        msg = "Rows of q must sum to 1."
        raise ValueError(msg)

    n_rows = Q.shape[0]
    if mu is None:
        mu_arr = np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    else:
        mu_arr = np.asarray(mu, dtype=np.float64)
        if mu_arr.ndim != 1 or mu_arr.shape[0] != n_rows:
            msg = f"mu must have shape ({n_rows},), got {mu_arr.shape}."
            raise ValueError(msg)
        if not np.all(np.isfinite(mu_arr)):
            msg = "mu must be finite."
            raise ValueError(msg)
        if np.any(mu_arr < 0.0):
            msg = "mu must be nonnegative."
            raise ValueError(msg)
        mass = float(mu_arr.sum())
        if mass <= 0.0:
            msg = "mu must have positive total mass."
            raise ValueError(msg)
        mu_arr = mu_arr / mass

    row_cost = np.sum(Q * np.clip(U, 0.0, None), axis=1)
    return float(np.sum(mu_arr * row_cost))


def solve_lambda_for_budget(
    u: np.ndarray,
    b: float,
    bracket: tuple[float, float] = (0.0, 50.0),
    tol: float = 1e-10,
    mu: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    """Solve for lambda such that expected cost matches target budget b."""
    if tol <= 0.0:
        msg = f"tol must be positive, got {tol}."
        raise ValueError(msg)

    U = np.asarray(u, dtype=np.float64)
    if U.ndim != 2:
        msg = f"u must be 2D, got shape {U.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(U)):
        msg = "u must contain only finite entries."
        raise ValueError(msg)
    if np.any(U < -1e-15):
        msg = "u must be nonnegative up to numerical tolerance."
        raise ValueError(msg)
    U = np.clip(U, 0.0, None)

    n_rows = U.shape[0]
    if mu is None:
        mu_arr = np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    else:
        mu_arr = np.asarray(mu, dtype=np.float64)
        if mu_arr.ndim != 1 or mu_arr.shape[0] != n_rows:
            msg = f"mu must have shape ({n_rows},), got {mu_arr.shape}."
            raise ValueError(msg)
        if not np.all(np.isfinite(mu_arr)):
            msg = "mu must be finite."
            raise ValueError(msg)
        if np.any(mu_arr < 0.0):
            msg = "mu must be nonnegative."
            raise ValueError(msg)
        mass = float(mu_arr.sum())
        if mass <= 0.0:
            msg = "mu must have positive total mass."
            raise ValueError(msg)
        mu_arr = mu_arr / mass

    lo, hi = bracket
    if lo < 0.0 or hi <= lo:
        msg = f"Invalid bracket {bracket}; require 0 <= lo < hi."
        raise ValueError(msg)

    if not np.isfinite(b):
        msg = f"b must be finite, got {b}."
        raise ValueError(msg)

    q0 = maxent_kernel(U, 0.0)
    cost0 = expected_cost(q0, U, mu_arr)
    if b >= cost0 - tol:
        return 0.0, q0, cost0

    cost_inf_approx = float(np.sum(mu_arr * np.min(U, axis=1)))
    if b < cost_inf_approx - 1e-12:
        msg = (
            "budget below achievable minimum: "
            f"b={b}, minimum={cost_inf_approx}."
        )
        raise ValueError(msg)

    def cost_at(lam_val: float) -> float:
        q_val = maxent_kernel(U, lam_val)
        return expected_cost(q_val, U, mu_arr)

    f_lo = cost_at(lo) - b
    if f_lo < 0.0:
        msg = "Lower bracket endpoint already below budget; adjust bracket."
        raise ValueError(msg)

    f_hi = cost_at(hi) - b
    while f_hi > 0.0:
        hi *= 2.0
        if hi > 1e6:
            msg = "Failed to bracket solution before lambda exceeded 1e6."
            raise RuntimeError(msg)
        f_hi = cost_at(hi) - b

    lam = cast(float, brentq(lambda x: cost_at(x) - b, lo, hi, xtol=tol, rtol=tol))
    q = maxent_kernel(U, lam)
    achieved = expected_cost(q, U, mu_arr)
    if abs(achieved - b) > 10.0 * tol:
        msg = (
            "Root solve did not hit budget within tolerance: "
            f"|{achieved} - {b}| > {10.0 * tol}."
        )
        raise RuntimeError(msg)

    return lam, q, achieved

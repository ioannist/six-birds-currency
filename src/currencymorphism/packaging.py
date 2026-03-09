"""Packaging endomap and idempotence-defect diagnostics."""

from __future__ import annotations

import numpy as np

from currencymorphism.lens import lift_dist, pushforward_dist
from currencymorphism.markov import stationary_dist


def E_tau_f(
    mu: np.ndarray,
    P: np.ndarray,
    tau: int,
    part: np.ndarray,
    prototype: str = "stationary",
    pi_micro: np.ndarray | None = None,
) -> np.ndarray:
    """Apply empirical packaging endomap E_{tau,f}(mu)."""
    if not isinstance(tau, int) or tau < 0:
        msg = f"tau must be a nonnegative integer, got {tau}."
        raise ValueError(msg)

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
    mu_arr = np.asarray(mu, dtype=np.float64)
    if mu_arr.ndim != 1 or mu_arr.shape[0] != n:
        msg = f"mu must have shape ({n},), got {mu_arr.shape}."
        raise ValueError(msg)
    if np.any(mu_arr < 0.0):
        msg = "mu must be nonnegative."
        raise ValueError(msg)
    total = float(mu_arr.sum())
    if total <= 0.0:
        msg = "mu must have positive total mass."
        raise ValueError(msg)
    mu_arr = mu_arr / total

    part_arr = np.asarray(part, dtype=np.int64)
    _ = pushforward_dist(np.ones(n, dtype=np.float64), part_arr)

    nu = mu_arr.copy()
    for _ in range(tau):
        nu = nu @ K

    rho_macro = pushforward_dist(nu, part_arr)

    if prototype == "uniform":
        lifted = lift_dist(rho_macro, part_arr, scheme="uniform")
    elif prototype == "stationary":
        pi = (
            stationary_dist(K, method="eigs")
            if pi_micro is None
            else np.asarray(pi_micro, dtype=np.float64)
        )
        lifted = lift_dist(rho_macro, part_arr, scheme="stationary", pi_micro=pi)
    else:
        msg = f"Unsupported prototype: {prototype!r}."
        raise ValueError(msg)

    lifted = np.clip(lifted, 0.0, None)
    mass = float(lifted.sum())
    if mass <= 0.0:
        msg = "Packaged distribution has zero mass after clipping."
        raise RuntimeError(msg)
    lifted = lifted / mass

    if not np.isfinite(lifted).all():
        raise RuntimeError("Packaged distribution contains non-finite values.")

    return lifted


def idempotence_defect(
    mu: np.ndarray,
    P: np.ndarray,
    tau: int,
    part: np.ndarray,
    prototype: str = "stationary",
    pi_micro: np.ndarray | None = None,
) -> float:
    """Compute L1 idempotence defect ||E(E(mu)) - E(mu)||_1."""
    e_mu = E_tau_f(
        mu,
        P,
        tau,
        part,
        prototype=prototype,
        pi_micro=pi_micro,
    )
    e_e_mu = E_tau_f(
        e_mu,
        P,
        tau,
        part,
        prototype=prototype,
        pi_micro=pi_micro,
    )

    defect = float(np.linalg.norm(e_e_mu - e_mu, ord=1))
    if not np.isfinite(defect) or defect < 0.0:
        raise RuntimeError("Idempotence defect must be finite and nonnegative.")
    return defect

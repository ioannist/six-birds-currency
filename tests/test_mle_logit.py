import numpy as np

from currencymorphism.maxcal_single import maxent_kernel
from currencymorphism.mle_logit import fit_lambda


def test_synthetic_recovery() -> None:
    rng = np.random.default_rng(20260305)
    n = 8
    u = rng.random((n, n)) + 0.1
    lambda_true = 3.0

    q_true = maxent_kernel(u, lambda_true)
    n_per_row = 10_000
    C = np.vstack([rng.multinomial(n_per_row, q_true[i]) for i in range(n)]).astype(
        np.float64
    )

    lam_hat, se_hat = fit_lambda(C, u)

    assert abs(lam_hat - lambda_true) < 0.1
    assert np.isfinite(se_hat)
    assert se_hat > 0.0

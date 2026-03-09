import numpy as np

from currencymorphism.lens import random_partition
from currencymorphism.markov import normalize_rows
from currencymorphism.packaging import E_tau_f, idempotence_defect


def test_E_tau_f_preserves_mass() -> None:
    rng = np.random.default_rng(123)
    n = 12
    k = 4

    A = rng.random((n, n))
    P = normalize_rows(A, eps=1e-3)
    part = random_partition(n=n, k=k, rng=rng)

    mu = rng.random(n)
    mu = mu / mu.sum()

    nu = E_tau_f(mu, P, tau=3, part=part, prototype="uniform")

    assert nu.shape == (n,)
    assert np.all(nu >= -1e-15)
    assert abs(float(nu.sum()) - 1.0) < 1e-12


def test_idempotence_defect_zero_on_closure_compatible_kernel() -> None:
    P = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    part = np.array([0, 0, 1, 1], dtype=np.int64)
    mu = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)

    defect = idempotence_defect(mu, P, tau=1, part=part, prototype="uniform")
    assert defect < 1e-12


def test_idempotence_defect_nonnegative_and_finite() -> None:
    rng = np.random.default_rng(456)
    n = 10
    k = 5

    P = normalize_rows(rng.random((n, n)), eps=1e-3)
    part = random_partition(n=n, k=k, rng=rng)
    mu = rng.random(n)
    mu /= mu.sum()

    defect = idempotence_defect(mu, P, tau=2, part=part, prototype="stationary")

    assert np.isfinite(defect)
    assert defect >= 0.0

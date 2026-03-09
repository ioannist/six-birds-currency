import numpy as np

from currencymorphism.markov import normalize_rows, stationary_dist


def test_normalize_rows_outputs_stochastic_matrix() -> None:
    rng = np.random.default_rng(123)
    A = rng.random((8, 8)) - 0.2

    P = normalize_rows(A, eps=1e-3)

    assert np.all(P >= 0.0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_stationary_distribution_power_small_residual() -> None:
    rng = np.random.default_rng(456)
    A = rng.random((10, 10))
    P = normalize_rows(A, eps=1e-3)

    pi = stationary_dist(P, method="power", tol=1e-14, max_iter=1_000_000)
    err = np.linalg.norm(pi @ P - pi, ord=1)

    assert np.all(pi >= 0.0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-12)
    assert err < 1e-8


def test_stationary_distribution_eigs_matches_power() -> None:
    rng = np.random.default_rng(789)
    A = rng.random((10, 10))
    P = normalize_rows(A, eps=1e-3)

    pi_power = stationary_dist(P, method="power", tol=1e-14, max_iter=1_000_000)
    pi_eigs = stationary_dist(P, method="eigs")

    err = np.linalg.norm(pi_eigs @ P - pi_eigs, ord=1)
    assert err < 1e-8
    assert np.linalg.norm(pi_power - pi_eigs, ord=1) < 1e-6

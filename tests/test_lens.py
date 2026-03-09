import numpy as np

from currencymorphism.lens import (
    lift_dist,
    lumped_kernel,
    pushforward_dist,
    random_partition,
)
from currencymorphism.markov import normalize_rows


def test_pushforward_then_uniform_lift_preserves_mass() -> None:
    rng = np.random.default_rng(123)
    n = 20
    k = 5
    part = random_partition(n=n, k=k, rng=rng)

    rho = rng.random(n)
    rho_macro = pushforward_dist(rho, part)
    rho_lift = lift_dist(rho_macro, part, scheme="uniform")

    assert abs(rho_lift.sum() - rho.sum()) < 1e-12
    assert np.all(rho_lift >= 0.0)


def test_lumped_kernel_stochastic_and_nonnegative() -> None:
    rng = np.random.default_rng(456)
    n = 20
    k = 5
    A = rng.random((n, n))
    P = normalize_rows(A, eps=1e-3)
    part = random_partition(n=n, k=k, rng=rng)

    Q = lumped_kernel(P, part)

    assert Q.shape == (k, k)
    assert np.all(Q >= -1e-15)
    assert Q.min() >= -1e-15
    assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-12)

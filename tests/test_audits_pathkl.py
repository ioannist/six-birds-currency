import networkx as nx
import numpy as np

from currencymorphism.audits_pathkl import sigma_T_empirical, sigma_T_markov
from currencymorphism.generators import reversible_kernel, ring_drift_kernel
from currencymorphism.markov import simulate_many, stationary_dist


def test_sigma_reversible_is_near_zero_analytic_and_empirical() -> None:
    rng = np.random.default_rng(123)
    G = nx.cycle_graph(12)
    P = reversible_kernel(G)
    pi = stationary_dist(P, method="eigs")
    T = 5

    sig_a = sigma_T_markov(P, pi, T)
    assert sig_a.value < 1e-6

    paths = simulate_many(P, pi, T=T, N=10_000, rng=rng)
    paths_sym = np.concatenate([paths, paths[:, ::-1]], axis=0)
    sig_e = sigma_T_empirical(paths_sym, T)

    assert sig_e.value < 1e-6
    assert sig_e.infinite_risk_mass < 1e-12


def test_sigma_driven_is_positive_analytic_and_empirical() -> None:
    rng = np.random.default_rng(456)
    n = 10
    A_target = 2.0
    T = 5

    P = ring_drift_kernel(n=n, affinity_A=A_target, stay_prob=0.0)
    rho0 = np.full(n, 1.0 / n)

    sig_a = sigma_T_markov(P, rho0, T)
    paths = simulate_many(P, rho0, T=T, N=50_000, rng=rng)
    sig_e = sigma_T_empirical(paths, T)

    assert sig_a.value > 1e-3
    assert sig_e.value > 1e-3
    assert sig_e.infinite_risk_mass < 1e-12

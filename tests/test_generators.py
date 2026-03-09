import networkx as nx
import numpy as np

from currencymorphism.generators import (
    module_rings_chain,
    reversible_kernel,
    ring_drift_kernel,
)
from currencymorphism.markov import stationary_dist


def test_reversible_kernel_detailed_balance_residual_is_small() -> None:
    G = nx.cycle_graph(12)
    P = reversible_kernel(G)
    pi = stationary_dist(P, method="eigs")

    residual = np.max(np.abs(pi[:, None] * P - pi[None, :] * P.T))

    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert residual < 1e-8


def test_ring_drift_kernel_affinity_calibration() -> None:
    n = 25
    A_target = 1.7
    P = ring_drift_kernel(n=n, affinity_A=A_target, stay_prob=0.1)

    idx = np.arange(n)
    A_hat = np.sum(np.log(P[idx, (idx + 1) % n] / P[(idx + 1) % n, idx]))

    assert abs(A_hat - A_target) < 1e-2
    assert A_hat * A_target > 0.0


def test_module_rings_chain_shape_and_stochasticity() -> None:
    P = module_rings_chain(M=3, L=7, affinities=[0.0, 0.8, -0.5], bridge_weight=0.3)

    assert P.shape == (21, 21)
    assert np.all(P >= 0.0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)

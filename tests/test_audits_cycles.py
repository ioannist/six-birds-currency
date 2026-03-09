import networkx as nx
import numpy as np

from currencymorphism.audits_cycles import (
    cycle_affinities,
    cycle_basis,
    cycle_rank,
    undirected_support_graph,
)
from currencymorphism.generators import (
    module_rings_chain,
    reversible_kernel,
    ring_drift_kernel,
)
from currencymorphism.lens import lumped_kernel


def test_reversible_cycle_affinities_are_near_zero() -> None:
    G = nx.cycle_graph(12)
    P = reversible_kernel(G)

    Gsupp = undirected_support_graph(P, thresh=0.0)
    cycles = cycle_basis(Gsupp)
    aff = cycle_affinities(P, cycles, orientation="canonical")

    assert len(cycles) >= 1
    assert np.allclose(aff, 0.0, atol=1e-10)


def test_ring_drift_cycle_affinity_matches_target() -> None:
    n = 25
    A_target = 1.7
    P = ring_drift_kernel(n=n, affinity_A=A_target, stay_prob=0.1)

    Gsupp = undirected_support_graph(P, thresh=0.0)
    cycles = cycle_basis(Gsupp)
    aff = cycle_affinities(P, cycles, orientation="canonical")

    assert len(cycles) == 1
    assert abs(aff[0] - A_target) < 1e-2
    assert aff[0] * A_target > 0.0


def test_module_rings_chain_macro_cycle_rank_is_zero() -> None:
    M = 3
    L = 10
    P_micro = module_rings_chain(
        M=M, L=L, affinities=[0.5, 1.0, -0.7], bridge_weight=0.2
    )
    part = np.arange(M * L, dtype=np.int64) // L
    Q = lumped_kernel(P_micro, part)

    G_micro = undirected_support_graph(P_micro, thresh=0.0)
    G_macro = undirected_support_graph(Q, thresh=0.0)

    beta_micro = cycle_rank(G_micro)
    beta_macro = cycle_rank(G_macro)

    assert beta_macro == 0
    assert beta_micro > 0

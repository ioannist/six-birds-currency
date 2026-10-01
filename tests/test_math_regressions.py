"""Mathematical counterexamples and independent finite-state checks."""

from itertools import product

import numpy as np
import pytest
from scipy.special import xlogy

from currencymorphism.audits_cycles import cycle_affinities, edge_log_ratio
from currencymorphism.audits_pathkl import sigma_T_empirical, sigma_T_markov
from currencymorphism.lens import coarse_path, lumped_kernel, pushforward_dist
from currencymorphism.markov import stationary_dist
from currencymorphism.maxcal_single import (
    expected_cost,
    maxent_kernel,
    solve_lambda_for_budget,
)
from currencymorphism.mle_logit import fit_lambda
from currencymorphism.packaging import E_tau_f, idempotence_defect


def enumerated_reversal_kl(P, rho, horizon):
    """Independent path enumeration, including the singular probability mass."""
    rho = np.asarray(rho) / np.sum(rho)
    laws = {}
    for path in product(range(len(rho)), repeat=horizon + 1):
        mass = rho[path[0]]
        for i, j in zip(path[:-1], path[1:], strict=True):
            mass *= P[i, j]
        laws[path] = mass
    risk = sum(p for path, p in laws.items() if laws[path[::-1]] == 0)
    if risk > 0:
        return np.inf, risk
    return (
        sum(p * np.log(p / laws[path[::-1]]) for path, p in laws.items() if p > 0),
        risk,
    )


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
@pytest.mark.parametrize("case", ["positive", "one_way", "boundary", "periodic"])
def test_analytic_kl_and_singular_mass_match_enumerated_law(horizon, case):
    if case == "positive":
        P = np.array([[0.2, 0.8], [0.3, 0.7]])
        rho = [0.9, 0.1]
    elif case == "one_way":
        P = np.array([[0.5, 0.5], [0.0, 1.0]])
        rho = [0.5, 0.5]
    elif case == "boundary":
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        rho = [1.0, 0.0]
    else:
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        rho = [1.0, 0.0]
    expected, risk = enumerated_reversal_kl(P, rho, horizon)
    actual = sigma_T_markov(P, np.array(rho), horizon)
    assert actual.value == pytest.approx(expected, abs=1e-12)
    assert actual.infinite_risk_mass == pytest.approx(risk, abs=1e-12)


def test_missing_reverse_is_infinite_and_regularization_obeys_dpi():
    # Deleting unsupported terms would yield micro=0 < coarse=0.8*log(9).
    paths = np.array([[0, 1]] * 9 + [[2, 3]])
    coarse = coarse_path(paths, np.array([0, 1, 1, 0]))
    raw = sigma_T_empirical(paths, 1)
    assert np.isinf(raw.value)
    assert raw.infinite_risk_mass == pytest.approx(1.0)
    assert sigma_T_empirical(coarse, 1).value == pytest.approx(0.8 * np.log(9))
    micro = sigma_T_empirical(paths, 1, reverse_mix=0.01)
    macro = sigma_T_empirical(coarse, 1, reverse_mix=0.01)
    assert micro.value >= macro.value


@pytest.mark.parametrize("mix", [0.001, 0.01, 0.2])
def test_regularized_empirical_dpi_on_sparse_windows(mix):
    rng = np.random.default_rng(731)
    paths = rng.integers(0, 5, size=(12, 8))
    for horizon in [1, 2, 4]:
        micro = sigma_T_empirical(paths, horizon, reverse_mix=mix)
        for part in [np.zeros(5, dtype=int), np.array([0, 1, 0, 1, 2])]:
            coarse = sigma_T_empirical(
                coarse_path(paths, part), horizon, reverse_mix=mix
            )
            assert micro.value >= coarse.value - 1e-12


def test_empty_empirical_law_is_not_zero_asymmetry():
    with pytest.raises(ValueError, match="No windows"):
        sigma_T_empirical(np.array([0, 1]), 2)


def test_minimum_budget_has_limiting_uniform_minimizer_law():
    u = np.array([[1.0, 1.0, 3.0], [0.0, 2.0, 2.0]])
    lam, q, achieved = solve_lambda_for_budget(u, 0.5)
    assert np.isinf(lam)
    np.testing.assert_array_equal(q, [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    assert achieved == 0.5
    with pytest.raises(ValueError, match="below achievable minimum"):
        solve_lambda_for_budget(u, np.nextafter(0.5, 0.0))


def test_cost_units_do_not_cap_the_price():
    u = np.array([[0.0, 1e-9], [1e-9, 0.0]])
    lam, q, achieved = solve_lambda_for_budget(u, 2e-10)
    assert lam == pytest.approx(np.log(4) * 1e9, rel=1e-8)
    assert achieved == pytest.approx(2e-10, rel=1e-8)
    assert np.isfinite(q).all()


def test_slack_tolerance_does_not_override_an_active_bound():
    u = np.array([[0.0, 1.0]])
    lam, _, _ = solve_lambda_for_budget(u, 0.5 - 5e-11, tol=1e-10)
    assert lam > 0.0


def test_softmax_and_information_ignore_large_row_offsets():
    u = np.array([[1e12, 1e12 + 1], [1e12 + 1, 1e12]])
    np.testing.assert_allclose(maxent_kernel(u, 1e6), [[1, 0], [0, 1]])
    counts = np.array([[80, 20], [20, 80]])
    lam, se = fit_lambda(counts, u)
    assert lam == pytest.approx(np.log(4), rel=1e-9)
    assert se == pytest.approx(1 / np.sqrt(32), rel=1e-9)


def test_separated_mle_has_no_finite_maximizer():
    counts = np.array([[10, 0, 0], [0, 10, 0]])
    costs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    lam, se = fit_lambda(counts, costs)
    assert np.isinf(lam) and np.isinf(se)


def test_cost_identifiability_does_not_use_an_absolute_variance_cutoff():
    u = np.array([[0.0, 1e-9], [1e-9, 0.0]])
    lam, se = fit_lambda(np.array([[80, 20], [20, 80]]), u)
    assert lam == pytest.approx(np.log(4) * 1e9, rel=1e-8)
    assert np.isfinite(se)
    assert se == pytest.approx(1e9 / np.sqrt(32), rel=1e-8)


def test_weighted_gibbs_optimality_and_shadow_support():
    u = np.array([[0.0, 1.0, 3.0], [4.0, 1.0, 2.0]])
    mu = np.array([0.2, 0.8])
    q = maxent_kernel(u, 0.7)
    budget = expected_cost(q, u, mu)
    lam, q_solved, _ = solve_lambda_for_budget(u, budget, mu=mu)
    assert lam == pytest.approx(0.7)
    np.testing.assert_allclose(q_solved, q)
    entropy_q = -np.sum(mu[:, None] * xlogy(q, q))
    rng = np.random.default_rng(22)
    for _ in range(30):
        p = rng.dirichlet(np.ones(3), size=2)
        entropy_p = -np.sum(mu[:, None] * xlogy(p, p))
        c_p = expected_cost(p, u, mu)
        divergence = np.sum(mu[:, None] * p * np.log(p / q))
        assert entropy_q - entropy_p + lam * (c_p - budget) == pytest.approx(divergence)
        assert entropy_p <= entropy_q + lam * (c_p - budget) + 1e-12


def test_tiny_positive_edges_retain_cycle_affinity():
    P = np.array(
        [
            [1 - 3e-16, 2e-16, 1e-16],
            [1e-16, 1 - 3e-16, 2e-16],
            [2e-16, 1e-16, 1 - 3e-16],
        ]
    )
    assert cycle_affinities(P, [[0, 1, 2]])[0] == pytest.approx(3 * np.log(2))
    assert edge_log_ratio(P)[0, 1] > 0


@pytest.mark.parametrize("cycle", [[0, 1], [0, 1, 0], [0, 1, 2], [0, 1, -1]])
def test_cycle_audit_rejects_fake_cycles(cycle):
    P = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        cycle_affinities(P, [cycle])


def test_lumping_preserves_stationarity_and_rejects_false_witness():
    P = np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.6, 0.1, 0.3]])
    pi = stationary_dist(P)
    part = np.array([0, 0, 1])
    Q = lumped_kernel(P, part, pi)
    macro = pushforward_dist(pi, part)
    np.testing.assert_allclose(macro @ Q, macro, atol=1e-12)
    with pytest.raises(ValueError, match="stationary"):
        lumped_kernel(P, part, np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="integer dtype"):
        coarse_path(np.array([0, 1]), np.array([0.1, 0.9, 1.0]))


def test_stationary_packaging_exit_bound_and_zero_time_projection():
    P = np.array([[0.8, 0.19, 0.01], [0.2, 0.79, 0.01], [0.01, 0.01, 0.98]])
    part = np.array([0, 0, 1])
    pi = stationary_dist(P)
    max_exit = np.max(np.sum(P * (part[:, None] != part[None, :]), axis=1))
    for mu in [np.array([1.0, 0.0, 0.0]), np.array([0.2, 0.3, 0.5])]:
        assert idempotence_defect(mu, P, 0, part, pi_micro=pi) < 1e-12
        for tau in [1, 3, 5]:
            assert (
                idempotence_defect(mu, P, tau, part, pi_micro=pi)
                <= 2 * tau * max_exit + 1e-12
            )
    with pytest.raises(ValueError, match="stationary"):
        E_tau_f(np.ones(3), P, 1, part, pi_micro=np.ones(3))


def test_stationary_iteration_handles_nonuniform_periodic_chain():
    P = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    pi = stationary_dist(P, max_iter=100)
    np.testing.assert_allclose(pi, [0.25, 0.5, 0.25], atol=1e-12)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01])
def test_audits_reject_nonprobability_kernels(bad):
    P = np.array([[bad, 1.0 - bad], [0.5, 0.5]])
    with pytest.raises(ValueError):
        sigma_T_markov(P, np.array([0.5, 0.5]), 1)


def test_coarsening_can_create_cycle_rank_even_for_reversible_tree():
    import networkx as nx

    from currencymorphism.audits_cycles import cycle_rank, undirected_support_graph
    from currencymorphism.generators import reversible_kernel

    P = reversible_kernel(nx.path_graph(4))
    Q = lumped_kernel(P, np.array([0, 1, 2, 0]))
    assert cycle_rank(undirected_support_graph(P)) == 0
    assert cycle_rank(undirected_support_graph(Q)) == 1
    assert cycle_affinities(Q, [[0, 1, 2]])[0] == pytest.approx(0.0, abs=1e-12)

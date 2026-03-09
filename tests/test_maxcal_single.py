import numpy as np

from currencymorphism.maxcal_single import (
    expected_cost,
    maxent_kernel,
    solve_lambda_for_budget,
)


def test_budget_monotonicity_decreasing_budget_increases_lambda() -> None:
    rng = np.random.default_rng(123)
    n = 10
    u = rng.random((n, n)) + 0.1

    q0 = maxent_kernel(u, 0.0)
    cost0 = expected_cost(q0, u)
    cost_min = float(np.mean(np.min(u, axis=1)))

    gap = cost0 - cost_min
    low = cost_min + 0.2 * gap
    high = cost0 - 0.2 * gap
    budgets_desc = np.linspace(high, low, 5)

    lambdas: list[float] = []
    for b in budgets_desc:
        lam, _, achieved = solve_lambda_for_budget(u, b, tol=1e-10)
        lambdas.append(lam)
        assert abs(achieved - b) < 1e-7

    diffs = np.diff(np.asarray(lambdas))
    assert np.all(diffs >= -1e-10)


def test_slack_budget_returns_zero_lambda() -> None:
    rng = np.random.default_rng(456)
    u = rng.random((8, 8)) + 0.1

    q0 = maxent_kernel(u, 0.0)
    cost0 = expected_cost(q0, u)

    b_slack = cost0 + 0.5
    lam, _, achieved = solve_lambda_for_budget(u, b_slack, tol=1e-10)

    assert lam == 0.0
    assert abs(achieved - cost0) < 1e-12

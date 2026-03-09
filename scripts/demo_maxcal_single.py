from __future__ import annotations

import numpy as np

from currencymorphism.maxcal_single import (
    expected_cost,
    maxent_kernel,
    solve_lambda_for_budget,
)


def main() -> None:
    rng = np.random.default_rng(20260305)
    n = 10
    u = rng.random((n, n)) + 0.1

    q0 = maxent_kernel(u, 0.0)
    cost0 = expected_cost(q0, u)
    cost_min = float(np.mean(np.min(u, axis=1)))

    gap = cost0 - cost_min
    high = cost0 - 0.05 * gap
    low = cost_min + 0.05 * gap
    budgets = np.linspace(high, low, 5)

    for b in budgets:
        lam, _, achieved = solve_lambda_for_budget(u, float(b), tol=1e-10)
        print(f"b={b:.12e} lam={lam:.12e} achieved={achieved:.12e}")


if __name__ == "__main__":
    main()

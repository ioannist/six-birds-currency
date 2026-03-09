import numpy as np

from currencymorphism.markov import normalize_rows, stationary_dist


def main() -> None:
    rng = np.random.default_rng(20260305)
    n = 10
    P = normalize_rows(rng.random((n, n)), eps=1e-3)
    pi = stationary_dist(P, method="power", tol=1e-14, max_iter=1_000_000)
    err = np.linalg.norm(pi @ P - pi, ord=1)
    print(f"pi_error={err:.12e}")


if __name__ == "__main__":
    main()

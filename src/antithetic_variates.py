# src/antithetic_variates.py

import numpy as np

from src.models import compute_estimator_statistics, asian_arithmetic_call_payoff, european_call_payoff


def generate_antithetic_normals(n_pairs, n_steps=None, seed=None):
    """
    Generate paired normal variates Z and -Z.
    """
    rng = np.random.default_rng(seed)
    if n_steps is None:
        Z = rng.standard_normal(size=n_pairs)
        return Z, -Z
    else:
        Z = rng.standard_normal(size=(n_pairs, n_steps))
        return Z, -Z


def mc_price_european_call_antithetic(params, n_pairs, seed=None):
    """
    Price a European call using antithetic variates.

    Inputs:
        params: option parameter dictionary
        n_pairs: number of antithetic pairs
        seed: random seed

    Returns:
        price, standard_error, ci_low, ci_high
    """
    Z, Z_antithetic = generate_antithetic_normals(n_pairs, seed=seed)

    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ST_antithetic = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_antithetic)

    payoffs = (european_call_payoff(ST, params["K"]) + european_call_payoff(ST_antithetic, params["K"])) / 2

    discounted_payoffs = (np.exp(-params["r"] * params["T"]) * payoffs)

    return compute_estimator_statistics(discounted_payoffs)


def mc_price_asian_call_antithetic(params, n_pairs, n_steps, seed=None):
    """
    Price an arithmetic-average Asian call using antithetic variates.

    Inputs:
        params: option parameter dictionary
        n_pairs: number of antithetic pairs
        n_steps: number of time steps in each path
        seed: random seed

    Returns:
        price, standard_error, ci_low, ci_high
    """
    Z, Z_antithetic = generate_antithetic_normals(n_pairs=n_pairs, n_steps=n_steps, seed=seed)

    S0 = params["S0"]
    r = params["r"]
    K = params["K"]
    sigma = params["sigma"]
    T = params["T"]

    dt = T / n_steps

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    increments_antithetic = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_antithetic

    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_pairs, 1)), log_paths])
    log_paths_antithetic = np.cumsum(increments_antithetic, axis=1)
    log_paths_antithetic = np.hstack([np.zeros((n_pairs, 1)), log_paths_antithetic])

    paths = S0 * np.exp(log_paths)
    paths_antithetic = S0 * np.exp(log_paths_antithetic)

    payoffs = (asian_arithmetic_call_payoff(paths, K) + asian_arithmetic_call_payoff(paths_antithetic, K)) / 2

    discounted_payoffs = (np.exp(-r * T) * payoffs)

    return compute_estimator_statistics(discounted_payoffs)








# src/multilevel_monte_carlo.py

import numpy as np

from src.models import european_call_payoff, asian_arithmetic_call_payoff


def simulate_coarse_fine_terminal(
    params,
    level,
    n_paths,
    seed=None,
):
    """
    Simulate coupled coarse and fine GBM terminal values for one MLMC level.

    Inputs:
        params: option parameter dictionary

        level: MLMC level

        n_paths: number of coupled path pairs

        seed: random seed

    Returns:
        fine_terminal: ndarray of shape (n_paths,)

        coarse_terminal: ndarray of shape (n_paths,)
    """
    rng = np.random.default_rng(seed)

    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    n_fine = 2 ** level
    n_coarse = max(1, n_fine // 2)

    dt_fine = T / n_fine
    dt_coarse = T / n_coarse

    Z_fine = rng.standard_normal((n_paths, n_fine))
    if level == 0:
        Z_coarse = Z_fine
    else:
        Z_coarse = (Z_fine[:, 0::2] + Z_fine[:, 1::2]) / np.sqrt(2)

    fine_increments = (r - 0.5 * sigma**2) * dt_fine + sigma * np.sqrt(dt_fine) * Z_fine
    coarse_increments = (r - 0.5 * sigma**2) * dt_coarse + sigma * np.sqrt(dt_coarse) * Z_coarse

    fine_log_terminal = np.sum(fine_increments, axis=1)
    coarse_log_terminal = np.sum(coarse_increments, axis=1)

    fine_terminal = S0 * np.exp(fine_log_terminal)
    coarse_terminal = S0 * np.exp(coarse_log_terminal)

    return fine_terminal, coarse_terminal


def simulate_coarse_fine_paths(
    params,
    level,
    n_paths,
    seed=None,
):
    """
    Simulate coupled coarse and fine GBM paths.

    Inputs:
        params: option parameter dictionary

        level: MLMC level

        n_paths: number of coupled path pairs

        seed: random seed

    Returns:
        fine_paths: ndarray

        coarse_paths: ndarray
    """
    rng = np.random.default_rng(seed)

    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    n_fine = 2 ** level
    n_coarse = max(1, n_fine // 2)

    dt_fine = T / n_fine
    dt_coarse = T / n_coarse

    Z_fine = rng.standard_normal((n_paths, n_fine))
    if level == 0:
        Z_coarse = Z_fine
    else:
        Z_coarse = (Z_fine[:, 0::2] + Z_fine[:, 1::2]) / np.sqrt(2)

    fine_increments = (r - 0.5 * sigma**2) * dt_fine + sigma * np.sqrt(dt_fine) * Z_fine
    coarse_increments = (r - 0.5 * sigma**2) * dt_coarse + sigma * np.sqrt(dt_coarse) * Z_coarse

    fine_log_paths = np.cumsum(fine_increments, axis=1)
    coarse_log_paths = np.cumsum(coarse_increments, axis=1)

    fine_log_paths = np.hstack([np.zeros((n_paths, 1)), fine_log_paths])
    coarse_log_paths = np.hstack([np.zeros((n_paths, 1)), coarse_log_paths])

    fine_paths = S0 * np.exp(fine_log_paths)
    coarse_paths = S0 * np.exp(coarse_log_paths)

    return fine_paths, coarse_paths


def european_level_correction(
    fine_terminal,
    coarse_terminal,
    params,
    level,
):
    """
    Compute one MLMC correction level for the European call.

    Inputs:
        fine_terminal: fine terminal prices

        coarse_terminal: coarse terminal prices

        params: option parameters

        level: MLMC level

    Returns:
        discounted payoff differences
    """
    K = params["K"]
    r = params["r"]
    T = params["T"]

    fine_payoff = european_call_payoff(fine_terminal, K)
    if level == 0:
        return np.exp(-r * T) * fine_payoff
    
    coarse_payoff = european_call_payoff(coarse_terminal, K)

    return np.exp(-r * T) * (fine_payoff - coarse_payoff)


def asian_level_correction(
    fine_paths,
    coarse_paths,
    params,
    level,
):
    """
    Compute one MLMC correction level for the arithmetic Asian call.

    Inputs:
        fine_paths: fine GBM paths

        coarse_paths: coarse GBM paths

        params: option parameters

        level: MLMC level

    Returns:
        discounted payoff differences
    """
    K = params["K"]
    r = params["r"]
    T = params["T"]

    fine_payoff = asian_arithmetic_call_payoff(fine_paths, K)
    if level == 0:
        return np.exp(-r * T) * fine_payoff

    coarse_payoff = asian_arithmetic_call_payoff(coarse_paths, K)

    return np.exp(-r * T) * (fine_payoff - coarse_payoff)


def calculate_mlmc_levels(
    n_paths,
):
    """
    Calculate the sample counts per level.

    Inputs:
        n_paths: number of coupled path pairs

    Returns:
        level_samples: list of (level, n_level_paths) tuples
    """
    level_samples = []

    max_level = int(np.log2(n_paths))

    for level in range(max_level + 1):
        level_paths = max(2, n_paths // (2 ** level))
        level_samples.append((level, level_paths))

    return level_samples


def mlmc_price_european_call(
    params,
    n_paths,
    seed=None,
):
    """
    Price a European call using Multilevel Monte Carlo.

    Inputs:
        params:
            option parameters

        n_paths:
            base number of paths

        seed:
            random seed

    Returns:
        price
        std_error
        ci_low
        ci_high
    """
    levels = calculate_mlmc_levels(n_paths)

    level_estimates = []
    level_std_errors = []

    for level, level_paths in levels:
        level_seed = None if seed is None else seed + level

        fine_terminal, coarse_terminal = simulate_coarse_fine_terminal(
            params=params,
            level=level,
            n_paths=level_paths,
            seed=level_seed,
        )

        discounted_payoffs = european_level_correction(
            fine_terminal,
            coarse_terminal,
            params,
            level,
        )

        level_estimates.append(np.mean(discounted_payoffs))

        level_std_error = (
            np.std(discounted_payoffs, ddof=1)
            / np.sqrt(level_paths)
        )

        level_std_errors.append(level_std_error)

    price = np.sum(level_estimates)
    std_error = np.sqrt((np.sum(np.square(level_std_errors))))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error

    return price, std_error, ci_low, ci_high


def mlmc_price_asian_call(
    params,
    n_paths,
    seed=None,
):
    """
    Price an arithmetic Asian call using Multilevel Monte Carlo.

    Inputs:
        params:
            option parameters

        n_paths:
            base number of paths

        seed:
            random seed

    Returns:
        price
        std_error
        ci_low
        ci_high
    """
    levels = calculate_mlmc_levels(n_paths)

    level_estimates = []
    level_std_errors = []

    for level, level_paths in levels:
        level_seed = None if seed is None else seed + level

        fine_paths, coarse_paths = simulate_coarse_fine_paths(
            params=params,
            level=level,
            n_paths=level_paths,
            seed=level_seed,
        )

        discounted_payoffs = asian_level_correction(
            fine_paths,
            coarse_paths,
            params,
            level,
        )

        level_estimates.append(np.mean(discounted_payoffs))

        level_std_error = (
            np.std(discounted_payoffs, ddof=1)
            / np.sqrt(level_paths)
        )

        level_std_errors.append(level_std_error)

    price = np.sum(level_estimates)
    std_error = np.sqrt((np.sum(np.square(level_std_errors))))
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error

    return price, std_error, ci_low, ci_high

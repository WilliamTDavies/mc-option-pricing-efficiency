# src/quasi_monte_carlo.py

import numpy as np

from scipy.stats import qmc, norm

from src.models import (
    compute_estimator_statistics,
    european_call_payoff,
    asian_arithmetic_call_payoff,
)


def generate_sobol_normals(
    n_paths,
    dimension,
    scramble=True,
    seed=None,
):
    """
    Generate Sobol quasi-random numbers and transform them to standard normals.

    Inputs:
        n_paths: number of quasi-random samples to generate
        dimension: dimension of the Sobol sequence (number of random variables)
        scramble: whether to apply scrambling to the Sobol sequence
        seed: random seed for reproducibility
    Returns:
        normal_samples: array of shape (n_paths, dimension) containing standard normal samples
    """
    sampler = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
    m = int(np.log2(n_paths))
    sobol_samples = sampler.random_base2(m=m)
    normal_samples = norm.ppf(sobol_samples)
    return normal_samples


def qmc_price_european_call(
    params,
    n_paths,
    seed=None,
):
    """
    Price a European call option using quasi-Monte Carlo with Sobol sequences.

    Inputs:
        params: dictionary containing option parameters (S0, K, r, sigma, T)
        n_paths: number of quasi-random paths to simulate
        seed: random seed for reproducibility
    Returns:
        price: estimated price of the European call option
    """
    S0 = params["S0"]
    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    Z = generate_sobol_normals(n_paths, dimension=1, scramble=True, seed=seed).ravel()

    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs = european_call_payoff(ST, K)

    discounted_payoffs = np.exp(-r * T) * payoffs

    return compute_estimator_statistics(discounted_payoffs)


def qmc_price_asian_call(
    params,
    n_paths,
    n_steps,
    seed=None,
):
    """
    Price an arithmetic-average Asian call option using quasi-Monte Carlo with Sobol sequences.

    Inputs:
        params: dictionary containing option parameters (S0, K, r, sigma, T)
        n_paths: number of quasi-random paths to simulate
        n_steps: number of time steps for the path simulation
        seed: random seed for reproducibility
    Returns:
        price: estimated price of the Asian call option
    """
    S0 = params["S0"]
    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    dt = T / n_steps

    Z = generate_sobol_normals(n_paths, dimension=n_steps, scramble=True, seed=seed)

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    paths = S0 * np.exp(log_paths)

    payoffs = asian_arithmetic_call_payoff(paths, K)

    discounted_payoffs = np.exp(-r * T) * payoffs

    return compute_estimator_statistics(discounted_payoffs)
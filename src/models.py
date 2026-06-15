# src/models.py

import numpy as np
from scipy.stats import norm


def make_option_params(S0, K, r, sigma, T):
    """
    Create a plain parameter dictionary for an option model.
    """
    return {
        "S0": float(S0),
        "K": float(K),
        "r": float(r),
        "sigma": float(sigma),
        "T": float(T),
    }


def compute_estimator_statistics(discounted_payoffs):
    """
    Compute Monte Carlo estimator statistics.
    Returns price, standard error, ci_low, ci_high.
    """
    n = len(discounted_payoffs)

    price = float(np.mean(discounted_payoffs))

    if n > 1:
        std_error = float(
            np.std(discounted_payoffs, ddof=1) / np.sqrt(n)
        )
    else:
        std_error = 0.0

    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error

    return price, std_error, ci_low, ci_high


def black_scholes_call(params):
    """
    Closed-form Black-Scholes price for a European call option.
    """
    S0 = params["S0"]
    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    if T <= 0:
        return max(S0 - K, 0.0)

    if sigma <= 0:
        ST = S0 * np.exp(r * T)
        return np.exp(-r * T) * max(ST - K, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def simulate_gbm_terminal(params, n_paths, seed=None):
    """
    Simulate terminal GBM values S_T under the risk-neutral measure.
    Returns an array of shape (n_paths,).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)

    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z

    return S0 * np.exp(drift + diffusion)


def simulate_gbm_paths(params, n_paths, n_steps, seed=None):
    """
    Simulate full GBM paths.
    Returns array of shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)

    S0 = params["S0"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    z = rng.standard_normal((n_paths, n_steps))
    increments = drift + vol * z

    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    return S0 * np.exp(log_paths)


def european_call_payoff(ST, K):
    """
    European call payoff max(ST - K, 0).
    """
    return np.maximum(ST - K, 0.0)


def asian_arithmetic_call_payoff(paths, K):
    """
    Arithmetic-average Asian call payoff using the full simulated path.
    Includes S_0 in the average.
    """
    avg_price = np.mean(paths, axis=1)
    return np.maximum(avg_price - K, 0.0)


def mc_price_european_call(params, n_paths, seed=None):
    """
    Monte Carlo price for a European call.
    """
    ST = simulate_gbm_terminal(
        params=params,
        n_paths=n_paths,
        seed=seed,
    )

    payoffs = european_call_payoff(
        ST,
        params["K"],
    )

    discounted_payoffs = (
        np.exp(-params["r"] * params["T"])
        * payoffs
    )

    return compute_estimator_statistics(
        discounted_payoffs
    )


def mc_price_asian_call(params, n_paths, n_steps, seed=None):
    """
    Monte Carlo price for an arithmetic-average Asian call.
    """
    paths = simulate_gbm_paths(
        params=params,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    payoffs = asian_arithmetic_call_payoff(
        paths,
        params["K"],
    )

    discounted_payoffs = (
        np.exp(-params["r"] * params["T"])
        * payoffs
    )

    return compute_estimator_statistics(
        discounted_payoffs
    )
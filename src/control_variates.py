# src/control_variates.py

import numpy as np
from scipy.stats import norm

from src.models import (
    compute_estimator_statistics,
    simulate_gbm_terminal,
    simulate_gbm_paths,
    asian_arithmetic_call_payoff,
    european_call_payoff,
)


def control_variate_adjustment(
    target_payoffs,
    control_payoffs,
    control_expectation,
):
    """
    Adjust target payoffs using the control variate technique.

    Inputs:
        target_payoffs: array of discounted target payoffs
        control_payoffs: array of discounted control payoffs
        control_expectation: known expected value of the discounted control payoff

    Returns:
        adjusted_payoffs: control-variate-adjusted discounted payoffs
    """
    target_payoffs = np.asarray(target_payoffs, dtype=float)
    control_payoffs = np.asarray(control_payoffs, dtype=float)

    cov = np.cov(
        target_payoffs,
        control_payoffs,
        ddof=1,
    )[0, 1]

    var_control = np.var(
        control_payoffs,
        ddof=1,
    )

    if var_control < 1e-12: # Avoid floating-point issues when control payoffs have very low variance
        return target_payoffs

    beta = cov / var_control
    adjusted_payoffs = target_payoffs - beta * (control_payoffs - control_expectation)
    return adjusted_payoffs


def geometric_asian_call_price(params, n_steps):
    """
    Closed-form price of the geometric Asian call option.

    This uses the same averaging convention as the Monte Carlo code:
    paths include S_0 plus n_steps future observations, so there are
    n_steps + 1 points in the geometric average.

    Inputs:
        params: option parameter dictionary
        n_steps: number of time steps / intervals

    Returns:
        discounted geometric Asian call price
    """
    S0 = params["S0"]
    K = params["K"]
    r = params["r"]
    sigma = params["sigma"]
    T = params["T"]

    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    # Log of the geometric average G is normal with mean and variance given by:
    geometric_log_mean = np.log(S0) + (r - 0.5 * sigma**2) * T / 2
    geometric_log_variance = (
        sigma**2 * T * (2 * n_steps + 1) / (6 * (n_steps + 1))
    )

    sigma_g = np.sqrt(geometric_log_variance)

    d1 = (
        geometric_log_mean
        - np.log(K)
        + geometric_log_variance
    ) / sigma_g
    d2 = d1 - sigma_g

    undiscounted_call = (
        np.exp(geometric_log_mean + 0.5 * geometric_log_variance) * norm.cdf(d1)
        - K * norm.cdf(d2)
    )

    discounted_call = np.exp(-r * T) * undiscounted_call
    return discounted_call


def mc_price_european_call_control_variate(
    params,
    n_paths,
    seed=None,
):
    """
    Price a European call using control variates.

    Control variable:
        discounted terminal stock price e^{-rT} S_T

    Known expectation:
        E[e^{-rT} S_T] = S_0

    Returns:
        price, standard_error, ci_low, ci_high
    """
    S0 = params["S0"]
    r = params["r"]
    T = params["T"]

    ST = simulate_gbm_terminal(params, n_paths, seed)

    discount = np.exp(-r * T)

    target_payoffs = discount * european_call_payoff(ST, params["K"])
    control_payoffs = discount * ST
    control_expectation = S0

    adjusted_payoffs = control_variate_adjustment(
        target_payoffs,
        control_payoffs,
        control_expectation,
    )

    return compute_estimator_statistics(adjusted_payoffs)


def mc_price_asian_call_control_variate(
    params,
    n_paths,
    n_steps,
    seed=None,
):
    """
    Price an arithmetic-average Asian call using control variates.

    Control variable:
        discounted geometric Asian call payoff

    Known expectation:
        geometric_asian_call_price(params, n_steps)

    Returns:
        price, standard_error, ci_low, ci_high
    """
    r = params["r"]
    T = params["T"]
    K = params["K"]

    paths = simulate_gbm_paths(
        params=params,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    discount = np.exp(-r * T)

    target_payoffs = discount * asian_arithmetic_call_payoff(paths, K)

    geometric_average = np.exp(np.mean(np.log(paths), axis=1))
    control_payoffs = discount * np.maximum(geometric_average - K, 0.0)

    control_expectation = geometric_asian_call_price(params, n_steps)

    adjusted_payoffs = control_variate_adjustment(
        target_payoffs,
        control_payoffs,
        control_expectation,
    )

    return compute_estimator_statistics(adjusted_payoffs)
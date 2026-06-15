# src/benchmarks.py

from src.config import ASIAN_N_STEPS, REFERENCE_PATHS, REFERENCE_SEED
from src.models import black_scholes_call, mc_price_asian_call


def european_call_reference(params):
    """
    Reference price for the European call.
    """
    return black_scholes_call(params)


def asian_call_reference(
    params,
    n_paths=REFERENCE_PATHS,
    n_steps=ASIAN_N_STEPS,
    seed=REFERENCE_SEED,
):
    """
    High-precision Monte Carlo reference price for the Asian call.
    This is not exact, but it is used as the benchmark value.
    """
    price, _, _, _ = mc_price_asian_call(
        params=params,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    return price
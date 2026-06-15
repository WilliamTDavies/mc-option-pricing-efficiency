# src/experiments.py

import time
from pathlib import Path

import pandas as pd

from src.benchmarks import asian_call_reference, european_call_reference
from src.config import (
    ASIAN_N_STEPS,
    ASIAN_SEED_OFFSET,
    BASE_SEED,
    EUROPEAN_SEED_OFFSET,
    N_REPLICATIONS,
    OPTION_PARAMS,
    PATH_GRID,
    RESULTS_DIR,
)
from src.models import make_option_params, mc_price_asian_call, mc_price_european_call


def get_default_params():
    """
    Default option parameters used throughout the project.
    """
    return make_option_params(**OPTION_PARAMS)


def build_seed(base_seed, offset, n_paths, rep):
    """
    Deterministic seed construction for reproducible repetitions.
    """
    return base_seed + offset + 1000 * rep + n_paths


def run_european_sweep(
    params,
    path_grid=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated Monte Carlo experiments for the European call.
    Returns a summary DataFrame.
    """
    if path_grid is None:
        path_grid = PATH_GRID
    if n_replications is None:
        n_replications = N_REPLICATIONS

    reference = european_call_reference(params)
    rows = []

    for n_paths in path_grid:
        for rep in range(n_replications):
            rep_seed = build_seed(seed, EUROPEAN_SEED_OFFSET, n_paths, rep)

            start = time.perf_counter()
            price, std_err, ci_low, ci_high = mc_price_european_call(
                params=params,
                n_paths=n_paths,
                seed=rep_seed,
            )
            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "european",
                    "n_paths": n_paths,
                    "replication": rep,
                    "price": price,
                    "reference_price": reference,
                    "abs_error": abs(price - reference),
                    "signed_error": price - reference,
                    "std_error": std_err,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_width": ci_high - ci_low,
                    "runtime_sec": runtime,
                }
            )

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["option_type", "n_paths"], as_index=False)
        .agg(
            reference_price=("reference_price", "first"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            mean_abs_error=("abs_error", "mean"),
            mean_signed_error=("signed_error", "mean"),
            mean_std_error=("std_error", "mean"),
            mean_ci_width=("ci_width", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .sort_values("n_paths")
        .reset_index(drop=True)
    )

    return summary


def run_asian_sweep(
    params,
    path_grid=None,
    n_steps=None,
    n_replications=None,
    seed=BASE_SEED,
    reference_n_paths=None,
    reference_seed=None,
):
    """
    Run repeated Monte Carlo experiments for the Asian call.
    Returns a summary DataFrame.
    """
    if path_grid is None:
        path_grid = PATH_GRID
    if n_steps is None:
        n_steps = ASIAN_N_STEPS
    if n_replications is None:
        n_replications = N_REPLICATIONS

    reference = asian_call_reference(
        params=params,
        n_paths=reference_n_paths if reference_n_paths is not None else 200000,
        n_steps=n_steps,
        seed=reference_seed if reference_seed is not None else 999999,
    )

    rows = []

    for n_paths in path_grid:
        for rep in range(n_replications):
            rep_seed = build_seed(seed, ASIAN_SEED_OFFSET, n_paths, rep)

            start = time.perf_counter()
            price, std_err, ci_low, ci_high = mc_price_asian_call(
                params=params,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=rep_seed,
            )
            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "asian",
                    "n_steps": n_steps,
                    "n_paths": n_paths,
                    "replication": rep,
                    "price": price,
                    "reference_price": reference,
                    "abs_error": abs(price - reference),
                    "signed_error": price - reference,
                    "std_error": std_err,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_width": ci_high - ci_low,
                    "runtime_sec": runtime,
                }
            )

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["option_type", "n_steps", "n_paths"], as_index=False)
        .agg(
            reference_price=("reference_price", "first"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            mean_abs_error=("abs_error", "mean"),
            mean_signed_error=("signed_error", "mean"),
            mean_std_error=("std_error", "mean"),
            mean_ci_width=("ci_width", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .sort_values(["n_steps", "n_paths"])
        .reset_index(drop=True)
    )

    return summary


def run_all_experiments():
    """
    Run the baseline experiments and return all summary tables.
    """
    params = get_default_params()

    european = run_european_sweep(params=params)
    asian = run_asian_sweep(params=params)

    return {
        "european": european,
        "asian": asian,
    }


def save_results(results, results_dir=RESULTS_DIR):
    """
    Save summary results to CSV files.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    results["european"].to_csv(results_path / "european_summary.csv", index=False)
    results["asian"].to_csv(results_path / "asian_summary.csv", index=False)
# src/experiments.py

import time
import pandas as pd
import subprocess

from pathlib import Path

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
from src.antithetic_variates import mc_price_european_call_antithetic, mc_price_asian_call_antithetic
from src.control_variates import mc_price_european_call_control_variate, mc_price_asian_call_control_variate
from src.quasi_monte_carlo import qmc_price_european_call, qmc_price_asian_call
from src.multilevel_monte_carlo import mlmc_price_european_call, mlmc_price_asian_call


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
        n_steps=n_steps,
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


def run_european_antithetic_sweep(
    params,
    path_grid=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated Monte Carlo experiments for the European call with antithetic variates.

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
            price, std_err, ci_low, ci_high = mc_price_european_call_antithetic(
                params=params,
                n_pairs=n_paths // 2,
                seed=rep_seed,
            )
            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "european_antithetic",
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


def run_asian_antithetic_sweep(
    params,
    path_grid=None,
    n_steps=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated Monte Carlo experiments for the Asian call with antithetic variates.

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
        n_steps=n_steps,
    )

    rows = []

    for n_paths in path_grid:
        for rep in range(n_replications):
            rep_seed = build_seed(seed, ASIAN_SEED_OFFSET, n_paths, rep)

            start = time.perf_counter()
            price, std_err, ci_low, ci_high = mc_price_asian_call_antithetic(
                params=params,
                n_pairs=n_paths // 2,
                n_steps=n_steps,
                seed=rep_seed,
            )
            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "asian_antithetic",
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


def run_european_control_variate_sweep(
    params,
    path_grid=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated Monte Carlo experiments for the European call with control variates.

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
            rep_seed = build_seed(
                seed,
                EUROPEAN_SEED_OFFSET,
                n_paths,
                rep,
            )

            start = time.perf_counter()

            price, std_err, ci_low, ci_high = (
                mc_price_european_call_control_variate(
                    params=params,
                    n_paths=n_paths,
                    seed=rep_seed,
                )
            )

            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "european_control",
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
        df.groupby(
            ["option_type", "n_paths"],
            as_index=False,
        )
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


def run_asian_control_variate_sweep(
    params,
    path_grid=None,
    n_steps=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated Monte Carlo experiments for the Asian call with control variates.

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
        n_steps=n_steps,
    )

    rows = []

    for n_paths in path_grid:
        for rep in range(n_replications):
            rep_seed = build_seed(
                seed,
                ASIAN_SEED_OFFSET,
                n_paths,
                rep,
            )

            start = time.perf_counter()

            price, std_err, ci_low, ci_high = (
                mc_price_asian_call_control_variate(
                    params=params,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    seed=rep_seed,
                )
            )

            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "asian_control",
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
        df.groupby(
            ["option_type", "n_steps", "n_paths"],
            as_index=False,
        )
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


def run_european_quasi_monte_carlo_sweep(
    params,
    path_grid=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated quasi-Monte Carlo experiments for the European call using Sobol sequences.

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
            rep_seed = build_seed(
                seed,
                EUROPEAN_SEED_OFFSET,
                n_paths,
                rep,
            )

            start = time.perf_counter()

            price, std_err, ci_low, ci_high = (
                qmc_price_european_call(
                    params=params,
                    n_paths=n_paths,
                    seed=rep_seed,
                )
            )

            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "european_qmc",
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
        df.groupby(
            ["option_type", "n_paths"],
            as_index=False,
        )
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


def run_asian_quasi_monte_carlo_sweep(
    params,
    path_grid=None,
    n_steps=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated quasi-Monte Carlo experiments for the Asian call using Sobol sequences.

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
        n_steps=n_steps,
    )

    rows = []

    for n_paths in path_grid:
        for rep in range(n_replications):
            rep_seed = build_seed(
                seed,
                ASIAN_SEED_OFFSET,
                n_paths,
                rep,
            )

            start = time.perf_counter()

            price, std_err, ci_low, ci_high = (
                qmc_price_asian_call(
                    params=params,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    seed=rep_seed,
                )
            )

            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "asian_qmc",
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
        df.groupby(
            ["option_type", "n_steps", "n_paths"],
            as_index=False,
        )
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


def run_european_multilevel_sweep(
    params,
    path_grid=None,
    n_replications=None,
    seed=BASE_SEED,
):
    """
    Run repeated quasi-Monte Carlo experiments for the European call using multilevel Monte Carlo.

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
            rep_seed = build_seed(
                seed,
                EUROPEAN_SEED_OFFSET,
                n_paths,
                rep,
            )

            start = time.perf_counter()

            price, std_err, ci_low, ci_high = (
                mlmc_price_european_call(
                    params=params,
                    n_paths=n_paths,
                    seed=rep_seed,
                )
            )

            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "european_mlmc",
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
        df.groupby(
            ["option_type", "n_paths"],
            as_index=False,
        )
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


def run_asian_multilevel_sweep(
    params,
    path_grid=None,
    n_replications=None,
    n_steps = ASIAN_N_STEPS,
    seed=BASE_SEED,
):
    """
    Run repeated quasi-Monte Carlo experiments for the Asian call using multilevel Monte Carlo.

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
        n_steps=n_steps,
    )

    rows = []

    for n_paths in path_grid:
        for rep in range(n_replications):
            rep_seed = build_seed(
                seed,
                ASIAN_SEED_OFFSET,
                n_paths,
                rep,
            )

            start = time.perf_counter()

            price, std_err, ci_low, ci_high = (
                mlmc_price_asian_call(
                    params=params,
                    n_paths=n_paths,
                    seed=rep_seed,
                )
            )

            runtime = time.perf_counter() - start

            rows.append(
                {
                    "option_type": "asian_mlmc",
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
        df.groupby(
            ["option_type", "n_steps", "n_paths"],
            as_index=False,
        )
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


def run_cpp_benchmarks():
    """
    Compile and execute the C++ benchmark program.
    """
    cpp_dir = Path("cpp")

    cpp_file = cpp_dir / "option_pricer.cpp"

    exe = cpp_dir / "option_pricer.exe"

    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            str(cpp_file),
            "-o",
            str(exe),
        ],
        check=True,
    )

    subprocess.run(
        [str(exe)],
        check=True,
    )


def run_all_experiments():
    """
    Run all experiments and return summary tables.
    """
    params = get_default_params()

    experiments = {
        "european_mc": (
            "Running European call sweep...",
            run_european_sweep,
        ),
        "asian_mc": (
            "Running Asian call sweep...",
            run_asian_sweep,
        ),
        "european_antithetic": (
            "Running European call antithetic sweep...",
            run_european_antithetic_sweep,
        ),
        "asian_antithetic": (
            "Running Asian call antithetic sweep...",
            run_asian_antithetic_sweep,
        ),
        "european_control_variate": (
            "Running European call control variate sweep...",
            run_european_control_variate_sweep,
        ),
        "asian_control_variate": (
            "Running Asian call control variate sweep...",
            run_asian_control_variate_sweep,
        ),
        "european_quasi_monte_carlo": (
            "Running European call quasi-Monte Carlo sweep...",
            run_european_quasi_monte_carlo_sweep,
        ),
        "asian_quasi_monte_carlo": (
            "Running Asian call quasi-Monte Carlo sweep...",
            run_asian_quasi_monte_carlo_sweep,
        ),
        "european_multilevel_monte_carlo": (
            "Running European call multilevel Monte Carlo sweep...",
            run_european_multilevel_sweep,
        ),
        "asian_multilevel_monte_carlo": (
            "Running Asian call multilevel Monte Carlo sweep...",
            run_asian_multilevel_sweep,
        ),
    }

    results = {}

    for name, (message, experiment) in experiments.items():
        print(f"\n{message}")
        results[name] = experiment(params=params)

    print("\nRunning C++ benchmarks...")
    run_cpp_benchmarks()

    results["cpp_european_mc"] = pd.read_csv(
        "results/data/cpp_european_mc_summary.csv"
    )

    results["cpp_asian_mc"] = pd.read_csv(
        "results/data/cpp_asian_mc_summary.csv"
    )

    return results


def save_results(results, results_dir=RESULTS_DIR):
    """
    Save summary results to CSV files.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        if name.startswith("cpp_"):
            continue

        df.to_csv(
            results_path / f"{name}_summary.csv",
            index=False,
        )

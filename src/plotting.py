# src/plotting.py

import matplotlib.pyplot as plt
from pathlib import Path


def ensure_dir(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def _plot_multiple_series(
    data,
    x_key,
    y_key,
    title,
    xlabel,
    ylabel,
    outpath,
    xscale="log",
    yscale="log",
):
    """
    Generic helper for plotting multiple summary series on the same axes.
    """
    ensure_dir(outpath)

    plt.figure(figsize=(8, 5))

    for label, df in data.items():
        plt.plot(df[x_key], df[y_key], marker="o", label=label)

    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(frameon=False)
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_error_comparison(data, title, outpath):
    """
    Compare mean absolute error against number of paths for multiple methods.
    """
    _plot_multiple_series(
        data=data,
        x_key="n_paths",
        y_key="mean_abs_error",
        title=title,
        xlabel="Number of paths",
        ylabel="Mean absolute error",
        outpath=outpath,
        xscale="log",
        yscale="log",
    )


def plot_runtime_error_comparison(data, title, outpath):
    """
    Compare runtime against mean absolute error for multiple methods.
    """
    _plot_multiple_series(
        data=data,
        x_key="mean_abs_error",
        y_key="mean_runtime_sec",
        title=title,
        xlabel="Mean absolute error",
        ylabel="Mean runtime (sec)",
        outpath=outpath,
        xscale="log",
        yscale="log",
    )


def plot_all(results, figures_dir="results/figures"):
    """
    Create the main comparison plots for the project.
    """

    european = {
        "Standard MC": results["european_mc"],
        "Antithetic": results["european_antithetic"],
        "Control Variate": results["european_control_variate"],
        "Sobol Quasi-MC": results["european_quasi_monte_carlo"],
        "C++ MC": results["cpp_european_mc"]
    }

    asian = {
        "Standard MC": results["asian_mc"],
        "Antithetic": results["asian_antithetic"],
        "Control Variate": results["asian_control_variate"],
        "Sobol Quasi-MC": results["asian_quasi_monte_carlo"],
        "C++ MC": results["cpp_asian_mc"]
    }

    plot_error_comparison(
        european,
        "European Call: Error vs Number of Paths",
        f"{figures_dir}/european_error_comparison.png",
    )

    plot_runtime_error_comparison(
        european,
        "European Call: Runtime vs Error",
        f"{figures_dir}/european_runtime_error_comparison.png",
    )

    plot_error_comparison(
        asian,
        "Asian Call: Error vs Number of Paths",
        f"{figures_dir}/asian_error_comparison.png",
    )

    plot_runtime_error_comparison(
        asian,
        "Asian Call: Runtime vs Error",
        f"{figures_dir}/asian_runtime_error_comparison.png",
    )


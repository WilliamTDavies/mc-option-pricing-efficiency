# src/plotting.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import METHOD_STYLES


def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _prepare_summary(df):
    """
    Add derived metrics used by the plots.
    """
    tmp = df.copy()

    if "rmse" not in tmp.columns:
        bias = tmp["mean_price"] - tmp["reference_price"]
        tmp["rmse"] = np.sqrt(bias**2 + tmp["std_price"]**2)

    return tmp


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
    annotate_slope=False,
):
    """
    Generic helper for plotting multiple summary series.
    """
    ensure_dir(outpath)

    plt.figure(figsize=(9, 6))

    for label, df in data.items():
        tmp = _prepare_summary(df)

        tmp = tmp.sort_values(x_key)

        style = METHOD_STYLES.get(
            label,
            dict(marker="o", linestyle="-", color=None),
        )

        line, = plt.plot(
            tmp[x_key],
            tmp[y_key],
            label=label,
            linewidth=2,
            markersize=6,
            **style,
        )

        if annotate_slope and len(tmp) >= 2 and xscale == "log" and yscale == "log":
            x = np.log(tmp[x_key].to_numpy(dtype=float))
            y = np.log(tmp[y_key].to_numpy(dtype=float))

            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                slope, intercept = np.polyfit(x, y, 1)
                x_fit = np.exp(x)
                y_fit = np.exp(intercept + slope * x)

                plt.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    linewidth=1.5,
                    color=line.get_color(),
                    alpha=0.7,
                )

                last_x = tmp[x_key].iloc[-1]
                last_y = tmp[y_key].iloc[-1]
                plt.annotate(
                    f"{slope:.2f}",
                    xy=(last_x, last_y),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=9,
                    color=line.get_color(),
                )

    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    plt.grid(True, which="major", alpha=0.35)
    plt.grid(True, which="minor", alpha=0.15)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_comparison(data, title, outpath):
    """
    Compare RMSE against number of paths for multiple methods.
    """
    _plot_multiple_series(
        data=data,
        x_key="n_paths",
        y_key="rmse",
        title=title,
        xlabel="Number of Paths",
        ylabel="RMSE",
        outpath=outpath,
        xscale="log",
        yscale="log",
    )


def plot_runtime_error_comparison(data, title, outpath):
    """
    Compare runtime against RMSE for multiple methods.
    Lower-left is better.
    """
    _plot_multiple_series(
        data=data,
        x_key="mean_runtime_sec",
        y_key="rmse",
        title=title,
        xlabel="Runtime (seconds)",
        ylabel="RMSE",
        outpath=outpath,
        xscale="log",
        yscale="log",
    )


def plot_runtime_paths(data, title, outpath):
    """
    Compare runtime against number of paths.
    """
    _plot_multiple_series(
        data=data,
        x_key="n_paths",
        y_key="mean_runtime_sec",
        title=title,
        xlabel="Number of Paths",
        ylabel="Runtime (seconds)",
        outpath=outpath,
        xscale="log",
        yscale="log",
    )


def plot_all(results, figures_dir="results/figures"):
    """
    Create all figures for the project.
    """

    european = {
        "Standard MC": results["european_mc"],
        "Antithetic Variate": results["european_antithetic"],
        "Control Variate": results["european_control_variate"],
        "Sobol Quasi-MC": results["european_quasi_monte_carlo"],
        "Multilevel MC": results["european_multilevel_monte_carlo"],
        "C++ MC": results["cpp_european_mc"],
    }

    asian = {
        "Standard MC": results["asian_mc"],
        "Antithetic Variate": results["asian_antithetic"],
        "Control Variate": results["asian_control_variate"],
        "Sobol Quasi-MC": results["asian_quasi_monte_carlo"],
        "Multilevel MC": results["asian_multilevel_monte_carlo"],
        "C++ MC": results["cpp_asian_mc"],
    }

    # European
    plot_error_comparison(
        european,
        "European Call: RMSE vs Number of Paths",
        f"{figures_dir}/european_error_comparison.png",
    )

    plot_runtime_error_comparison(
        european,
        "European Call: Runtime vs RMSE",
        f"{figures_dir}/european_runtime_error.png",
    )

    plot_runtime_paths(
        european,
        "European Call: Runtime vs Number of Paths",
        f"{figures_dir}/european_runtime_paths.png",
    )

    # Asian
    plot_error_comparison(
        asian,
        "Asian Call: RMSE vs Number of Paths",
        f"{figures_dir}/asian_error_comparison.png",
    )

    plot_runtime_error_comparison(
        asian,
        "Asian Call: Runtime vs RMSE",
        f"{figures_dir}/asian_runtime_error.png",
    )

    plot_runtime_paths(
        asian,
        "Asian Call: Runtime vs Number of Paths",
        f"{figures_dir}/asian_runtime_paths.png",
    )

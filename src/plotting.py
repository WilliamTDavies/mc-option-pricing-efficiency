import matplotlib.pyplot as plt
from pathlib import Path


def ensure_dir(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def plot_error_vs_paths(df, title, outpath):
    """
    Plot mean absolute error against number of paths.
    """
    ensure_dir(outpath)

    plt.figure(figsize=(8, 5))
    plt.plot(df["n_paths"], df["mean_abs_error"], marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of paths")
    plt.ylabel("Mean absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_runtime_vs_error(df, title, outpath):
    """
    Plot runtime against mean absolute error.
    """
    ensure_dir(outpath)

    plt.figure(figsize=(8, 5))
    plt.plot(df["mean_abs_error"], df["mean_runtime_sec"], marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean absolute error")
    plt.ylabel("Mean runtime (sec)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_all(results, figures_dir="results/figures"):
    """
    Create all standard plots for the project.
    """
    european = results["european"]
    asian = results["asian"]

    plot_error_vs_paths(
        european,
        "European call: error vs number of paths",
        f"{figures_dir}/european_error_vs_paths.png",
    )
    plot_runtime_vs_error(
        european,
        "European call: runtime vs error",
        f"{figures_dir}/european_runtime_vs_error.png",
    )

    plot_error_vs_paths(
        asian,
        "Asian call: error vs number of paths",
        f"{figures_dir}/asian_error_vs_paths.png",
    )
    plot_runtime_vs_error(
        asian,
        "Asian call: runtime vs error",
        f"{figures_dir}/asian_runtime_vs_error.png",
    )
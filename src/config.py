# src/config.py

OPTION_PARAMS = {
    "S0": 100.0,
    "K": 100.0,
    "r": 0.05,
    "sigma": 0.20,
    "T": 1.0,
}

PATH_GRID = [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]

N_REPLICATIONS = 100
ASIAN_N_STEPS = 252

REFERENCE_PATHS = 5000000
REFERENCE_SEED = 42

BASE_SEED = 0

EUROPEAN_SEED_OFFSET = 100000
ASIAN_SEED_OFFSET = 200000

RESULTS_DIR = "results/data"
FIGURES_DIR = "results/figures"

METHOD_STYLES = {
    "Standard MC": dict(color="#1f77b4", marker="o", linestyle="-"),
    "Antithetic Variate": dict(color="#ff7f0e", marker="s", linestyle="-"),
    "Control Variate": dict(color="#2ca02c", marker="^", linestyle="-"),
    "Sobol Quasi-MC": dict(color="#d62728", marker="D", linestyle="-"),
    "Multilevel MC": dict(color="#7b3294", marker="v", linestyle="-"),
    "C++ MC": dict(color="#444444", marker="P", linestyle="-"),
}
from src.experiments import run_all_experiments, save_results
from src.plotting import plot_all


def main():
    print("Running experiments...")
    results = run_all_experiments()
    print("Experiments completed.")

    save_results(results)
    print("Saved CSVs to results/data/processed/")

    plot_all(results)
    print("Saved figures to results/figures/")


if __name__ == "__main__":
    main()
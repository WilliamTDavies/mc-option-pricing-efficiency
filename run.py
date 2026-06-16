from src.experiments import run_all_experiments, save_results
from src.plotting import plot_all


def main():
    print("Running experiments...")
    results = run_all_experiments()
    print("\nExperiments completed.")

    save_results(results)
    print("\nSaved CSVs to results/data/")

    plot_all(results)
    print("\nSaved figures to results/figures/")


if __name__ == "__main__":
    main()
"""
Example usage of the sensitivity analysis script.

This script demonstrates how to:
1. Run the sensitivity analysis
2. Access and filter results
3. Save results to CSV files
4. Generate plots for all effect values
"""

from sensanaly_capeff import analyze_sensitivity, plot_all_effects
import pandas as pd
import os
import sys

# Configuration
results_dir = sys.argv[1]
metrics = [
    "offense_rate",
    "incarceration_rate",
]
# policies = ["null", "high-risk", "low-risk", "age-first", "age-first-high-risk"]
# policies = ["null", "high-risk", "low-risk", "age-first-high-risk"]
policies = ["null", "high-risk", "low-risk", "age-first"]
summary_wd_last = 20  # Window size for computing equilibrium (last N periods)
summary_wd_first = 50  # Window size for computing equilibrium (first N periods)
summary_wd_start = 0  # Starting period index when use_first=True

print("=" * 60)
print("Running Sensitivity Analysis")
print("=" * 60)
print(f"Results directory: {results_dir}")
print(f"Metrics: {metrics}")
print(f"Policies: {policies}")
print(f"Summary window (last/equilibrium): {summary_wd_last} periods")
print(f"Summary window (first): {summary_wd_first} periods")
print(f"Summary window (start): {summary_wd_start} periods")
print()

# Create base output directory inside results_dir
base_output_dir = os.path.join(results_dir, "sensitivity_analysis")
os.makedirs(base_output_dir, exist_ok=True)


def run_analysis_and_plot(
    summary_wd: int, use_first: bool, suffix: str, start_from: int = 0
):
    """Run analysis and generate plots for a given window configuration."""
    print("\n" + "=" * 60)
    print(
        f"Running Analysis: {suffix} (window={summary_wd}, use_first={use_first}, start_from={start_from})"
    )
    print("=" * 60)

    # Run analysis
    dfs = analyze_sensitivity(
        results_dir,
        metrics,
        policies,
        summary_wd,
        use_first=use_first,
        start_from=start_from,
    )

    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    # Create output subdirectory
    output_dir = os.path.join(base_output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)

    # Save each metric to a separate CSV file
    for metric, df in dfs.items():
        filename = os.path.join(output_dir, f"sensitivity_{metric}.csv")
        df.to_csv(filename)
        print(f"Saved {filename} (shape: {df.shape})")

    print("\n" + "-" * 40)
    print("Generating Plots")
    print("-" * 40)

    # Generate plots for all effects (absolute)
    plot_all_effects(
        dfs=dfs,
        output_dir=output_dir,
        policies=policies,
        show_std=True,
    )
    # Generate plots for all effects (relative to null)
    plot_all_effects(
        dfs=dfs,
        output_dir=output_dir,
        policies=policies,
        show_std=True,
        relative=True,
        relative_policy="null",
    )

    print(f"\nResults saved to {output_dir}/")
    return dfs


# Run for equilibrium (last N periods)
dfs_last = run_analysis_and_plot(summary_wd_last, use_first=False, suffix="equilibrium")

# Run for first N periods (starting from summary_wd_start)
dfs_first = run_analysis_and_plot(
    summary_wd_first, use_first=True, suffix="first", start_from=summary_wd_start
)

print("\n" + "=" * 60)
print(f"All results saved to {base_output_dir}/")
print("=" * 60)

"""
Example usage of the prison scale factor sensitivity analysis script.

This script demonstrates how to:
1. Run the sensitivity analysis
2. Access and filter results
3. Save results to CSV files
4. Generate plots for all metrics
"""

from .sensanaly_prscale import analyze_sensitivity, plot_all_term_lengths
import pandas as pd
import os
import sys

# Configuration
results_dir = sys.argv[1]
metrics = [
    "offense_rate",
    "offense_per_capita",
    "incarceration_rate",
    "total_population",
    "total_offenses",
    "total_departures",
    "total_offenses/total_population",
    "total_departures/total_population",
]
policies = [
    "null",
    "high-risk",
    "low-risk",
    "age-first",
    # "age-first-high-risk",
    # "high-risk-young-first",
]
summary_wd_last = 20  # Window size for computing equilibrium (last N periods)
summary_wd_first = 40  # Window size for computing first N periods
summary_wd_start = 40

print("=" * 60)
print("Running Prison Scale Factor Sensitivity Analysis")
print("=" * 60)
print(f"Results directory: {results_dir}")
print(f"Metrics: {metrics}")
print(f"Policies: {policies}")
print(f"Summary window (last/equilibrium): {summary_wd_last} periods")
print(f"Summary window (first): {summary_wd_first} periods")
print()

# Create base output directory inside results_dir
base_output_dir = os.path.join(results_dir, "sensitivity_analysis")
os.makedirs(base_output_dir, exist_ok=True)


def run_analysis_and_plot(summary_wd: int, use_first: bool, suffix: str):
    """Run analysis and generate plots for a given window configuration."""
    print("\n" + "=" * 60)
    print(f"Running Analysis: {suffix} (window={summary_wd}, use_first={use_first})")
    print("=" * 60)

    # Run analysis
    dfs = analyze_sensitivity(
        results_dir,
        metrics,
        policies,
        summary_wd,
        use_first=use_first,
        start_from=summary_wd_start,
    )

    print("\n" + "-" * 40)
    print("Saving Results")
    print("-" * 40)

    # Create output subdirectory
    output_dir = os.path.join(base_output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)

    # Save each metric to a separate CSV file
    for metric, df in dfs.items():
        # Sanitize metric name for filename (replace / with _per_)
        metric_filename = metric.replace("/", "_per_")
        filename = os.path.join(output_dir, f"sensitivity_{metric_filename}.csv")
        df.to_csv(filename)
        print(f"Saved {filename} (shape: {df.shape})")

    print("\n" + "-" * 40)
    print("Generating Plots")
    print("-" * 40)

    # Generate plots for all metrics (absolute)
    plot_all_term_lengths(
        dfs=dfs,
        output_dir=output_dir,
        policies=policies,
        show_std=True,
        scale=0.12,
        relative=False,
    )
    # Generate plots for all metrics (relative to null)
    plot_all_term_lengths(
        dfs=dfs,
        output_dir=output_dir,
        policies=policies,
        show_std=True,
        scale=0.12,
        relative=True,
        relative_policy="null",
    )

    print(f"\nResults saved to {output_dir}/")
    return dfs


# Run for equilibrium (last N periods)
dfs_last = run_analysis_and_plot(summary_wd_last, use_first=False, suffix="equilibrium")

# Run for first N periods
dfs_first = run_analysis_and_plot(summary_wd_first, use_first=True, suffix="first")

print("\n" + "=" * 60)
print(f"All results saved to {base_output_dir}/")
print("=" * 60)

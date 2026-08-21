"""
Sensitivity Analysis Script for Treatment Capacity and Effect Parameters.

This script analyzes simulation results across varying treatment capacity (C) and
treatment effect (e) parameters, computing performance metrics relative to null baseline.

Returns DataFrames indexed by (effect, capacity, policy) with mean ± std of
per-repetition differences from null baseline.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings

# Import existing functionality
from simulation_summary import read_metrics_from_h5
from simulation_sensanaly import compute_equilibrium_value, plot_sensitivity_trends


def parse_directory_structure(
    results_dir: str,
) -> Tuple[List[int], List, List[str]]:
    """
    Scan results directory to find all (C, e) combinations and available policies.

    Args:
        results_dir: Path to result-capa-eff directory

    Returns:
        Tuple of (capacities, effects, policies)
        - capacities: List of capacity values (e.g., [50, 100, 200, ...])
        - effects: List of effect values (e.g., [0.1, 0.2, ...])
        - policies: List of policy names (e.g., ['high-risk', 'low-risk', ...])
    """
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")

    capacities = set()
    effects = set()
    policies = set()

    # Pattern to match directory names like "c-50-e-0.1" or "c-50-e-type-1"
    # Effect can be numeric (e.g., 0.1, 0.731) or string (e.g., type-1)
    pattern = re.compile(r"c-(\d+)-e-(.+)")

    for dirname in os.listdir(results_dir):
        dirpath = os.path.join(results_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        match = pattern.match(dirname)
        if match:
            capacity = int(match.group(1))
            effect_str = match.group(2)
            # Try to parse as float, otherwise keep as string
            try:
                effect = float(effect_str)
            except ValueError:
                effect = effect_str
            capacities.add(capacity)
            effects.add(effect)

            # Find policies in this directory
            for policy_name in os.listdir(dirpath):
                policy_path = os.path.join(dirpath, policy_name)
                if os.path.isdir(policy_path) and policy_name != "logs":
                    policies.add(policy_name)

    # Keep 'null' in policies list to show baseline
    # Sort effects: numeric first (sorted), then strings (sorted)
    numeric_effects = sorted([e for e in effects if isinstance(e, (int, float))])
    string_effects = sorted([e for e in effects if isinstance(e, str)])
    sorted_effects = numeric_effects + string_effects
    return (sorted(list(capacities)), sorted_effects, sorted(list(policies)))


def load_policy_metrics(
    results_dir: str, capacity: int, effect, policy_name: str
) -> Tuple[Dict, Dict]:
    """
    Load metrics for a specific policy at given capacity and effect.

    Args:
        results_dir: Path to result-capa-eff directory
        capacity: Treatment capacity value
        effect: Treatment effect value
        policy_name: Name of policy (e.g., 'high-risk', 'null')

    Returns:
        Tuple of (policy_metrics, policy_agg) from read_metrics_from_h5()
        - policy_metrics: dict[metric_key] -> 2D array (repeats × episodes)
        - policy_agg: dict[metric_key] -> dict with 'mean', 'std', 'min', 'max'
    """
    policy_dir = f"c-{capacity}-e-{effect}"
    output_dir = os.path.join(results_dir, policy_dir)

    if not os.path.exists(os.path.join(output_dir, policy_name)):
        raise ValueError(
            f"Policy directory not found: {os.path.join(output_dir, policy_name)}"
        )

    return read_metrics_from_h5(policy_name, output_dir)


def compute_relative_performance(
    policy_values: np.ndarray, null_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute performance statistics relative to null baseline.

    Computes per-repetition differences (policy - null), then aggregates.

    Args:
        policy_values: Array of equilibrium values for policy (n_reps,)
        null_values: Array of equilibrium values for null baseline (n_reps,)

    Returns:
        Dict with keys: 'mean', 'std', 'n_reps'
    """
    # Ensure arrays have same length
    min_len = min(len(policy_values), len(null_values))
    if len(policy_values) != len(null_values):
        warnings.warn(
            f"Mismatched repetition counts: policy={len(policy_values)}, "
            f"null={len(null_values)}. Using first {min_len} repetitions."
        )
        policy_values = policy_values[:min_len]
        null_values = null_values[:min_len]

    # Per-repetition differences
    differences = policy_values - null_values

    # Compute statistics
    return {
        "mean": float(np.mean(differences)),
        "std": float(np.std(differences)),
        "n_reps": len(differences),
    }


def analyze_sensitivity(
    results_dir: str,
    metrics: List[str],
    policies: List[str],
    summary_wd: int,
    use_first: bool = False,
    start_from: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze sensitivity across capacity and effect parameters.

    Args:
        results_dir: Path to result-capa-eff directory
        metrics: List of metric names (e.g., ['total_offenses'])
        policies: List of policy names (e.g., ['high-risk', 'low-risk', 'null'])
        summary_wd: Window size for equilibrium computation (num_period_window)
        use_first: If False (default), use last summary_wd periods (equilibrium).
                   If True, use first summary_wd periods starting from start_from.
        start_from: Starting period index when use_first=True (default: 0).

    Returns:
        dict: {metric_name: DataFrame} where each DataFrame has:
            - Index: MultiIndex (effect, capacity, policy)
            - Columns: ['mean', 'std', 'n_reps', 'enrollment_mean', 'enrollment_std']
            - Values: Absolute values averaged over summary_wd periods
            - All statistics are per-period averages (divided by summary_wd)
    """
    print(f"Parsing directory structure from {results_dir}...")
    capacities, effects, available_policies = parse_directory_structure(results_dir)

    print(f"Found {len(capacities)} capacities: {capacities}")
    print(f"Found {len(effects)} effects: {effects}")
    print(f"Found {len(available_policies)} policies: {available_policies}")

    # Validate requested policies
    for policy in policies:
        if policy not in available_policies:
            warnings.warn(f"Policy '{policy}' not found in results directory")

    # Filter to available policies
    policies_to_analyze = [p for p in policies if p in available_policies]
    print(f"Analyzing {len(policies_to_analyze)} policies: {policies_to_analyze}")

    # Collect results for each metric
    results = {metric: [] for metric in metrics}

    # Loop through all combinations
    total_combinations = len(effects) * len(capacities) * len(policies_to_analyze)
    processed = 0

    for effect in effects:
        for capacity in capacities:
            for policy in policies_to_analyze:
                processed += 1
                print(
                    f"Processing [{processed}/{total_combinations}]: "
                    f"effect={effect}, capacity={capacity}, policy={policy}"
                )

                try:
                    # Load policy metrics
                    policy_metrics, _ = load_policy_metrics(
                        results_dir, capacity, effect, policy
                    )

                    # Compute enrollment statistics (absolute values)
                    # Divide by summary_wd to get average per period
                    enrollment_vals = compute_equilibrium_value(
                        policy_metrics,
                        "total_enrollment",
                        summary_wd,
                        use_first,
                        start_from,
                    )
                    enrollment_mean = float(np.mean(enrollment_vals)) / summary_wd
                    enrollment_std = float(np.std(enrollment_vals)) / summary_wd

                    # Compute statistics for each metric
                    for metric in metrics:
                        try:
                            # Compute values (sum over summary_wd periods)
                            policy_vals = compute_equilibrium_value(
                                policy_metrics,
                                metric,
                                summary_wd,
                                use_first,
                                start_from,
                            )

                            # Compute absolute statistics (averaged per period)
                            # Divide by summary_wd to get per-period average
                            policy_vals_per_period = policy_vals / summary_wd

                            # Store result
                            results[metric].append(
                                {
                                    "effect": effect,
                                    "capacity": capacity,
                                    "policy": policy,
                                    "mean": float(np.mean(policy_vals_per_period)),
                                    "std": float(np.std(policy_vals_per_period)),
                                    "n_reps": len(policy_vals_per_period),
                                    "enrollment_mean": enrollment_mean,
                                    "enrollment_std": enrollment_std,
                                }
                            )

                        except Exception as e:
                            warnings.warn(
                                f"Failed to compute metric '{metric}' for "
                                f"C={capacity}, e={effect}, policy={policy}: {e}"
                            )
                            continue

                except Exception as e:
                    warnings.warn(
                        f"Failed to load policy '{policy}' for "
                        f"C={capacity}, e={effect}: {e}"
                    )
                    continue

    # Convert to DataFrames with MultiIndex
    dfs = {}
    for metric in metrics:
        if not results[metric]:
            warnings.warn(f"No results collected for metric '{metric}'")
            continue

        df = pd.DataFrame(results[metric])
        df = df.set_index(["effect", "capacity", "policy"])
        df = df.sort_index()
        dfs[metric] = df

        print(f"\nMetric '{metric}': {len(df)} combinations")

    return dfs


def compute_metric_ylim(df: pd.DataFrame, show_std: bool = True) -> Tuple[float, float]:
    """
    Compute consistent y-axis limits for a metric across all filter values.

    Args:
        df: DataFrame with 'mean' and optionally 'std' columns
        show_std: Whether std bands will be shown (affects limit computation)

    Returns:
        Tuple of (ymin, ymax) with some padding
    """
    if show_std and "std" in df.columns:
        ymin = (df["mean"] - df["std"]).min()
        ymax = (df["mean"] + df["std"]).max()
    else:
        ymin = df["mean"].min()
        ymax = df["mean"].max()

    # Add 5% padding
    padding = (ymax - ymin) * 0.05
    return (ymin - padding, ymax + padding)


def plot_capacity_trends(
    df: pd.DataFrame,
    metric_name: str,
    effect,
    output_path: str,
    policies: Optional[List[str]] = None,
    ylabel: Optional[str] = None,
    show_std: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    zoom_margin: Optional[float] = None,
    relative: bool = False,
    relative_policy: str = "null",
):
    """
    Plot trends vs. capacity for each policy at a given effect level.

    Args:
        df: DataFrame from analyze_sensitivity (for one metric)
        metric_name: Name of the metric (for title)
        effect: Effect value to plot (e.g., 0.5)
        output_path: Base path to save files (will save .png and .pgf)
        policies: List of policies to plot (default: all in df)
        ylabel: Y-axis label (default: metric_name)
        show_std: Whether to show confidence interval region (default: True)
        ylim: Y-axis limits as (min, max) tuple (default: None, auto)
        zoom_margin: If set, zoom y-axis to data range with this fraction as margin (e.g., 0.1 = 10%)
        relative: If True, plot difference from relative_policy (default: False)
        relative_policy: Policy to use as baseline for relative plots (default: "null")
    """
    plot_sensitivity_trends(
        df=df,
        metric_name=metric_name,
        filter_key="effect",
        filter_value=effect,
        x_key="capacity",
        output_path=output_path,
        policies=policies,
        ylabel=ylabel,
        show_std=show_std,
        ylim=ylim,
        zoom_margin=zoom_margin,
        relative=relative,
        relative_policy=relative_policy,
    )


def plot_all_effects(
    dfs: Dict[str, pd.DataFrame],
    output_dir: str,
    policies: Optional[List[str]] = None,
    show_std: bool = True,
    shared_ylim: bool = False,
    zoom_margin: Optional[float] = None,
    relative: bool = False,
    relative_policy: str = "null",
):
    """
    Plot capacity trends for all metrics and all effect values.

    Args:
        dfs: Dict of DataFrames from analyze_sensitivity
        output_dir: Directory to save PNG and PGF files
        policies: List of policies to plot (default: all)
        show_std: Whether to show confidence interval region (default: True)
        shared_ylim: Whether to use same y-axis limits for all plots of same metric (default: True)
        zoom_margin: If set, zoom y-axis to data range with this fraction as margin (e.g., 0.1 = 10%)
        relative: If True, plot difference from relative_policy (default: False)
        relative_policy: Policy to use as baseline for relative plots (default: "null")
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, df in dfs.items():
        # Get all effect values (sort numeric first, then strings)
        all_effects = df.index.get_level_values("effect").unique()
        numeric_effects = sorted(
            [e for e in all_effects if isinstance(e, (int, float))]
        )
        string_effects = sorted([e for e in all_effects if isinstance(e, str)])
        effects = numeric_effects + string_effects

        # Compute consistent y-limits for this metric if requested (only for absolute values)
        ylim = (
            compute_metric_ylim(df, show_std) if shared_ylim and not relative else None
        )

        print(f"\nPlotting {metric_name} for {len(effects)} effect values...")
        if shared_ylim and not relative:
            print(f"  Using shared y-limits: {ylim}")
        if relative:
            print(f"  Plotting relative to '{relative_policy}'")

        for effect in effects:
            # Create base filename (without extension)
            # Handle both numeric and string effect values
            if isinstance(effect, float):
                effect_str = f"{effect}"
            else:
                effect_str = str(effect)
            filename = f"{metric_name}_effect_{effect_str}"
            if relative:
                filename += f"_rel_{relative_policy}"
            output_path = os.path.join(output_dir, filename)

            try:
                plot_capacity_trends(
                    df=df,
                    metric_name=metric_name,
                    effect=effect,
                    output_path=output_path,
                    policies=policies,
                    show_std=show_std,
                    ylim=ylim,
                    zoom_margin=zoom_margin,
                    relative=relative,
                    relative_policy=relative_policy,
                )
            except Exception as e:
                warnings.warn(f"Failed to plot {metric_name} for effect={effect}: {e}")
                continue

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Sensitivity Analysis Example")
    print("=" * 60)

    # Configuration
    results_dir = "result-capa-eff"
    metrics = ["total_offenses", "offense_rate_treated"]
    policies = ["high-risk", "low-risk", "random"]
    summary_wd = 20

    # Run analysis
    print(f"\nConfiguration:")
    print(f"  Results directory: {results_dir}")
    print(f"  Metrics: {metrics}")
    print(f"  Policies: {policies}")
    print(f"  Summary window: {summary_wd}")
    print()

    try:
        dfs = analyze_sensitivity(results_dir, metrics, policies, summary_wd)

        # Display results
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        for metric, df in dfs.items():
            print(f"\n{metric}:")
            print(f"  Shape: {df.shape}")
            print(f"  Index levels: {df.index.names}")
            print(f"  Columns: {df.columns.tolist()}")
            print("\nFirst few rows:")
            print(df.head(10))

            # Example queries
            print(f"\nExample: All results for effect=0.1:")
            if 0.1 in df.index.get_level_values("effect"):
                print(df.loc[0.1].head())

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

"""
Sensitivity Analysis Script for Off-Probation Term Length and Prison Scale Factor.

This script analyzes simulation results across varying off-probation term length (tl) and
prison scale factor (sf) parameters, computing performance metrics for different policies.

Returns DataFrames indexed by (term_length, scale_factor, policy) with mean ± std of
per-repetition values.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings

# Import existing functionality
from simulation_summary import read_metrics_from_h5
from simulation_sensanaly import compute_equilibrium_value, plot_sensitivity_trends


def parse_directory_structure(
    results_dir: str,
) -> Tuple[List[int], List[float], List[str]]:
    """
    Scan results directory to find all (term_length, scale_factor) combinations and available policies.

    Args:
        results_dir: Path to result-ofpl-scl directory

    Returns:
        Tuple of (term_lengths, scale_factors, policies)
        - term_lengths: List of term length values (e.g., [500, 1000, 1500, ...])
        - scale_factors: List of scale factor values (e.g., [0.05, 0.1, 0.2, ...])
        - policies: List of policy names (e.g., ['high-risk', 'low-risk', ...])
    """
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")

    term_lengths = set()
    scale_factors = set()
    policies = set()

    # Pattern to match directory names like "tl-4000-sf-1.0"
    pattern = re.compile(r"tl-(\d+)-sf-([\d.]+)")

    for dirname in os.listdir(results_dir):
        dirpath = os.path.join(results_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        match = pattern.match(dirname)
        if match:
            term_length = int(match.group(1))
            scale_factor = float(match.group(2))
            term_lengths.add(term_length)
            scale_factors.add(scale_factor)

            # Find policies in this directory
            for policy_name in os.listdir(dirpath):
                policy_path = os.path.join(dirpath, policy_name)
                if os.path.isdir(policy_path) and policy_name != "logs":
                    policies.add(policy_name)

    return (
        sorted(list(term_lengths)),
        sorted(list(scale_factors)),
        sorted(list(policies)),
    )


def load_policy_metrics(
    results_dir: str, term_length: int, scale_factor: float, policy_name: str
) -> Tuple[Dict, Dict]:
    """
    Load metrics for a specific policy at given term length and scale factor.

    Args:
        results_dir: Path to result-ofpl-scl directory
        term_length: Off-probation term length value
        scale_factor: Prison scale factor value
        policy_name: Name of policy (e.g., 'high-risk', 'null')

    Returns:
        Tuple of (policy_metrics, policy_agg) from read_metrics_from_h5()
        - policy_metrics: dict[metric_key] -> 2D array (repeats × episodes)
        - policy_agg: dict[metric_key] -> dict with 'mean', 'std', 'min', 'max'
    """
    policy_dir = f"tl-{term_length}-sf-{scale_factor}"
    output_dir = os.path.join(results_dir, policy_dir)

    if not os.path.exists(os.path.join(output_dir, policy_name)):
        raise ValueError(
            f"Policy directory not found: {os.path.join(output_dir, policy_name)}"
        )

    return read_metrics_from_h5(policy_name, output_dir)


def analyze_sensitivity(
    results_dir: str,
    metrics: List[str],
    policies: List[str],
    summary_wd: int,
    use_first: bool = False,
    start_from: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze sensitivity across term length and scale factor parameters.

    Args:
        results_dir: Path to result-ofpl-scl directory
        metrics: List of metric names (e.g., ['total_offenses'])
        policies: List of policy names (e.g., ['high-risk', 'low-risk', 'null'])
        summary_wd: Window size for equilibrium computation (num_period_window)
        use_first: If False (default), use last summary_wd periods (equilibrium).
                   If True, use first summary_wd periods starting from start_from.
        start_from: Starting period index when use_first=True (default: 0).

    Returns:
        dict: {metric_name: DataFrame} where each DataFrame has:
            - Index: MultiIndex (term_length, scale_factor, policy)
            - Columns: ['mean', 'std', 'n_reps', 'enrollment_mean', 'enrollment_std']
            - Values: Absolute values averaged over summary_wd periods
            - All statistics are per-period averages (divided by summary_wd)
    """
    print(f"Parsing directory structure from {results_dir}...")
    term_lengths, scale_factors, available_policies = parse_directory_structure(
        results_dir
    )

    print(f"Found {len(term_lengths)} term lengths: {term_lengths}")
    print(f"Found {len(scale_factors)} scale factors: {scale_factors}")
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
    total_combinations = (
        len(term_lengths) * len(scale_factors) * len(policies_to_analyze)
    )
    processed = 0

    for term_length in term_lengths:
        for scale_factor in scale_factors:
            for policy in policies_to_analyze:
                processed += 1
                print(
                    f"Processing [{processed}/{total_combinations}]: "
                    f"term_length={term_length}, scale_factor={scale_factor}, policy={policy}"
                )

                try:
                    # Load policy metrics
                    policy_metrics, _ = load_policy_metrics(
                        results_dir, term_length, scale_factor, policy
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
                            # Check if this is a computed metric (e.g., "total_offenses/total_population")
                            if "/" in metric:
                                # Parse numerator and denominator
                                numerator_key, denominator_key = metric.split("/")
                                numerator_vals = compute_equilibrium_value(
                                    policy_metrics,
                                    numerator_key,
                                    summary_wd,
                                    use_first,
                                    start_from,
                                )
                                denominator_vals = compute_equilibrium_value(
                                    policy_metrics,
                                    denominator_key,
                                    summary_wd,
                                    use_first,
                                    start_from,
                                )
                                # Compute ratio (both are sums over summary_wd periods)
                                policy_vals = numerator_vals / denominator_vals
                                # For ratios, we don't divide by summary_wd again
                                policy_vals_per_period = policy_vals
                            else:
                                # Compute values (sum over summary_wd periods)
                                policy_vals = compute_equilibrium_value(
                                    policy_metrics,
                                    metric,
                                    summary_wd,
                                    use_first,
                                    start_from,
                                )
                                # Compute absolute statistics (averaged per period)
                                # Dividec by summary_wd to get per-period average
                                policy_vals_per_period = policy_vals / summary_wd

                            # Store result
                            results[metric].append(
                                {
                                    "term_length": term_length,
                                    "scale_factor": scale_factor,
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
                                f"tl={term_length}, sf={scale_factor}, policy={policy}: {e}"
                            )
                            continue

                except Exception as e:
                    warnings.warn(
                        f"Failed to load policy '{policy}' for "
                        f"tl={term_length}, sf={scale_factor}: {e}"
                    )
                    continue

    # Convert to DataFrames with MultiIndex
    dfs = {}
    for metric in metrics:
        if not results[metric]:
            warnings.warn(f"No results collected for metric '{metric}'")
            continue

        df = pd.DataFrame(results[metric])
        df = df.set_index(["term_length", "scale_factor", "policy"])
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


def plot_scale_factor_trends(
    df: pd.DataFrame,
    metric_name: str,
    term_length: int,
    output_path: str,
    policies: Optional[List[str]] = None,
    ylabel: Optional[str] = None,
    show_std: bool = True,
    scale: float = 0.12,
    ylim: Optional[Tuple[float, float]] = None,
    zoom_margin: Optional[float] = None,
    relative: bool = False,
    relative_policy: str = "null",
):
    """
    Plot trends vs. scale factor for each policy at a given term length.

    Args:
        df: DataFrame from analyze_sensitivity (for one metric)
        metric_name: Name of the metric (for title)
        term_length: Term length value to plot (e.g., 2000)
        output_path: Base path to save files (will save .png and .pgf)
        policies: List of policies to plot (default: all in df)
        ylabel: Y-axis label (default: metric_name)
        show_std: Whether to show confidence interval region (default: True)
        scale: scale of x-axis value to plot (default: 0.12)
        ylim: Y-axis limits as (min, max) tuple (default: None, auto)
        zoom_margin: If set, zoom y-axis to data range with this fraction as margin (e.g., 0.1 = 10%)
        relative: If True, plot difference from relative_policy (default: False)
        relative_policy: Policy to use as baseline for relative plots (default: "null")
    """

    plot_sensitivity_trends(
        df=df,
        metric_name=metric_name,
        filter_key="term_length",
        filter_value=term_length,
        x_key="scale_factor",
        output_path=output_path,
        policies=policies,
        ylabel=ylabel,
        show_std=show_std,
        x_scale=scale,
        figsize=(8, 4),
        ylim=ylim,
        zoom_margin=zoom_margin,
        relative=relative,
        relative_policy=relative_policy,
    )


def plot_all_term_lengths(
    dfs: Dict[str, pd.DataFrame],
    output_dir: str,
    policies: Optional[List[str]] = None,
    show_std: bool = True,
    scale: float = 0.12,
    shared_ylim: bool = True,
    zoom_margin: Optional[float] = None,
    relative: bool = False,
    relative_policy: str = "null",
):
    """
    Plot scale factor trends for all metrics and all term length values.

    Args:
        dfs: Dict of DataFrames from analyze_sensitivity
        output_dir: Directory to save PNG and PGF files
        policies: List of policies to plot (default: all)
        show_std: Whether to show confidence interval region (default: True)
        scale: Scale factor for x-axis values (default: 0.12)
        shared_ylim: Whether to use same y-axis limits for all plots of same metric (default: True)
        zoom_margin: If set, zoom y-axis to data range with this fraction as margin (e.g., 0.1 = 10%)
        relative: If True, plot difference from relative_policy (default: False)
        relative_policy: Policy to use as baseline for relative plots (default: "null")
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, df in dfs.items():
        # Get all term length values
        term_lengths = sorted(df.index.get_level_values("term_length").unique())

        # Compute consistent y-limits for this metric if requested (only for absolute values)
        ylim = (
            compute_metric_ylim(df, show_std) if shared_ylim and not relative else None
        )

        print(f"\nPlotting {metric_name} for {len(term_lengths)} term lengths...")
        if shared_ylim and not relative:
            print(f"  Using shared y-limits: {ylim}")
        if relative:
            print(f"  Plotting relative to '{relative_policy}'")

        for term_length in term_lengths:
            # Create base filename (without extension)
            # Sanitize metric name for filename (replace / with _per_)
            metric_filename = metric_name.replace("/", "_per_")
            filename = f"{metric_filename}_term_length_{term_length}"
            if relative:
                filename += f"_rel_{relative_policy}"
            output_path = os.path.join(output_dir, filename)

            try:
                plot_scale_factor_trends(
                    df=df,
                    metric_name=metric_name,
                    term_length=term_length,
                    output_path=output_path,
                    policies=policies,
                    show_std=show_std,
                    scale=scale,
                    ylim=ylim,
                    zoom_margin=zoom_margin,
                    relative=relative,
                    relative_policy=relative_policy,
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to plot {metric_name} for term_length={term_length}: {e}"
                )
                continue

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Off-Probation Term Length and Scale Factor Sensitivity Analysis")
    print("=" * 60)

    # Configuration
    results_dir = "result-ofpl-scl-4"
    metrics = ["total_offenses", "offense_rate"]
    policies = ["high-risk", "low-risk", "age-first"]
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
            print(f"\nExample: All results for term_length=2000:")
            if 2000 in df.index.get_level_values("term_length"):
                print(df.loc[2000].head())

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()

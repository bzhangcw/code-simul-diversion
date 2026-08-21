"""
Common utilities for sensitivity analysis scripts.

This module provides shared functions for:
- Computing equilibrium values from simulation metrics
- Plotting sensitivity trends across parameter variations
"""

import os
import numpy as np
import pandas as pd
import warnings

from simulation_summary import get_policy_color, get_policy_linestyle


def compute_equilibrium_value(
    policy_metrics: dict,
    metric_key: str,
    num_period_window: int,
    use_first: bool = False,
    start_from: int = 0,
) -> np.ndarray:
    """
    Compute value by summing over a window of periods.

    Args:
        policy_metrics: Dict from read_metrics_from_h5 with 2D arrays
        metric_key: Name of metric to compute (e.g., 'total_offenses')
        num_period_window: Number of periods to sum over
        use_first: If False (default), sum over last num_period_window periods (equilibrium).
                   If True, sum over first num_period_window periods starting from start_from.
        start_from: Starting period index when use_first=True (default: 0).
                    Ignored when use_first=False.

    Returns:
        Array of values, one per repetition (shape: n_reps,)
    """
    if metric_key not in policy_metrics:
        raise ValueError(f"Metric '{metric_key}' not found in policy_metrics")

    data = policy_metrics[metric_key]

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(
            f"Expected 2D array for metric '{metric_key}', got {type(data)} with shape {getattr(data, 'shape', 'N/A')}"
        )

    # Sum over specified window for each repetition
    # data shape: (n_reps, n_episodes)
    if use_first:
        end_idx = start_from + num_period_window
        vals = np.sum(data[:, start_from:end_idx], axis=1)
    else:
        vals = np.sum(data[:, -num_period_window:], axis=1)

    return vals


def plot_sensitivity_trends(
    df: pd.DataFrame,
    metric_name: str,
    filter_key: str,
    filter_value,
    x_key: str,
    output_path: str,
    policies: list = None,
    ylabel: str = None,
    show_std: bool = True,
    x_scale: float = 1.0,
    figsize: tuple = (8, 4),
    xticks: list = None,
    ylim: tuple = None,
    zoom_margin: float = None,
    relative: bool = False,
    relative_policy: str = "null",
):
    """
    Generic function to plot sensitivity trends.

    Args:
        df: DataFrame with MultiIndex containing filter_key, x_key, and 'policy'
        metric_name: Name of the metric (for title)
        filter_key: Index level to filter on (e.g., 'effect', 'term_length')
        filter_value: Value to filter for (e.g., 0.5, 2000)
        x_key: Index level to use as x-axis (e.g., 'capacity', 'scale_factor')
        output_path: Base path to save files (will save .png and .pgf)
        policies: List of policies to plot (default: all in df)
        ylabel: Y-axis label (default: metric_name)
        show_std: Whether to show confidence interval region (default: True)
        x_scale: Scale factor for x-axis values (default: 1.0)
        figsize: Figure size tuple (default: (8, 4))
        xticks: Custom x-axis tick values (default: None, auto)
        ylim: Y-axis limits as (min, max) tuple (default: None, auto)
        zoom_margin: If set, zoom y-axis to data range with this fraction as margin (e.g., 0.1 = 10%)
        relative: If True, plot difference from relative_policy (default: False)
        relative_policy: Policy to use as baseline for relative plots (default: "null")
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter data for the given filter value
    if filter_value not in df.index.get_level_values(filter_key):
        raise ValueError(f"{filter_key}={filter_value} not found in data")

    subset = df.loc[filter_value]

    # Get policies to plot
    if policies is None:
        policies = sorted(subset.index.get_level_values("policy").unique())

    # Get baseline data if relative mode
    baseline_means = None
    if relative:
        if relative_policy not in subset.index.get_level_values("policy"):
            warnings.warn(
                f"Relative policy '{relative_policy}' not found for {filter_key}={filter_value}. "
                "Plotting absolute values instead."
            )
            relative = False
        else:
            baseline_data = subset.xs(relative_policy, level="policy")
            baseline_means = baseline_data["mean"].to_numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each policy
    for i, policy in enumerate(policies):
        if policy not in subset.index.get_level_values("policy"):
            warnings.warn(
                f"Policy '{policy}' not found for {filter_key}={filter_value}"
            )
            continue

        policy_data = subset.xs(policy, level="policy")

        # Get color and linestyle
        color = get_policy_color(policy, fallback_idx=i)
        linestyle = get_policy_linestyle(policy)

        # Get data as numpy arrays
        x_values = policy_data.index.to_numpy() * x_scale
        means = policy_data["mean"].to_numpy()

        # Compute relative difference if requested
        if relative and baseline_means is not None:
            means = means - baseline_means

        # Add confidence interval region if requested
        if show_std and "std" in policy_data.columns:
            stds = policy_data["std"].to_numpy()
            ax.fill_between(
                x_values,
                means - stds,
                means + stds,
                color=color,
                alpha=0.2,
            )

        # Plot line on top of the shaded region
        ax.plot(
            x_values,
            means,
            marker="o",
            label=policy,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
        )

    # Formatting
    ax.grid(True, alpha=0.3)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    elif zoom_margin is not None:
        # Auto-zoom to data range with margin
        y_min, y_max = ax.get_ylim()
        data_range = y_max - y_min
        margin = data_range * zoom_margin
        ax.set_ylim(y_min - margin, y_max + margin)

    # Save figure in both PNG and PGF formats
    plt.tight_layout()

    # Save PNG
    png_path = output_path + ".png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    # Save PGF
    pgf_path = output_path + ".pgf"
    plt.savefig(pgf_path, bbox_inches="tight")

    plt.close()

    print(f"Saved plot to {png_path} and {pgf_path}")

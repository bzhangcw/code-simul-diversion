import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

import simulation
import sirakaya

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Default font for all plots (HTML and PDF)
# ------------------------------------------------------------

# Configure matplotlib font globally
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# ------------------------------------------------------------
# Policy color mapping for consistent colors across figures
# (matches policy names from get_tests() in simulation_setup.py)
# ------------------------------------------------------------
POLICY_COLORS = {
    "null": "#7f7f7f",  # gray
    "random": "#9467bd",  # purple
    # ------------------------------------------------------------
    # priority policies; will exhaust the capacity
    # ------------------------------------------------------------
    "high-risk": "#d62728",  # red
    "low-risk": "#546de5",  # green
    "high-risk-cutoff": "#e377c2",  # pink
    "high-risk-only-young": "#e377c2",  # pink
    "high-risk-lean-young": "#2ca02c",  # green
    "age-tolerance": "#17becf",  # cyan
    "age-first": "#51a3e6",  # soft blue
    "age-first-high-risk": "#e67051",  # soft red
    "high-risk-young-first": "#e377c2",  # soft red
    # ------------------------------------------------------------
    # fluid policies; may not exhaust the capacity
    # ------------------------------------------------------------
    "fluid-low-age-low-prev": "#1f77b4",  # blue
    "fluid-low-age-threshold-offenses": "#2ca02c",  # green
}
POLICY_ALIAS = {
    "age-first": "age-first-low-risk",
}

# Hatch patterns for legend: policies starting with "age-" get dashed pattern
# See matplotlib hatch patterns: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
POLICY_HATCHES = {
    "age-tolerance": "//",
    "age-first": "//",
    "age-first-high-risk": "//",
    "fluid-low-age-low-prev": "//",
    "fluid-low-age-threshold-offenses": "//",
}

# Line styles for plotting series: policies starting with "age-" get dashed lines
# See matplotlib linestyles: '-' (solid), '--' (dashed), '-.' (dashdot), ':' (dotted)
POLICY_LINESTYLES = {
    "age-tolerance": "--",
    "age-first": "--",
    "age-first-high-risk": "--",
    "high-risk-young-first": "--",
    "fluid-low-age-low-prev": "--",
    "fluid-low-age-threshold-offenses": "--",
}


def get_policy_color(policy_name, fallback_idx=0):
    """Get color for a policy name, with fallback to tab10 colors."""
    import matplotlib.pyplot as plt

    if policy_name in POLICY_COLORS:
        return POLICY_COLORS[policy_name]
    # Fallback to tab10 colors
    colors = plt.cm.tab10.colors
    return colors[fallback_idx % len(colors)]


def get_policy_hatch(policy_name):
    """Get hatch pattern for a policy name.

    Returns hatch pattern if defined in POLICY_HATCHES,
    or '//' for policies starting with 'age-', otherwise None (solid fill).
    """
    if policy_name in POLICY_HATCHES:
        return POLICY_HATCHES[policy_name]
    # Default: age- policies get dashed hatch, others get solid fill
    if policy_name.startswith("age-"):
        return "//"
    return None


def get_policy_linestyle(policy_name):
    """Get line style for a policy name.

    Returns linestyle if defined in POLICY_LINESTYLES,
    or '--' for policies starting with 'age-', otherwise '-' (solid line).
    """
    if policy_name in POLICY_LINESTYLES:
        return POLICY_LINESTYLES[policy_name]
    # Default: age- policies get dashed lines, others get solid lines
    if policy_name.startswith("age-"):
        return "--"
    return "-"


# ------------------------------------------------------------
# dump / read metrics to / from h5 file (per-rep structure)
# ------------------------------------------------------------
def dump_rep_metrics(simulator, output_dir, p_freeze):
    """
    Save a single repetition's metrics to {output_dir}/metrics.h5.

    Args:
        simulator: a single simulator object
        output_dir: directory to save metrics.h5 (e.g., results/policy_name/0/)
        p_freeze: p_freeze parameter for summarize_trajectory
        window_for_ratios: window_for_ratios parameter for summarize_trajectory
    """
    import h5py

    simulator.summarize(save=False)
    mean_df, results, results_df, results_flow, results_retention = (
        simulator.summarize_trajectory(
            p_freeze=p_freeze, state_lst_columns=["state_lst"], state_columns=["state"]
        )
    )

    # save the individual and community dataframes
    simulator.dfi.to_excel(f"{output_dir}/dfi.xlsx")
    simulator.df_episode_stats.to_excel(f"{output_dir}/df_episode_stats.xlsx", index=False)
    results_df[-1].to_excel(f"{output_dir}/df_result_last.xlsx")

    metrics = simulation.evaluation_metrics(results_df)

    # Save to HDF5 file with flat structure (one 1D array per metric)
    filepath = f"{output_dir}/metrics.h5"
    with h5py.File(filepath, "w") as f:
        for metric_key, values in metrics.items():
            f.create_dataset(metric_key, data=values)

    print(f"Saved metrics to {filepath}")


def read_metrics_from_h5(name, output_dir="results"):
    """
    Load and aggregate metrics from all reps in {output_dir}/{name}/*/metrics.h5.

    Args:
        name: policy name
        output_dir: base directory containing policy subdirectories

    Returns:
        policy_metrics: dict[metric_key] -> 2D array (repeats x episodes)
        policy_agg: dict[metric_key] -> dict with 'mean', 'max', 'min', 'std'
    """
    import h5py

    policy_dir = f"{output_dir}/{name}"

    # Find all rep directories (numeric subdirectories)
    rep_dirs = sorted(
        [
            d
            for d in os.listdir(policy_dir)
            if os.path.isdir(f"{policy_dir}/{d}") and d.isdigit()
        ],
        key=int,
    )

    if not rep_dirs:
        raise ValueError(f"No repetition directories found in {policy_dir}")

    # Load metrics from each rep
    all_rep_metrics = []
    for rep in rep_dirs:
        filepath = f"{policy_dir}/{rep}/metrics.h5"
        if not os.path.exists(filepath):
            print(f"Warning: metrics.h5 not found for {rep}, skipping")
            continue
        with h5py.File(filepath, "r") as f:
            rep_metrics = {key: f[key][:] for key in f.keys()}
        all_rep_metrics.append(rep_metrics)

    # Stack into matrices and compute aggregates
    if not all_rep_metrics:
        raise ValueError(
            f"No metrics.h5 files found in any repetition directory under {policy_dir}"
        )
    metric_keys = all_rep_metrics[0].keys()
    policy_metrics = {}
    policy_agg = {}

    # Per-state fields are flattened and vary in length - store as list, no aggregates
    # These share the same index structure (state_index_flat / state_lengths)
    per_state_keys = {
        # Shared indices
        "state_index_flat",
        "state_lengths",
        # Treatment data
        "treatment_tau_flat",
        "treatment_tau_rel_flat",
        # Population data
        "population_x0_flat",
        "population_yin_flat",
        "population_inc_flat",
        "population_lv_flat",
        # Legacy names (for backward compatibility)
        "treatment_index_flat",
        "treatment_lengths",
    }

    for metric_key in metric_keys:
        arrays = [m[metric_key] for m in all_rep_metrics]

        if metric_key in per_state_keys:
            # Store as list of arrays (no stacking/aggregation)
            policy_metrics[metric_key] = arrays
            # No aggregation for treatment fields
            continue

        # Find minimum length across all reps (in case of different episode counts)
        min_len = min(len(arr) for arr in arrays)

        # Truncate to minimum length and stack into matrix (n_reps x n_episodes)
        matrix = np.array([arr[:min_len] for arr in arrays])
        policy_metrics[metric_key] = matrix

        # Compute aggregates
        policy_agg[metric_key] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0),
            "min": np.min(matrix, axis=0),
            "max": np.max(matrix, axis=0),
        }

    return policy_metrics, policy_agg


def load_to_metrics(
    all_metrics, agg_metrics, name, output_dir="results", print_tree=False
):
    """
    Load metrics from policy directory and add to existing dicts.

    Args:
        all_metrics: existing all_metrics dict to update
        agg_metrics: existing agg_metrics dict to update
        name: policy name (subdirectory under output_dir)
        output_dir: base directory containing policy subdirectories
        print_tree: if True, print directory tree after loading all policies

    Returns:
        all_metrics, agg_metrics (updated in place, also returned for convenience)

    Example:
        all_metrics, agg_metrics = {}, {}
        load_to_metrics(all_metrics, agg_metrics, "null", "results")
        load_to_metrics(all_metrics, agg_metrics, "random", "results", print_tree=True)
        plot_metric(all_metrics, "total_offenses", start=20)
    """
    import os

    policy_metrics, policy_agg = read_metrics_from_h5(name, output_dir)
    all_metrics[name] = policy_metrics
    agg_metrics[name] = policy_agg

    # Print directory tree if requested (only for this policy)
    if print_tree:
        _print_directory_tree(output_dir, loaded_policies=[name])

    return all_metrics, agg_metrics


def _print_directory_tree(
    directory, prefix="", max_depth=2, current_depth=0, loaded_policies=None
):
    """
    Print directory tree structure, showing only loaded policies.

    Args:
        directory: root directory to print
        prefix: prefix for indentation (used internally for recursion)
        max_depth: maximum depth to traverse
        current_depth: current depth (used internally for recursion)
        loaded_policies: list of policy names that were loaded (only these will be shown)
    """
    import os

    if current_depth == 0:
        print(f"\nDirectory tree for '{directory}':")
        print(f"{directory}/")

    if current_depth >= max_depth:
        return

    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return

    # Separate directories and files
    all_dirs = [
        e
        for e in entries
        if os.path.isdir(os.path.join(directory, e)) and not e.startswith(".")
    ]
    all_files = [
        e
        for e in entries
        if os.path.isfile(os.path.join(directory, e)) and not e.startswith(".")
    ]

    # Only show loaded policies at root level
    if current_depth == 0 and loaded_policies is not None:
        # Show only loaded policy names and their subdirectories
        dirs = [d for d in all_dirs if d in loaded_policies]

        for i, d in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}{d}/")

            # Recurse into this policy directory to show its contents
            extension = "    " if is_last_dir else "│   "
            _print_directory_tree(
                os.path.join(directory, d),
                prefix + extension,
                max_depth,
                current_depth + 1,
                loaded_policies,
            )
    elif current_depth > 0:
        # Inside policy directories, show all subdirectories
        dirs = all_dirs

        for i, d in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}{d}/")

            # Continue recursing
            extension = "    " if is_last_dir else "│   "
            _print_directory_tree(
                os.path.join(directory, d),
                prefix + extension,
                max_depth,
                current_depth + 1,
                loaded_policies,
            )


# ------------------------------------------------------------
# add metrics to existing dicts, not from h5 file
# ------------------------------------------------------------
def add_to_metrics(all_metrics, agg_metrics, k, simulators, p_freeze, windowsize=20):
    """
    Add a new policy's metrics to existing all_metrics and agg_metrics dicts.

    Args:
        all_metrics: existing all_metrics dict to update
        agg_metrics: existing agg_metrics dict to update
        k: name/key for the new policy
        simulators: list of simulators (or single simulator) for the new policy
        p_freeze: p_freeze parameter for summarize_trajectory

    Returns:
        all_metrics, agg_metrics (updated in place, also returned for convenience)
    """
    # Handle both single simulator and list of simulators
    if not isinstance(simulators, list):
        simulators = [simulators]

    # Collect metrics from each repeat
    repeat_metrics = []
    for simulator in simulators:
        simulator.summarize(save=False)
        mean_df, results, results_df, results_flow, results_retention = (
            simulator.summarize_trajectory(
                p_freeze=p_freeze,
                state_lst_columns=["state_lst"],
                state_columns=["state"],
            )
        )
        repeat_metrics.append(simulation.evaluation_metrics(results_df))

    # Convert to matrix form: each metric becomes (n_repeats x n_episodes)
    metric_keys = repeat_metrics[0].keys()
    all_metrics[k] = {}
    agg_metrics[k] = {}

    for metric_key in metric_keys:
        # Stack all repeats into a matrix
        matrix = np.array([m[metric_key] for m in repeat_metrics])
        all_metrics[k][metric_key] = matrix

        # Compute aggregates across repeats (axis=0)
        agg_metrics[k][metric_key] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0),
            "min": np.min(matrix, axis=0),
            "max": np.max(matrix, axis=0),
        }

    return all_metrics, agg_metrics


# ------------------------------------------------------------
# plot metrics from all_metrics, which is a dict of results
# ------------------------------------------------------------
def _compute_rolling_sum_per_rep(data_2d, num_period_window):
    """
    Compute rolling sum over window for each rep.

    Args:
        data_2d: 2D array (n_reps, n_episodes)
        num_period_window: window size for rolling sum

    Returns:
        2D array (n_reps, n_episodes) with rolling sums
    """
    n_reps, n_episodes = data_2d.shape
    cumsum = np.cumsum(data_2d, axis=1)
    result = np.zeros_like(data_2d)
    for i in range(n_episodes):
        if i < num_period_window:
            result[:, i] = cumsum[:, i]
        else:
            result[:, i] = cumsum[:, i] - cumsum[:, i - num_period_window]
    return result


def _compute_cumulative_mean_per_rep(data_2d):
    """
    Compute cumulative mean for each rep.

    Args:
        data_2d: 2D array (n_reps, n_episodes)

    Returns:
        2D array (n_reps, n_episodes) with cumulative means
    """
    cumsum = np.cumsum(data_2d, axis=1)
    divisor = np.arange(1, data_2d.shape[1] + 1)
    return cumsum / divisor


def plot_metric(
    metrics_dict,
    metric_key,
    ylabel=None,
    title=None,
    show_ci=True,
    start=20,
    output_path=None,
    num_period_window=None,
    relative=False,
    relative_policy="null",
):
    """
    Plot a single metric for all simulations with confidence intervals.
    Uses raw per-rep data for consistent statistics with equilibrium computation.
    Computes rolling sum over last num_period_window periods per rep, then aggregates.
    If num_period_window is None, computes cumulative mean per rep, then aggregates.

    Args:
        metrics_dict: all_metrics from collect_metrics() (dict with 2D arrays per metric)
        metric_key: e.g. 'total_population', 'total_offenses', 'total_enrollment',
                    'total_incarcerated', 'total_returns', 'offense_rate', etc.
        ylabel: optional y-axis label (defaults to metric_key)
        title: optional plot title
        show_ci: whether to show confidence interval (mean ± 2*std band)
        start: starting episode index (skip first N episodes)
        output_path: path to save PDF (default: /tmp/{metric_key}-trend.pdf)
        num_period_window: window size for rolling sum (if None, uses cumulative mean)
        relative: if True, plot values relative to relative_policy (default: False)
        relative_policy: policy name to use as baseline for relative plots (default: "null")
    """
    fig = go.Figure()
    import matplotlib.colors as mcolors

    # Compute baseline values if relative mode is enabled (per-rep)
    baseline_per_rep = None
    if relative:
        if relative_policy not in metrics_dict:
            print(
                f"Warning: relative_policy '{relative_policy}' not found in metrics_dict. "
                "Plotting absolute values instead."
            )
            relative = False
        elif metric_key not in metrics_dict[relative_policy]:
            print(
                f"Warning: metric_key '{metric_key}' not found for relative_policy '{relative_policy}'. "
                "Plotting absolute values instead."
            )
            relative = False
        else:
            baseline_data = metrics_dict[relative_policy][metric_key]
            # Check if it's 2D array (raw per-rep data) or dict (agg_metrics)
            if isinstance(baseline_data, np.ndarray) and baseline_data.ndim == 2:
                if num_period_window is not None:
                    baseline_per_rep = _compute_rolling_sum_per_rep(
                        baseline_data, num_period_window
                    )
                else:
                    baseline_per_rep = _compute_cumulative_mean_per_rep(baseline_data)
            else:
                print(
                    f"Warning: Expected 2D array for relative_policy '{relative_policy}'. "
                    "Plotting absolute values instead."
                )
                relative = False

    for i, (k, metrics) in enumerate(metrics_dict.items()):
        if metric_key not in metrics:
            continue

        data = metrics[metric_key]

        # Check if it's 2D array (raw per-rep data) or dict (agg_metrics - legacy)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # Raw per-rep data: compute rolling sum/cumulative mean per rep, then aggregate
            if num_period_window is not None:
                processed = _compute_rolling_sum_per_rep(data, num_period_window)
            else:
                processed = _compute_cumulative_mean_per_rep(data)

            # Subtract baseline if relative mode is enabled (per-rep subtraction)
            if relative and baseline_per_rep is not None:
                # Ensure shapes match (both rows/reps and columns/time)
                min_reps = min(processed.shape[0], baseline_per_rep.shape[0])
                min_len = min(processed.shape[1], baseline_per_rep.shape[1])
                processed = (
                    processed[:min_reps, :min_len]
                    - baseline_per_rep[:min_reps, :min_len]
                )

            # Aggregate across reps
            raw_mean = np.mean(processed, axis=0)
            raw_std = np.std(processed, axis=0)
        elif isinstance(data, dict) and "mean" in data:
            # Legacy agg_metrics format - keep for backward compatibility
            print(
                f"Warning: {k} uses legacy agg_metrics format. Consider using all_metrics."
            )
            if num_period_window is not None:
                cumsum = np.cumsum(data["mean"])
                raw_mean = np.array(
                    [
                        (
                            cumsum[i]
                            if i < num_period_window
                            else cumsum[i] - cumsum[i - num_period_window]
                        )
                        for i in range(len(data["mean"]))
                    ]
                )
                raw_std = data["std"] * np.sqrt(
                    np.minimum(np.arange(1, len(data["std"]) + 1), num_period_window)
                )
            else:
                raw_mean = np.cumsum(data["mean"]) / np.arange(1, len(data["mean"]) + 1)
                raw_std = data["std"] / np.sqrt(np.arange(1, len(data["std"]) + 1))

            if relative and baseline_per_rep is not None:
                baseline_mean = np.mean(baseline_per_rep, axis=0)
                min_len = min(len(raw_mean), len(baseline_mean))
                raw_mean = raw_mean[:min_len] - baseline_mean[:min_len]
        else:
            print(f"Skipping {k}: expected 2D array or dict with 'mean' and 'std'")
            continue

        # Use same policy colors as box plots
        color_raw = get_policy_color(k, fallback_idx=i)
        # Convert to hex for Plotly
        if isinstance(color_raw, str) and color_raw.startswith("#"):
            color = color_raw
        else:
            color = mcolors.to_hex(color_raw[:3])

        mean_vals = raw_mean[start:]
        std_vals = raw_std[start:]
        min_vals = mean_vals - 2 * std_vals
        max_vals = mean_vals + 2 * std_vals

        # Episodes start from (start)
        episodes = np.arange(start, start + len(mean_vals))

        # Convert color to rgba with alpha
        rgb = pc.hex_to_rgb(color) if color.startswith("#") else pc.unlabel_rgb(color)
        fill_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)"

        # Add confidence interval (min-max band)
        if show_ci and min_vals is not None and max_vals is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([episodes, episodes[::-1]]),
                    y=np.concatenate([max_vals, min_vals[::-1]]),
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    legendgroup=k,
                    name=f"{k} (range)",
                    hoverinfo="skip",
                )
            )

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=mean_vals,
                mode="lines",
                name=k,
                legendgroup=k,
                line=dict(color=color),
            )
        )

    fig.update_layout(
        title=title or metric_key,
        xaxis_title="episode",
        yaxis_title=None,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="grey",
            mirror=False,
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="grey",
            mirror=False,
        ),
    )

    # Save to PDF with no legend and no title
    if output_path is None:
        output_path = f"/tmp/{metric_key}-trend.pdf"
    # Create a copy for saving with modifications
    fig_save = go.Figure(fig)
    fig_save.update_layout(
        showlegend=False,
        title=None,
    )
    fig_save.write_image(output_path)
    print(f"Saved to {output_path}")

    fig.show()
    return fig


def plot_policy_legend(
    policy_names=None,
    output_path="/tmp/legend.pdf",
    ncol=None,
    nrows=1,
    orientation="horizontal",
    figsize=None,
    alpha=0.6,
    columnspacing=1.0,
    style="line",
    linewidth=6,
    pad=0.1,
):
    """
    Create a standalone legend figure showing policy colors.

    Args:
        policy_names: list of policy names to include (default: all in POLICY_COLORS)
        output_path: path to save the file (default "/tmp/legend.pdf")
                     Supports .pdf, .png, .pgf, or .tex extensions
        ncol: number of columns in legend (default: auto based on orientation and nrows)
        nrows: number of rows to display legend items (default: 1)
        orientation: "horizontal" or "vertical"
        figsize: tuple (width, height) in inches (default: auto)
        alpha: transparency for patches (default 0.6 to match box plots)
        columnspacing: space between columns in legend (default: 1.0, smaller = closer)
        style: "line" for line series legends, "patch" for box plot legends (default: "line")
        linewidth: line width for line style legends (default: 2)
        pad: padding around the legend (default: 0.1, smaller = tighter)

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.colors as mcolors

    if policy_names is None:
        policy_names = list(POLICY_COLORS.keys())

    # Create legend handles
    handles = []
    for i, name in enumerate(policy_names):
        if name in POLICY_ALIAS:
            real_name = POLICY_ALIAS[name]
        else:
            real_name = name
        color = get_policy_color(name, fallback_idx=i)

        if style == "line":
            # Line style for series plots
            linestyle = get_policy_linestyle(name)
            line = mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=real_name,
            )
            handles.append(line)
        else:
            # Patch style for box plots
            hatch = get_policy_hatch(name)
            # Convert to RGBA with alpha
            if isinstance(color, str) and color.startswith("#"):
                rgb = mcolors.hex2color(color)
                facecolor = (*rgb, alpha)
            else:
                facecolor = (*color[:3], alpha)
            patch = mpatches.Patch(
                facecolor=facecolor, edgecolor=color, hatch=hatch, label=name
            )
            handles.append(patch)

    # Determine layout
    if ncol is None:
        if orientation == "horizontal":
            # Distribute items across nrows
            ncol = (len(policy_names) + nrows - 1) // nrows  # Ceiling division
        else:
            ncol = 1

    if figsize is None:
        if orientation == "horizontal":
            figsize = (min(len(policy_names) * 1.5, 10), 0.3 * nrows)
        else:
            figsize = (2, len(policy_names) * 0.3)

    # Create figure with just the legend
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Use global font with size 10 for legend
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        columnspacing=columnspacing,
    )

    plt.tight_layout(pad=pad)

    # Determine output format from extension
    if output_path.endswith((".pgf", ".tex")):
        # Save as PGF/TikZ for LaTeX
        plt.savefig(output_path, bbox_inches="tight", backend="pgf")
    else:
        # Save as PDF or PNG
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.show()

    print(f"Saved legend to {output_path}")
    return fig


# ------------------------------------------------------------
# main function to produce figures for the papers.
# ------------------------------------------------------------


def plot_to_tex_equilibrium_box(
    metrics_dict,
    metric_key,
    ylabel="value",
    title=None,
    output_path="/tmp/fig.pdf",
    show_labels=False,
    num_period_window=None,
):
    """
    Plot the equilibrium box plot for the given metric.
    Computes sum over last num_period_window periods: sum_{t=T-W+1}^{T} x_t for each repetition
    If num_period_window is None, computes mean over ALL episodes: (1/T) * sum_{t=1}^{T} x_t

    Args:
        metrics_dict: policy_metrics from read_metrics_from_h5() (dict with 2D arrays: repeats x episodes)
        metric_key: e.g. 'total_population', 'total_offenses', 'total_enrollment',
                    'total_incarcerated', 'total_returns', 'offense_rate', etc.
        ylabel: optional y-axis label (defaults to metric_key)
        title: optional plot title
        output_path: path to save the file (default "/tmp/fig.pdf")
                     Supports .pdf, .png, .pgf, or .tex extensions
        show_labels: if True, show policy names on x-axis (default False)
        num_period_window: window size for sum over last periods (if None, uses mean over all)
    """

    # Collect data for box plot: list of (policy_name, flattened_values)
    box_data = []
    labels = []

    for k, metrics in metrics_dict.items():
        if metric_key not in metrics:
            continue

        data = metrics[metric_key]
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            print(f"Skipping {k}: expected 2D array format (repeats x episodes)")
            continue

        # Compute sum over last num_period_window or mean over ALL episodes for each repetition
        if num_period_window is not None:
            # Sum over last num_period_window periods for each repetition
            vals = np.sum(data[:, -num_period_window:], axis=1)
        else:
            # Mean over ALL episodes for each repetition
            vals = np.mean(data, axis=1)
        box_data.append(vals)
        labels.append(k)

    if not box_data:
        print(f"No data found for metric '{metric_key}'")
        return None

    # Create box plot
    fig, ax = plt.subplots(
        figsize=(max(6, len(labels) * 1.2), 3.5),
    )

    # Use empty labels if show_labels is False
    display_labels = labels if show_labels else [""] * len(labels)
    bp = ax.boxplot(box_data, labels=display_labels, patch_artist=True)

    # Style the box plot using policy colors and add mean values
    for i, (box, median, label) in enumerate(zip(bp["boxes"], bp["medians"], labels)):
        color = get_policy_color(label, fallback_idx=i)
        # Convert hex to rgba with alpha
        if isinstance(color, str) and color.startswith("#"):
            rgb = mcolors.hex2color(color)
            box.set_facecolor((*rgb, 0.6))
            box.set_edgecolor(color)
        else:
            box.set_facecolor((*color[:3], 0.6))
            box.set_edgecolor(color)
        median.set_color("black")
        median.set_linewidth(1.5)

    # Set font for axis labels and ticks
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Rotate labels if many policies and labels are shown
    if show_labels and len(labels) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Determine output format from extension
    if output_path.endswith((".pgf", ".tex")):
        # Save as PGF/TikZ for LaTeX
        plt.savefig(output_path, bbox_inches="tight", backend="pgf")
    else:
        # Save as raster/vector format (PDF, PNG, etc.)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.show()

    print(f"Saved to {output_path}")
    return fig


def plot_state_distribution(
    all_metrics,
    settings,
    selected_keys=None,
    metric_key="total_departures",
    score_cap=10,
    score_filter=None,
    median_score=None,
    output_path="/tmp/score_cdf.pgf",
    figsize=(10, 5),
    show=True,
):
    """
    Plot cumulative distribution of population by risk score across policies.

    Args:
        all_metrics: dict of policy_name -> metrics from read_metrics_from_h5
        settings: SimulationSetup instance with dfi containing score_state
        selected_keys: list of policy names to plot (default: ["high-risk", "low-risk"])
        metric_key: metric to use for population (default: "total_departures")
        score_cap: maximum score value to cap at (default: 10)
        score_filter: "high" for score >= median_score, "low" for score < median_score, None for all
        median_score: score threshold when using score_filter
        output_path: path to save figure (default: "/tmp/score_cdf.pgf")
        figsize: figure size tuple (default: (10, 5))
        show: whether to call plt.show() (default: True)

    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from simulation_stats import compute_equilibrium_stats

    if selected_keys is None:
        selected_keys = ["high-risk", "low-risk"]

    if score_filter is not None and median_score is None:
        raise ValueError("median_score must be provided when score_filter is set")

    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
        }
    )

    # Build the score function from risk table
    risk_table = settings.dfi.groupby(["age_dist", "offenses"])["score_state"].mean()
    r_a_dict = risk_table[:, 0].to_dict()
    score_aj = lambda a, j: r_a_dict.get(a, 0) + 0.1884 * j

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    markers = ["o", "s", "^", "x", "D", "v", "<", ">"]
    markevery = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24]

    # First pass: collect all unique scores across all policies
    all_scores = set()
    all_data = {}

    for i, policy_key in enumerate(selected_keys):
        df_mean_raw, _ = compute_equilibrium_stats(
            all_metrics[policy_key], metric_key=metric_key
        )

        score_pop = {}
        for a in df_mean_raw.index:
            for j in df_mean_raw.columns:
                raw_score = score_aj(a, j)

                # Apply score filter
                if score_filter == "high" and raw_score < median_score:
                    continue
                elif score_filter == "low" and raw_score >= median_score:
                    continue

                pop = df_mean_raw.loc[a, j]
                score = min(raw_score, score_cap)
                if score in score_pop:
                    score_pop[score] += pop
                else:
                    score_pop[score] = pop

        all_scores.update(score_pop.keys())
        all_data[policy_key] = score_pop

    # Create score-to-rank mapping (evenly spaced x-axis)
    sorted_all_scores = sorted(all_scores)
    score_to_rank = {s: i for i, s in enumerate(sorted_all_scores)}

    # Second pass: plot with rank-based x-axis
    for i, policy_key in enumerate(selected_keys):
        score_pop = all_data[policy_key]

        sorted_scores = np.array(sorted(score_pop.keys()))
        sorted_pops = np.array([score_pop[s] for s in sorted_scores])

        cumsum_pops = np.cumsum(sorted_pops)
        total_pop = cumsum_pops[-1]

        x_ranks = np.array([score_to_rank[s] for s in sorted_scores])

        color = get_policy_color(policy_key, fallback_idx=i)
        linestyle = get_policy_linestyle(policy_key)

        ax.plot(
            x_ranks,
            cumsum_pops / total_pop,
            color=color,
            linestyle=linestyle,
            linewidth=3.5,
            marker=markers[i % len(markers)],
            markersize=10,
            markevery=markevery[i % len(markevery)],
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.5,
        )

    # Set x-tick labels to show actual scores
    n_ticks = min(10, len(sorted_all_scores))
    tick_indices = np.linspace(0, len(sorted_all_scores) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [f"{sorted_all_scores[i]:.2f}" for i in tick_indices], fontsize=15
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative Population")
    ax.tick_params(axis="both", labelsize=15, width=2)
    ax.grid(True, alpha=0.3, linewidth=1.5)

    # Make spines bolder
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()

    return fig, ax


# ------------------------------------------------------------
# Compute metric by risk group H & L and plot them as mirrored bars
# ------------------------------------------------------------
def compute_metric_by_risk_group(
    all_metrics,
    settings,
    median_score,
    metric_key="population_start",
    selected_keys=None,
):
    """
    Compute total metric values split by high-risk vs low-risk groups.

    For each policy, sums the metric values for individuals with score below
    (low-risk) or above/equal to (high-risk) the median score.

    Args:
        all_metrics: dict of policy_name -> metrics from read_metrics_from_h5
        settings: SimulationSetup instance with dfi containing score_state
        median_score: score threshold to split high-risk vs low-risk
        metric_key: metric to compute (e.g., "population_start", "total_offenses", "total_departures", "total_enrollment", "total_incarcerated")
        selected_keys: list of policy names to compute (default: all keys in all_metrics)

    Returns:
        dict: {policy_name: {"low_risk": total_value, "high_risk": total_value}}
    """
    from simulation_stats import compute_equilibrium_stats

    if selected_keys is None:
        selected_keys = list(all_metrics.keys())

    # Build the score function from risk table
    risk_table = settings.dfi.groupby(["age_dist", "offenses"])["score_state"].mean()
    r_a_dict = risk_table[:, 0].to_dict()
    score_aj = lambda a, j: r_a_dict.get(a, 0) + 0.1884 * j

    results = {}

    for policy_key in selected_keys:
        if policy_key not in all_metrics:
            continue

        df_mean_raw, _ = compute_equilibrium_stats(
            all_metrics[policy_key], metric_key=metric_key
        )

        low_risk_total = 0.0
        high_risk_total = 0.0

        for a in df_mean_raw.index:
            for j in df_mean_raw.columns:
                val = df_mean_raw.loc[a, j]
                score = score_aj(a, j)

                if score < median_score:
                    low_risk_total += val
                else:
                    high_risk_total += val

        results[policy_key] = {
            "low_risk": low_risk_total,
            "high_risk": high_risk_total,
        }

    return results


def plot_risk_group_comparison(
    all_metrics,
    settings,
    median_score,
    selected_keys=None,
    output_path="/tmp/risk_group_comparison.pgf",
    figsize=(10, 5),
    show=True,
):
    """
    Plot population and offenses by risk group as mirrored bars.

    Population bars go upward, offense bars go downward (reflection style).
    Absolute values annotated on top/bottom, rates at center.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if selected_keys is None:
        selected_keys = list(all_metrics.keys())

    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
        }
    )

    # Compute the totals
    rxx = compute_metric_by_risk_group(
        all_metrics, settings, median_score, "population_start", selected_keys
    )
    ryy = compute_metric_by_risk_group(
        all_metrics, settings, median_score, "total_offenses", selected_keys
    )

    policies = [p for p in selected_keys if p in rxx and p in ryy]

    # Extract data
    x_low = np.array([rxx[p]["low_risk"] for p in policies], dtype=float)
    x_high = np.array([rxx[p]["high_risk"] for p in policies], dtype=float)
    y_low = np.array([ryy[p]["low_risk"] for p in policies], dtype=float)
    y_high = np.array([ryy[p]["high_risk"] for p in policies], dtype=float)

    # Compute rates (safe)
    rate_low = np.divide(y_low, x_low, out=np.zeros_like(y_low), where=x_low > 0)
    rate_high = np.divide(y_high, x_high, out=np.zeros_like(y_high), where=x_high > 0)

    # --- CENTERED BINS: define group centers and symmetric offsets ---
    group_pos = np.arange(len(policies))
    width = 0.35
    offset = width / 2
    pos_low = group_pos - offset
    pos_high = group_pos + offset

    fig, ax = plt.subplots(figsize=figsize)

    color_low = "#4ECDC4"  # Teal for low-risk
    color_high = "#FF6B6B"  # Red for high-risk

    # Scale offenses to match population range for visual symmetry
    max_pop = float(max(x_low.max(initial=0.0), x_high.max(initial=0.0)))
    max_off = float(max(y_low.max(initial=0.0), y_high.max(initial=0.0)))
    scale_factor = (max_pop / max_off) if max_off > 0 else 1.0

    y_low_scaled = y_low * scale_factor
    y_high_scaled = y_high * scale_factor

    # Population bars (upward) and Offense bars (downward, scaled)
    ax.bar(
        pos_low,
        x_low,
        width,
        label="Population (L)",
        color=color_low,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        pos_low,
        -y_low_scaled,
        width,
        label="Offenses (L)",
        color=color_low,
        alpha=1.0,
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )
    ax.bar(
        pos_high,
        x_high,
        width,
        label="Population (H)",
        color=color_high,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        pos_high,
        -y_high_scaled,
        width,
        label="Offenses (H)",
        color=color_high,
        alpha=1.0,
        edgecolor="black",
        linewidth=1,
        hatch="//",
    )

    # Rate at center (just above y=0)
    for i in range(len(policies)):
        ax.annotate(
            f"{rate_low[i]:.3f}",
            xy=(pos_low[i], 0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
        ax.annotate(
            f"{rate_high[i]:.3f}",
            xy=(pos_high[i], 0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.axhline(y=0, color="black", linewidth=1.5)

    # --- ticks centered on the policy bin ---
    ax.set_xticks(group_pos)
    ax.set_xticklabels([f"{p} policy" for p in policies], fontsize=16)

    # Tighten x-axis to reduce gaps between policies
    ax.set_xlim(group_pos[0] - 0.5, group_pos[-1] + 0.5)

    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))

    # Legend on top outside
    ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    # Print rates to stdout
    rate_overall = np.divide(
        (y_low + y_high),
        (x_low + x_high),
        out=np.zeros_like(y_low),
        where=(x_low + x_high) > 0,
    )
    print(
        f"\n{'Policy':<20} {'Low-Risk Rate':<15} {'High-Risk Rate':<15} {'Overall Rate':<15}"
    )
    print("-" * 65)
    for i, p in enumerate(policies):
        print(
            f"{p:<20} {rate_low[i]:<15.4f} {rate_high[i]:<15.4f} {rate_overall[i]:<15.4f}"
        )
    print()
    return fig, ax

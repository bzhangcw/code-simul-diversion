#!/usr/bin/env python3
"""
Script to compare sensitivity analysis results across different simulation configurations.

This script combines CSV files from baseline and cap-ef folders, then computes
relative differences compared to a selected baseline configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# GLOBAL CONFIGURATION - Modify these to change comparison settings
# =============================================================================

# Base directory containing all simulation folders
BASE_DIR = Path(__file__).parent

# Folders to include in the comparison (relative to BASE_DIR)
# Each folder should have sensitivity_analysis/{first,equilibrium}/ subfolders
# Format: dict mapping folder name to default values for missing columns
# - baseline uses sensanaly_prscale (has term_length, scale_factor) -> need effect, capacity
# - cap-ef uses sensanaly_capeff (has effect, capacity) -> need term_length, scale_factor
FOLDERS = {
    "baseline": {
        "effect": 0.71,  # No treatment effect in baseline
        "capacity": 80,  # No treatment capacity in baseline
    },
    "result-ofpl-scl-0.71-more-people": {
        "effect": 0.71,  # No treatment effect in baseline
        "capacity": 80,  # No treatment capacity in baseline
    },
    "result-ofpl-scl-0.71-less-people": {
        "effect": 0.71,  # No treatment effect in baseline
        "capacity": 80,  # No treatment capacity in baseline
    },
    "result-ofpl-scl-0.71-younger": {
        "effect": 0.71,  # No treatment effect in baseline
        "capacity": 80,  # No treatment capacity in baseline
    },
    "result-cap-ef-0.2": {
        "term_length": 1000,  # Fixed term length for this experiment
        "scale_factor": 0.2,  # Scale factor from folder name
    },
    "result-cap-ef-0.7": {
        "term_length": 1000,  # Fixed term length for this experiment
        "scale_factor": 0.7,  # Scale factor from folder name
    },
}

# Scopes to include (subfolders under sensitivity_analysis/)
SCOPES = ["first", "equilibrium"]

# Metric to analyze: "offense_rate" or "incarceration_rate"
METRIC = "offense_rate"

# Baseline policy is "null" (empty string in CSV) - all other policies compared relative to it
# In the CSV, null policy has empty/NaN in the policy column
BASELINE_POLICY = None  # Represents null/empty policy

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_and_tag_csv(
    folder_path: Path, folder_name: str, scope: str, defaults: dict
) -> pd.DataFrame:
    """Load a sensitivity CSV file and add folder/scope tags."""
    csv_path = folder_path / f"sensitivity_{METRIC}.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping...")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["folder"] = folder_name
    df["scope"] = scope

    # Ensure all four parameter columns exist and fill in defaults
    all_param_cols = ["term_length", "scale_factor", "effect", "capacity"]
    for col in all_param_cols:
        if col not in df.columns:
            # Column doesn't exist, add it with default value
            df[col] = defaults.get(col, None)
        elif df[col].isna().all():
            # Column exists but all NaN, fill with default
            df[col] = defaults.get(col, None)

    return df


def combine_all_csvs() -> pd.DataFrame:
    """Combine all sensitivity CSV files into one DataFrame."""
    all_dfs = []

    for folder_name, defaults in FOLDERS.items():
        for scope in SCOPES:
            folder_path = BASE_DIR / folder_name / "sensitivity_analysis" / scope

            df = load_and_tag_csv(folder_path, folder_name, scope, defaults)
            if not df.empty:
                all_dfs.append(df)
                print(f"Loaded {len(df)} rows from {folder_name}/{scope}")

    if not all_dfs:
        raise ValueError("No CSV files were loaded!")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns: folder, scope, then all 4 parameter columns, then rest
    priority_cols = [
        "folder",
        "scope",
        "term_length",
        "scale_factor",
        "effect",
        "capacity",
        "policy",
    ]
    other_cols = [c for c in combined.columns if c not in priority_cols]
    combined = combined[priority_cols + other_cols]

    print(f"\nTotal combined rows: {len(combined)}")
    return combined


def compute_relative_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute relative difference from null policy for each group.

    For baseline: group by (folder, scope, term_length, scale_factor)
    For cap-ef: group by (folder, scope, effect, capacity)

    Each policy's mean is compared to the null policy mean in the same group.
    """
    df = df.copy()
    df["rel_diff_mean"] = np.nan
    df["abs_diff_mean"] = np.nan

    # Determine grouping columns based on what's available
    # baseline has: term_length, scale_factor
    # cap-ef has: effect, capacity

    for folder in df["folder"].unique():
        for scope in df["scope"].unique():
            subset_mask = (df["folder"] == folder) & (df["scope"] == scope)
            subset = df[subset_mask]

            if subset.empty:
                continue

            # Determine parameter columns
            if "term_length" in subset.columns and subset["term_length"].notna().any():
                param_cols = ["term_length", "scale_factor"]
            elif "effect" in subset.columns and subset["effect"].notna().any():
                param_cols = ["effect", "capacity"]
            else:
                print(
                    f"Warning: Cannot determine parameter columns for {folder}/{scope}"
                )
                continue

            # Get unique parameter combinations
            param_combos = subset[param_cols].drop_duplicates()

            for _, params in param_combos.iterrows():
                # Build mask for this parameter combination
                param_mask = subset_mask.copy()
                for col in param_cols:
                    param_mask = param_mask & (df[col] == params[col])

                # Get null policy reference (empty/NaN policy column)
                null_mask = param_mask & (df["policy"].isna() | (df["policy"] == ""))
                null_rows = df[null_mask]

                if null_rows.empty:
                    continue

                null_mean = null_rows.iloc[0]["mean"]

                if null_mean == 0:
                    continue

                # Compute relative difference for all policies in this group
                df.loc[param_mask, "rel_diff_mean"] = (
                    (df.loc[param_mask, "mean"] - null_mean) / null_mean * 100
                )
                df.loc[param_mask, "abs_diff_mean"] = (
                    df.loc[param_mask, "mean"] - null_mean
                )

    return df


def summarize_by_folder(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics grouped by folder, scope, and policy."""
    # Exclude null policy from summary (it's the baseline)
    df_filtered = df[df["policy"].notna() & (df["policy"] != "")]

    summary = (
        df_filtered.groupby(["folder", "scope", "policy"])
        .agg(
            {
                "mean": ["mean", "min", "max", "std"],
                "rel_diff_mean": ["mean", "min", "max"],
            }
        )
        .round(4)
    )

    return summary


def print_comparison_table(df: pd.DataFrame):
    """Print a formatted comparison table."""
    # Exclude null policy from display
    df_display = df[df["policy"].notna() & (df["policy"] != "")]

    # Group by folder and scope
    for folder in df_display["folder"].unique():
        folder_df = df_display[df_display["folder"] == folder]

        print(f"\n{'='*70}")
        print(f"FOLDER: {folder}")
        print(f"{'='*70}")

        for scope in folder_df["scope"].unique():
            scope_df = folder_df[folder_df["scope"] == scope]

            print(f"\n  SCOPE: {scope}")
            print(f"  {'-'*60}")

            # Group by policy (handle NaN values)
            policies = scope_df["policy"].dropna().unique()
            for policy in sorted(policies):
                policy_df = scope_df[scope_df["policy"] == policy]

                print(f"\n    Policy: {policy}")

                # Show range of values
                mean_values = policy_df["mean"]
                rel_diff = policy_df["rel_diff_mean"].dropna()

                print(f"      Mean {METRIC}:")
                print(
                    f"        Range: [{mean_values.min():.4f}, {mean_values.max():.4f}]"
                )
                print(f"        Average: {mean_values.mean():.4f}")
                if not rel_diff.empty:
                    print(f"      Relative to null:")
                    print(
                        f"        Range: [{rel_diff.min():+.2f}%, {rel_diff.max():+.2f}%]"
                    )
                    print(f"        Average: {rel_diff.mean():+.2f}%")


def main():
    """Main function to run the comparison."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS COMPARISON")
    print(f"Metric: {METRIC}")
    print("Baseline policy: null (empty policy column)")
    print("=" * 70)

    # Step 1: Combine all CSVs
    print("\n[1] Loading and combining CSV files...")
    combined_df = combine_all_csvs()

    # Step 2: Save combined CSV
    output_path = BASE_DIR / f"combined_sensitivity_{METRIC}.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined CSV saved to: {output_path}")

    # Step 3: Compute relative differences (relative to null policy in each group)
    print("\n[2] Computing relative differences (vs null policy)...")
    compared_df = compute_relative_difference(combined_df)

    # Step 4: Save compared results
    compared_output_path = BASE_DIR / f"compared_sensitivity_{METRIC}.csv"
    compared_df.to_csv(compared_output_path, index=False)
    print(f"Compared CSV saved to: {compared_output_path}")

    # Step 5: Print summary
    print("\n[3] Comparison Summary:")
    print_comparison_table(compared_df)

    # Step 6: Summary statistics
    print("\n\n[4] Summary Statistics by Folder, Scope, and Policy:")
    print("-" * 70)
    summary = summarize_by_folder(compared_df)
    print(summary.to_string())

    return combined_df, compared_df


if __name__ == "__main__":
    combined_df, compared_df = main()

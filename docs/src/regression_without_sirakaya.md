# Regression without Sirakaya's estimates

This optional workflow re-estimates the proportional-hazards model directly
from the imputed covariates. It corresponds to the **Fit the Cox model from
scratch** section of `./simulation/py-sirakaya-fitting.ipynb`.

The fit does not use the hard-coded coefficient arrays in `sirakaya.py`, the
precomputed `score_fixed`, `score_comm`, `score_age_dist`, `offset`, or
`score` columns. The categorical level counts below describe how variables are
coded; they are not fitted or published coefficient values.

This page is optional. The main simulation continues to use its existing model
unless its implementation is changed explicitly after reviewing these
estimates.

## 1. Start Jupyter

Complete the [environment setup](index.md#environment-setup), activate the
virtual environment, and start Jupyter from `./simulation/`:

```bash
source ./.venv/bin/activate
cd ./simulation
python -m jupyter lab py-sirakaya-fitting.ipynb
```

Run the cells below in a new notebook or replace the notebook's from-scratch
Cox section with them.

## 2. Cell 1 — imports and output directory

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

import input

RESULTS_DIR = Path("results/regression-without-sirakaya")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

Notice that this cell does not import `sirakaya`.

## 3. Cell 2 — load covariates and recompute county statistics

`input.load_data()` returns several prepared tables. This workflow uses only
the imputed covariate table, `df`, and recomputes the two county statistics
needed by the Cox formula.

```python
_, df, _, _ = input.load_data()

county_statistics = (
    df.groupby("code_county")
    .agg(
        mean_re_time=("time", "mean"),
        percent_re=("observed", "mean"),
    )
)

print("imputed rows:", len(df))
print("counties:", len(county_statistics))
display(county_statistics.head())
```

Although `input.load_data()` also constructs score columns for the standard
simulation workflow, none of those columns enter the model below.

## 4. Cell 3 — assemble the modeling frame

Each categorical variable is rounded after imputation, restricted to its
documented level range, and marked as categorical. Level 1 becomes the
reference category. County percentages and the recomputed county statistics
enter as continuous variables.

```python
categorical_levels = {
    "employ": 3,
    "felony_prior_conviction": 3,
    "offense_type": 8,
    "sex": 2,
    "ethnicity": 2,
    "drug_abuse": 3,
    "race": 5,
    "supervision": 6,
    "age_dist": 6,
}
county_columns = ["house_female", "tax_property"]
county_statistic_columns = ["mean_re_time", "percent_re"]

cox_data = df.merge(
    county_statistics,
    left_on="code_county",
    right_index=True,
    how="left",
)

model_columns = (
    list(categorical_levels)
    + county_columns
    + county_statistic_columns
    + ["felony_arrest", "time", "observed"]
)
cox_data = cox_data[model_columns].dropna().copy()

# Cumulative felony arrests, added independently of the published score.
cox_data["offenses"] = cox_data["felony_arrest"].round()
cox_data["time"] = cox_data["time"].clip(lower=1.0)
cox_data["observed"] = (cox_data["observed"] > 0).astype(int)

for column, number_of_levels in categorical_levels.items():
    cox_data[column] = (
        cox_data[column]
        .round()
        .clip(1, number_of_levels)
        .astype(int)
        .astype("category")
    )

print("modeling frame:", cox_data.shape)
print("observed events:", int(cox_data["observed"].sum()))
display(cox_data.head())
```

Do not include `score_fixed`, `score_comm`, `score_age_dist`, `offset`, or
`score` in `model_columns`. Adding any of them would reintroduce Sirakaya's
coefficient values into the fit.

## 5. Cell 4 — fit the Cox model

```python
cox_formula = " + ".join(
    list(categorical_levels)
    + county_columns
    + county_statistic_columns
    + ["offenses"]
)

cox_model = CoxPHFitter(penalizer=0.0)
cox_model.fit(
    cox_data,
    duration_col="time",
    event_col="observed",
    formula=cox_formula,
    show_progress=True,
)

cox_model.print_summary(decimals=4)
cox_model.summary.to_csv(
    RESULTS_DIR / "cox_coefficient_summary.csv"
)
```

This is an unpenalized maximum-likelihood fit. If it reports convergence or
singular-matrix problems, inspect collinearity, sparse levels, and event counts
before changing the specification. Record any penalty or variable removal as
a model change.

## 6. Cell 5 — joint tests for categorical variables

The coefficient table tests individual dummy levels. The following cell adds
one joint Wald test for each complete categorical variable.

```python
coefficients = cox_model.params_
covariance = cox_model.variance_matrix_
wald_rows = []

for column in categorical_levels:
    coefficient_names = [
        name
        for name in coefficients.index
        if name == column
        or name.startswith(column + "[")
        or name.startswith(column + "_")
    ]

    beta = coefficients.loc[coefficient_names].to_numpy()
    covariance_block = covariance.loc[
        coefficient_names, coefficient_names
    ].to_numpy()
    statistic = float(
        beta @ np.linalg.solve(covariance_block, beta)
    )

    wald_rows.append(
        {
            "covariate": column,
            "df": len(coefficient_names),
            "wald_chi2": statistic,
            "p_value": chi2.sf(statistic, len(coefficient_names)),
        }
    )

joint_wald_tests = pd.DataFrame(wald_rows)
joint_wald_tests.to_csv(
    RESULTS_DIR / "cox_joint_wald_tests.csv",
    index=False,
)
display(joint_wald_tests)
```

## 7. Cell 6 — save scores and baseline survival

```python
cox_data["score_fitted"] = cox_model.predict_log_partial_hazard(
    cox_data
)
baseline_survival = cox_model.baseline_survival_

cox_data.to_csv(
    RESULTS_DIR / "cox_individual_scores.csv",
    index=False,
)
baseline_survival.to_csv(
    RESULTS_DIR / "cox_baseline_survival.csv"
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    baseline_survival.index,
    baseline_survival.iloc[:, 0],
    linewidth=2,
)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Baseline survival")
ax.set_title("Baseline survival without Sirakaya estimates")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "cox_baseline_survival.png",
    dpi=300,
)
plt.show()

cox_model.plot_partial_effects_on_outcome(
    covariates="offenses",
    values=[0, 2, 6, 10],
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "cox_offense_partial_effects.png",
    dpi=300,
)
plt.show()
```

Run the proportional-hazards diagnostics separately because they can produce
a long report:

```python
cox_model.check_assumptions(cox_data)
```

## 8. Outputs and handoff

The optional fit creates:

```text
results/
└── regression-without-sirakaya/
    ├── cox_coefficient_summary.csv
    ├── cox_joint_wald_tests.csv
    ├── cox_individual_scores.csv
    ├── cox_baseline_survival.csv
    ├── cox_baseline_survival.png
    └── cox_offense_partial_effects.png
```

These files are analysis outputs; `run_policy.py` does not load them. If this
fit supports changing the simulator's model specification or hard-coded
coefficients, make and record that code change before starting
[Step 2: Run the policy simulation](simulation_run.md). Otherwise, no manual
handoff is required.

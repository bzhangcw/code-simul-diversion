# Step 1: Fit and check the regression models

This page turns `./simulation/py-sirakaya-fitting.ipynb` into a linear,
copy-paste-ready workflow. Start with a fresh notebook and run the cells below
in order.

There are two fitting tracks:

| Track | Purpose | Required for simulation? |
| --- | --- | --- |
| Exponential AFT refit | Refit the baseline survival and the weights on the existing score and cumulative offenses. This matches the current `SimulationSetup`. | Yes; `run_policy.py` performs this refit automatically. |
| From-scratch Cox model | Re-estimate every individual and community coefficient and compare them with the values in `sirakaya.py`. | No; this is a model-validation exercise. |

The first track is the main workflow. The Cox section is optional.

## 1. Start Jupyter in the correct directory

The notebook imports modules directly from `./simulation/`, so start Jupyter
there:

```bash
cd ./simulation
python -m jupyter lab py-sirakaya-fitting.ipynb
```

Complete the [environment setup](index.md#environment-setup) first, and launch
Jupyter while that virtual environment is active. The notebook will then use
the repository's pinned dependencies instead of a personally configured
kernel. Restart the kernel before a formal run if it was already open.

## 2. Cell 1 — imports and output directory

This cell imports only the modules used below and puts persistent regression
artifacts under `results/regression/`.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import chi2, gamma, kstest, probplot

import input
import simulation
import sirakaya
from simulation_regression import refit_baseline

RESULTS_DIR = Path("results/regression")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR
```

Expected output:

```text
PosixPath('results/regression')
```

## 3. Cell 2 — load and validate the data

`input.load_data()` returns the original data, imputed data, individual
survival data, and county-level summaries. It reads the cached pickles in
`bin/` when available and otherwise rebuilds them from
`felony-1989-final.xlsx`.

```python
dfr, df, df_individual, df_community = input.load_data()

required_individual = {
    "time",
    "observed",
    "offset",
    "score",
    "score_fixed",
    "score_comm",
    "offenses",
    "code_county",
}
required_raw = {
    "time",
    "observed",
    "felony_arrest",
    "code_county",
    "house_female",
    "tax_property",
}

missing_individual = required_individual - set(df_individual.columns)
missing_raw = required_raw - set(df.columns)
assert not missing_individual, f"df_individual is missing: {missing_individual}"
assert not missing_raw, f"df is missing: {missing_raw}"

print("raw rows:", len(dfr))
print("imputed rows:", len(df))
print("individual rows:", len(df_individual))
print("counties:", df_individual["code_county"].nunique())
print("observed events:", int(df_individual["observed"].sum()))
display(df_individual[["time", "observed", "offset", "offenses", "score"]].head())
```

Do not continue if either assertion fails. The current individual-level column
name is `code_county`, not `code-county`.

## 4. Cell 3 — fit the exponential AFT model

The current simulator adds cumulative offenses to the existing Sirakaya score.
Use `offenses` here so that the notebook and `SimulationSetup.fit_kwargs`
estimate the same specification.

```python
simulator = simulation.Simulator(
    eval_score_fixed=sirakaya.eval_score_fixed,
    eval_score_comm=sirakaya.eval_score_comm,
    func_arrival=None,
    func_leaving=None,
)

fit_kwargs = {
    "new_col": "offenses",
    "bool_use_cph": False,
    "baseline": "exponential",
}

refit_baseline(simulator, df_individual, **fit_kwargs)
simulator.cph.print_summary(decimals=4)
```

The fitted exponential hazard has the form

\[
H(t \mid x) = \lambda_0 t
\exp\!\left(\theta_1\,\text{offset} +
            \theta_2\,\text{offenses}\right).
\]

Here `offset` is the existing individual, community, and age score. The refit
updates the simulator's score weights and baseline survival function.

## 5. Cell 4 — extract and save the fitted parameters

```python
aft_params = simulator.cph.params_["lambda_"]

aft_summary = pd.Series(
    {
        "theta_offset": aft_params["offset"],
        "theta_offenses": aft_params["offenses"],
        "lambda_0": np.exp(aft_params["Intercept"]),
        "baseline_survival_day_1095": float(
            simulator.s0_with_interpolation(1095)
        ),
    },
    name="estimate",
)

display(aft_summary)
aft_summary.to_csv(RESULTS_DIR / "aft_parameters.csv")
simulator.cph.summary.to_csv(RESULTS_DIR / "aft_coefficient_summary.csv")
```

Inspect the coefficient signs before continuing:

- a positive `theta_offenses` means more prior offenses increase the hazard;
- `lambda_0` must be positive;
- baseline survival must lie between zero and one.

## 6. Cell 5 — compare fitted survival with Kaplan–Meier

This cell compares the fitted survival curves at the minimum, median, and
maximum fitted scores with the nonparametric Kaplan–Meier estimate.

```python
times = np.linspace(
    df_individual["time"].min(),
    df_individual["time"].max(),
    200,
)

scores = {
    "minimum": df_individual["score"].min(),
    "median": df_individual["score"].median(),
    "maximum": df_individual["score"].max(),
}

kmf = KaplanMeierFitter()
kmf.fit(
    df_individual["time"],
    event_observed=df_individual["observed"],
)

fig, ax = plt.subplots(figsize=(8, 6))
kmf.plot_survival_function(ax=ax, label="Kaplan–Meier")

linestyles = {"minimum": "--", "median": "-.", "maximum": ":"}
for label, score in scores.items():
    fitted = [simulator.survival_function(t, score) for t in times]
    ax.plot(
        times,
        fitted,
        label=f"{label.title()} score ({score:.2f})",
        linestyle=linestyles[label],
        linewidth=2.5,
    )

ax.set_title("Survival function comparison")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival probability")
ax.set_ylim(0, 1.02)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "survival_function_comparison.png", dpi=300)
fig.savefig(RESULTS_DIR / "survival_function_comparison.pdf")
plt.show()

median_event_probability = 1 - simulator.survival_function(
    1095,
    scores["median"],
)
print(f"Event probability by day 1095 at median score: {median_event_probability:.4f}")
```

Check that every curve stays in `[0, 1]`, decreases with time, and has the
expected ordering by risk score.

## 7. Cell 6 — inspect score variation by county

```python
fig, ax = plt.subplots(figsize=(16, 6))
df_individual.boxplot(
    column="score",
    by="code_county",
    rot=90,
    ax=ax,
)
ax.set_title("Fitted score by county")
ax.set_xlabel("County code")
ax.set_ylabel("Score")
fig.suptitle("")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "score_by_county.png", dpi=300)
plt.show()
```

Large location or spread differences identify counties that deserve additional
data and model checks.

## 8. Cell 7 — fit the Gamma frailty distribution

The notebook models the exponentiated fixed individual score as a Gamma random
variable within each county. This cell fits the distribution, performs a
Kolmogorov–Smirnov test, and saves density and Q-Q plots.

```python
theta_offset = aft_params["offset"]
df_individual["exp_score_fixed"] = np.exp(
    theta_offset * df_individual["score_fixed"]
)

counties = sorted(df_individual["code_county"].unique())
batches = [counties[i : i + 16] for i in range(0, len(counties), 16)]
fit_results = {}

for batch_id, batch in enumerate(batches):
    fig_density, density_axes = plt.subplots(
        4, 4, figsize=(12, 10), sharex=True, sharey=True
    )
    fig_qq, qq_axes = plt.subplots(
        4, 4, figsize=(12, 10), sharex=True, sharey=True
    )

    for ax in density_axes.ravel()[len(batch) :]:
        ax.set_visible(False)
    for ax in qq_axes.ravel()[len(batch) :]:
        ax.set_visible(False)

    for density_ax, qq_ax, county in zip(
        density_axes.ravel(), qq_axes.ravel(), batch
    ):
        values = df_individual.loc[
            df_individual["code_county"] == county,
            "exp_score_fixed",
        ].to_numpy()

        shape, location, scale = gamma.fit(values, floc=0.0)
        ks_result = kstest(values, "gamma", args=(shape, location, scale))

        fit_results[county] = {
            "shape": shape,
            "location": location,
            "scale": scale,
            "rate": 1 / scale,
            "ks_statistic": ks_result.statistic,
            "ks_p_value": ks_result.pvalue,
            "n": len(values),
        }

        grid = np.linspace(0, values.max(), 200)
        density_ax.hist(values, bins=30, density=True, alpha=0.5)
        density_ax.plot(grid, gamma.pdf(grid, shape, location, scale))
        density_ax.set_title(f"County {county}")
        density_ax.text(
            0.98,
            0.98,
            f"shape={shape:.2f}\nKS p={ks_result.pvalue:.3f}",
            transform=density_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

        probplot(
            values,
            dist=gamma,
            sparams=(shape, location, scale),
            plot=qq_ax,
        )
        qq_ax.set_title(f"County {county}")

    fig_density.tight_layout()
    fig_qq.tight_layout()
    fig_density.savefig(
        RESULTS_DIR / f"gamma_density_batch_{batch_id}.pdf"
    )
    fig_qq.savefig(RESULTS_DIR / f"gamma_qq_batch_{batch_id}.pdf")
    plt.close(fig_density)
    plt.close(fig_qq)

gamma_results = (
    pd.DataFrame.from_dict(fit_results, orient="index")
    .rename_axis("code_county")
    .sort_index()
)
gamma_results.to_csv(RESULTS_DIR / "gamma_fits_by_county.csv")
display(gamma_results.head())
```

A small KS p-value is evidence against the fitted Gamma distribution. Interpret
it together with sample size and the Q-Q plot rather than as a pass/fail rule
by itself.

## 9. Optional Cell 8 — assemble the from-scratch Cox data

Stop here if the goal is only to validate the model used by the simulator. Run
the remaining cells only to re-estimate all score coefficients.

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
community_columns = ["house_female", "tax_property"]
time_varying_columns = ["mean_re_time", "percent_re"]

cox_data = df.merge(
    df_community[time_varying_columns],
    left_on="code_county",
    right_index=True,
    how="left",
)

model_columns = (
    list(categorical_levels)
    + community_columns
    + time_varying_columns
    + ["felony_arrest", "time", "observed"]
)
cox_data = cox_data[model_columns].dropna().copy()
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

print("modeling rows:", len(cox_data))
print("observed events:", int(cox_data["observed"].sum()))
display(cox_data.head())
```

## 10. Optional Cell 9 — fit the Cox proportional-hazards model

```python
cox_formula = " + ".join(
    list(categorical_levels)
    + community_columns
    + time_varying_columns
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
cox_model.summary.to_csv(RESULTS_DIR / "cox_coefficient_summary.csv")
```

If the optimizer reports convergence or singular-matrix problems, do not
silently add a penalty. First inspect collinearity, rare categorical levels,
and the event count; any specification change belongs in the regression
record.

## 11. Optional Cell 10 — joint tests for categorical variables

The standard coefficient table reports one test per dummy level. This cell
computes one joint Wald test for each categorical variable.

```python
beta = cox_model.params_
covariance = cox_model.variance_matrix_
wald_rows = []

for column in categorical_levels:
    coefficient_names = [
        name
        for name in beta.index
        if name == column
        or name.startswith(column + "[")
        or name.startswith(column + "_")
    ]

    coefficients = beta.loc[coefficient_names].to_numpy()
    covariance_block = covariance.loc[
        coefficient_names, coefficient_names
    ].to_numpy()
    statistic = float(
        coefficients
        @ np.linalg.solve(covariance_block, coefficients)
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

## 12. Optional Cell 11 — save scores and baseline survival

```python
cox_data["score_scratch"] = cox_model.predict_log_partial_hazard(cox_data)
baseline_survival = cox_model.baseline_survival_

cox_data.to_csv(RESULTS_DIR / "cox_individual_scores.csv", index=False)
baseline_survival.to_csv(RESULTS_DIR / "cox_baseline_survival.csv")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    baseline_survival.index,
    baseline_survival.iloc[:, 0],
    linewidth=2,
)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Baseline survival")
ax.set_title("Cox baseline survival")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cox_baseline_survival.png", dpi=300)
plt.show()

cox_model.plot_partial_effects_on_outcome(
    covariates="offenses",
    values=[0, 2, 6, 10],
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cox_offense_partial_effects.png", dpi=300)
plt.show()
```

Run `cox_model.check_assumptions(cox_data)` separately because it can produce a
large diagnostic report.

## 13. Optional Cell 12 — compare with the Sirakaya coefficients

This final cell puts the from-scratch estimates beside the coefficient values
currently used by `sirakaya.py`.

```python
sirakaya_categorical = {
    "employ": sirakaya._SCORE_EMPLOY_VALS,
    "felony_prior_conviction": sirakaya._SCORE_FELONY_PRIOR_CONV_VALS,
    "offense_type": sirakaya._SCORE_OFFENSE_TYPE_VALS,
    "sex": sirakaya._SCORE_SEX_VALS,
    "ethnicity": sirakaya._SCORE_ETHNICITY_VALS,
    "drug_abuse": sirakaya._SCORE_DRUG_VALS,
    "race": sirakaya._SCORE_RACE_VALS,
    "supervision": sirakaya._SCORE_SUPERVISION_VALS,
    "age_dist": sirakaya._SCORE_AGE_DIST_VALS,
}
sirakaya_continuous = {
    "house_female": sirakaya._SCORE_HOUSE_FEMALE_COEF,
    "tax_property": sirakaya._SCORE_PROPERTY_TAX_COEF,
    "mean_re_time": sirakaya.SIRAKAYA_COEFFS_COMM[0],
    "percent_re": sirakaya.SIRAKAYA_COEFFS_COMM[1],
}

estimated = cox_model.params_


def estimated_level_coefficient(column, level):
    """Return a fitted dummy coefficient despite patsy naming differences."""
    for name in estimated.index:
        if name.startswith(column) and (
            f"[T.{level}]" in name
            or f"[T.{level}.0]" in name
            or name == f"{column}_{level}"
        ):
            return estimated[name]
    return np.nan


comparison_rows = []
for column, published_values in sirakaya_categorical.items():
    for level, published in enumerate(published_values, start=1):
        fitted = (
            0.0
            if level == 1
            else estimated_level_coefficient(column, level)
        )
        comparison_rows.append(
            {
                "covariate": column,
                "level": str(level),
                "sirakaya": published,
                "fitted": fitted,
            }
        )

for column, published in sirakaya_continuous.items():
    comparison_rows.append(
        {
            "covariate": column,
            "level": "continuous",
            "sirakaya": published,
            "fitted": estimated.get(column, np.nan),
        }
    )

comparison_rows.append(
    {
        "covariate": "offenses",
        "level": "continuous",
        "sirakaya": np.nan,
        "fitted": estimated.get("offenses", np.nan),
    }
)

coefficient_comparison = pd.DataFrame(comparison_rows)
coefficient_comparison["difference"] = (
    coefficient_comparison["fitted"]
    - coefficient_comparison["sirakaya"]
)
coefficient_comparison.to_csv(
    RESULTS_DIR / "cox_vs_sirakaya.csv",
    index=False,
)
display(coefficient_comparison)
```

The comparison is diagnostic. Do not replace the values in `sirakaya.py`
without also checking uncertainty, coding of reference levels, proportional-
hazards assumptions, and out-of-sample behavior.

## 14. What the regression step produces

After all cells, the persistent outputs are:

```text
results/
└── regression/
    ├── aft_parameters.csv
    ├── aft_coefficient_summary.csv
    ├── survival_function_comparison.png
    ├── survival_function_comparison.pdf
    ├── score_by_county.png
    ├── gamma_fits_by_county.csv
    ├── gamma_density_batch_<n>.pdf
    ├── gamma_qq_batch_<n>.pdf
    ├── cox_coefficient_summary.csv
    ├── cox_joint_wald_tests.csv
    ├── cox_individual_scores.csv
    ├── cox_baseline_survival.csv
    ├── cox_baseline_survival.png
    ├── cox_offense_partial_effects.png
    └── cox_vs_sirakaya.csv
```

The Cox files are present only if the optional track is run.

## 15. Handoff to the simulation

The simulation does not load these CSV files. `run_policy.py` constructs a new
simulator and refits the exponential AFT model at run time using
`SimulationSetup.fit_kwargs`.

If the notebook supports a change to the model specification or hard-coded
Sirakaya coefficients, make that code change explicitly and record it before
starting [Step 2: Run the policy simulation](simulation_run.md). Otherwise,
the notebook is a validation record and no manual handoff is required.

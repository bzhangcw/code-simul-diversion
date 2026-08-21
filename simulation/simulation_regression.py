# survival function fitters
import json
import pandas as pd
from autograd import numpy as np
from lifelines import CoxPHFitter
from lifelines.fitters import ParametricRegressionFitter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# this is the default scoring weights
# @note: this is the default scoring weights for the original model
# these weights are obtained from the following AFTFitter.
DEFAULT_SCORING_PARAMS = {
    "score_fixed": 0.7904,
    "score_comm": 0.7904,
    "score_treatment": 1.0,
    "score_state": 1.0,
}


def scoring_all(idx, dfi, scoring_weights=DEFAULT_SCORING_PARAMS, **kwargs):

    # this part is already computed in the `input.py`
    return (
        dfi.loc[idx, "score_fixed"] * scoring_weights["score_fixed"]
        + dfi.loc[idx, "score_comm"] * scoring_weights["score_comm"]
        # note the individual state is weighted already.
        + dfi.loc[idx, "score_state"] * scoring_weights["score_state"]
        # the treatment effect is applied to the score.
        + dfi.loc[idx, "score_treatment"] * scoring_weights["score_treatment"]
    )


class ExponentialAFTFitter(ParametricRegressionFitter):

    # this class property is necessary, and should always be a non-empty list of strings.
    _fitted_parameter_names = ["lambda_"]

    # the below variables maps {dataframe columns, formulas} to parameters
    def _cumulative_hazard(self, params, t, Xs):
        # params is a dictionary that maps unknown parameters to a numpy vector.
        # Xs is a dictionary that maps unknown parameters to a numpy 2d array
        beta = params["lambda_"]
        X = Xs["lambda_"]
        lambda_ = np.exp(np.dot(X, beta))
        return lambda_ * t


"""
Refit the baseline hazard after adding a new covariate `new_col`.
    if refit, then the `produce_score` function will be updated.
"""


def refit_baseline(
    self,
    df_individual,
    new_col=None,
    bool_use_cph=False,
    baseline="breslow",
    verbosity=0,
):
    if new_col is None:
        refit_baseline_no_extra(self, df_individual, baseline=baseline)
    else:
        refit_baseline_extra(
            self,
            df_individual,
            new_col,
            bool_use_cph,
            baseline=baseline,
            verbosity=verbosity,
        )


def refit_baseline_no_extra(self, df_individual):
    raise ValueError("refit_baseline_no_extra is not implemented")


def refit_baseline_extra(
    self, df_individual, new_col, bool_use_cph=False, baseline="breslow", verbosity=0
):
    """
    Refit baseline hazard after adding one extra covariate `new_col`.
    Keeps `survival_function` unchanged by integrating the extra effect
    into each subject's 'score' before recomputing the baseline.
    """
    if verbosity > 0:
        print(df_individual.columns)

    if baseline in ["gompertz", "breslow"]:
        __refit_baseline_extra_breslow(
            self, df_individual, new_col, bool_use_cph, baseline, verbosity
        )
    else:
        __refit_baseline_extra_aft_exponential(
            self, df_individual, new_col, verbosity=verbosity
        )

    if verbosity > 0:
        print("The fitted state weights are:")
        print(json.dumps(self._scoring_state_weights, indent=4))
        print("The fitted weights are:")
        print(json.dumps(self._scoring_weights, indent=4))
    self.state_defs.scoring_weights = self._scoring_state_weights
    self.produce_score = lambda idx, dfi: scoring_all(idx, dfi, self._scoring_weights)


def __refit_baseline_extra_breslow(
    self, df_individual, new_col, bool_use_cph=False, baseline="breslow", verbosity=0
):
    """
    Refit baseline hazard using Breslow estimator with Cox PH model.

    Args:
        self: Simulator object to update
        df_individual: DataFrame with individual data
        new_col: Name of the new covariate column
        bool_use_cph: If True, use CoxPHFitter's baseline survival; otherwise compute manually
        baseline: "breslow" or "gompertz" (gompertz fits a parametric curve to Breslow estimate)
        verbosity: Verbosity level for logging
    """
    df = df_individual

    # Fit a Cox model with new_col as a covariate, old_score as offset
    self.cph = cph = CoxPHFitter(
        penalizer=0.15,
    )
    cph.fit(
        df,
        duration_col="time",
        event_col="observed",
        formula=f"offset + {new_col}",
        show_progress=(verbosity > 0),
    )

    # Compute a new linear predictor: score = α * offset + β * new_col
    # @note: the original score is kept in the `offset`
    df_individual["score"] = (
        cph.params_["offset"] * df_individual["offset"]
        + cph.params_[new_col] * df_individual[new_col]
    )

    # Recompute baseline cumulative hazard (Breslow) using combined_score
    df_sorted = df_individual.sort_values("time")
    times = df_sorted["time"].unique()
    baseline_cumhaz = []

    for t in times:
        n_deaths = (
            (df_individual["time"] == t) & (df_individual["observed"] == 1)
        ).sum()
        risk_set = df_individual["time"] >= t
        sum_risk = np.sum(np.exp(df_individual.loc[risk_set, "score"]))
        dLambda = n_deaths / sum_risk if sum_risk > 0 else 0
        baseline_cumhaz.append((t, dLambda))

    cumhaz_df = pd.DataFrame(baseline_cumhaz, columns=["time", "dLambda"])
    cumhaz_df["Lambda"] = cumhaz_df["dLambda"].cumsum()
    cumhaz_df["S0"] = np.exp(-cumhaz_df["Lambda"])

    self.cumhaz_df = cumhaz_df
    self.cumhaz_df_by_cph = cph.baseline_survival_.copy()

    if bool_use_cph:
        self.s0_with_interpolation = interp1d(
            self.cumhaz_df_by_cph.index.values,
            self.cumhaz_df_by_cph["baseline survival"].values,
            kind="previous",
            fill_value="extrapolate",
        )
    else:
        self.s0_with_interpolation = interp1d(
            cumhaz_df["time"].values,
            cumhaz_df["S0"].values,
            kind="previous",
            fill_value="extrapolate",
        )

    self.s0 = self.s0_with_interpolation
    if baseline == "gompertz":
        __override_Gompertz(self, verbosity=verbosity)

    print(cph.params_)
    self._scoring_state_weights = {
        "score_age_dist": cph.params_["offset"],
        f"score_{new_col}": cph.params_[new_col],
    }
    self._scoring_weights = {
        "score_fixed": cph.params_["offset"],
        "score_comm": cph.params_["offset"],
        "score_state": 1.0,
        "score_treatment": 1.0,
    }


def __override_Gompertz(self, verbosity=0):
    # 1) sample times
    t_min, t_max = self.cumhaz_df["time"].min(), self.cumhaz_df["time"].max()
    t_obs = np.linspace(t_min, t_max, 200)
    S_obs = self.s0_with_interpolation(t_obs)

    def S_gompertz(t, a, b):
        return np.exp(-(a / b) * (np.exp(b * t) - 1))

    # 3) fit params
    (a_hat, b_hat), _ = curve_fit(S_gompertz, t_obs, S_obs, p0=[0.1, 0.01])
    if verbosity > 0:
        print(a_hat, b_hat)

    # 4) override the survival function
    self.s0 = lambda t: S_gompertz(t, a_hat, b_hat)


def __refit_baseline_extra_aft_exponential(self, df_individual, new_col, verbosity=0):
    df = df_individual.copy()
    df["time"] += 1.0

    # 2) Fit a Cox model with only new_col as a covariate, old_score as offset
    self.cph = cph = ExponentialAFTFitter()
    cph.fit(
        df,
        duration_col="time",
        event_col="observed",
        regressors={"lambda_": f"offset + {new_col}"},
        show_progress=(verbosity > 0),
    )

    # 4) Compute a new linear predictor: score = α * offset + β * new_col
    _params = cph.params_["lambda_"]
    df_individual["score"] = (
        _params["offset"] * df_individual["offset"]
        + _params[new_col] * df_individual[new_col]
    )

    λ0 = np.exp(_params["Intercept"])

    # 5) Recompute baseline cumulative hazard (Breslow) using combined_score
    # But this is only for comparison use only.
    df_sorted = df_individual.sort_values("time")
    times = df_sorted["time"].unique()
    baseline_cumhaz = []

    for t in times:
        n_deaths = (
            (df_individual["time"] == t) & (df_individual["observed"] == 1)
        ).sum()
        risk_set = df_individual["time"] >= t
        sum_risk = np.sum(np.exp(df_individual.loc[risk_set, "score"]))
        dLambda = n_deaths / sum_risk if sum_risk > 0 else 0
        baseline_cumhaz.append((t, dLambda))

    cumhaz_df = pd.DataFrame(baseline_cumhaz, columns=["time", "dLambda"])
    cumhaz_df["Lambda"] = cumhaz_df["dLambda"].cumsum()
    cumhaz_df["S0"] = np.exp(-cumhaz_df["Lambda"])

    self.cumhaz_df = cumhaz_df

    self.s0_with_interpolation = lambda t: np.exp(-λ0 * t)
    self.s0 = self.s0_with_interpolation
    self._scoring_state_weights = {
        "score_age_dist": cph.params_["lambda_"]["offset"],
        f"score_{new_col}": cph.params_["lambda_"][new_col],
    }
    self._scoring_weights = {
        "score_fixed": cph.params_["lambda_"]["offset"],
        "score_comm": cph.params_["lambda_"]["offset"],
        "score_state": 1.0,
        "score_treatment": 1.0,
    }


def plot_score_survival(self, scores, output_path=None, figsize=(8, 4), ax=None):
    """
    Plot the survival function S(t | score) for one or more scores.

    The survival function is computed as:
        S(t | score) = S0(t) ^ exp(score)

    This is consistent with the sampling method in sample_survival_time().

    Args:
        self: Simulator object with cumhaz_df containing baseline survival
        scores: A single score value or a list of score values
        output_path: Optional path to save the figure (without extension, saves .png and .pgf)
        figsize: Figure size tuple (default: (8, 4))
        ax: Optional matplotlib axes to plot on (for combining multiple plots)

    Returns:
        matplotlib axes object
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Handle single score or list of scores
    if not hasattr(scores, "__iter__"):
        scores = [scores]

    # Get time points and baseline survival from cumhaz_df
    times = self.cumhaz_df["time"].values
    S0 = self.cumhaz_df["S0"].values

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot survival for each score
    for score in scores:
        S_score = S0 ** np.exp(score)
        ax.plot(times, S_score, label=f"score={score:.2f}", linewidth=2)

    # Formatting
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save if output_path provided
    if output_path is not None:
        plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path + ".pgf", bbox_inches="tight")
        print(f"Saved plot to {output_path}.png and {output_path}.pgf")

    return ax


def get_survival_probability(self, score, t):
    """
    Get the survival probability S(t | score) at a specific time t.

    The survival function is computed as:
        S(t | score) = S0(t) ^ exp(score)

    Args:
        self: Simulator object with s0 interpolation function
        score: The risk score value (single value or array-like)
        t: Time point to evaluate (single value or array-like)

    Returns:
        Survival probability (float or array depending on inputs)
    """
    S0_t = self.s0(t)
    return S0_t ** np.exp(score)

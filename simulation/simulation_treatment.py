# survival function fitters
import pandas as pd
from autograd import numpy as np
from functools import wraps
from lifelines import CoxPHFitter
from lifelines.fitters import ParametricRegressionFitter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os

TR_LOW = float(os.getenv("TR_LOW", "15.0"))
TR_HIGH = float(os.getenv("TR_HIGH", "0.0"))


# ------------------------------------------------------------------------------
# treatment rules
# ------------------------------------------------------------------------------
def treatment_null(dfi, **kwargs):
    dfi["bool_treat"] = 0.0
    return []


def batch_treatment_rule(func):
    """
    Decorator that handles common batch treatment rule logic.
    Applied periodically to select and enroll individuals into treatment.

    - Get qualifying and in-treatment populations
    - Calculate remaining capacity = C - sum(dosage)
    - Apply treatment to selected indices
    - Return selected indices

    The wrapped function should return the selected indices from candidates.
    """

    @wraps(func)
    def wrapper(
        dfi,
        capacity=100,
        str_qualifying="",
        str_current_enroll="",
        t=0.0,
        **kwargs,
    ):
        people_qualifying = dfi.query(str_qualifying)
        people_intreatment = dfi.query(str_current_enroll)
        remaining_capacity = capacity - people_intreatment["dosage"].sum()
        print(
            f"""
    capacity: {capacity},
    people_qualifying: {people_qualifying.shape[0]},
    people_intreatment: {people_intreatment.shape[0]},
    remaining_capacity: {remaining_capacity},
    score_range: {people_qualifying["score"].min()} - {people_qualifying["score"].max()},
"""
        )
        if remaining_capacity <= 0:
            return []

        # Get candidates (qualifying people not yet in treatment)
        candidates = people_qualifying.query("bool_treat == 0")

        if candidates.empty:
            return []

        # Let the wrapped function select from candidates
        idx_selected = func(
            candidates=candidates,
            remaining_capacity=remaining_capacity,
            **kwargs,
        )

        if len(idx_selected) == 0:
            return []

        return idx_selected

    return wrapper


@batch_treatment_rule
def treatment_rule_priority(
    candidates, remaining_capacity, key=None, ascending=False, exclude=None, **kwargs
):
    """
    Select individuals by priority (sorted by key).

    @note:
        - if `key` is None, select by highest `score`
        - if `ascending`, choose the top-C-smallest individuals
            else choose the top-C-largest individuals
        - if `exclude` is provided, filter out individuals where exclude(row) is True
    """
    cc = candidates.copy()

    # Apply exclusion filter if provided
    if exclude is not None:
        mask = cc.apply(lambda row: not exclude(row), axis=1)
        cc = cc[mask]

    key = "score" if key is None else key
    to_compute = kwargs.get("to_compute", None)
    if to_compute is not None:
        cc[key] = to_compute(cc)

    # Sort candidates by priority
    sorted_candidates = cc.sort_values(key, ascending=ascending)

    # Read dosages from dataframe and compute cumsum
    dosages = sorted_candidates["dosage"].values
    cumsum_dosages = np.cumsum(dosages)

    # Select where cumsum <= remaining_capacity
    mask = cumsum_dosages <= remaining_capacity

    # Handle ties at the boundary: if there are tied scores at the cutoff,
    # we must exclude all of them to avoid exceeding capacity
    if mask.sum() > 0 and mask.sum() < len(mask):
        # Get the score of the last included candidate
        last_included_idx = mask.sum() - 1
        last_included_score = sorted_candidates[key].iloc[last_included_idx]

        # Check if there are ties with the first excluded candidate
        first_excluded_score = sorted_candidates[key].iloc[last_included_idx + 1]

        if last_included_score == first_excluded_score:
            # Exclude all candidates with the tied score at the boundary
            # Use element-wise comparison that handles tuple values correctly
            tied_mask = sorted_candidates[key] == last_included_score
            mask = mask & ~tied_mask.values
    print(
        f"candidates:\n{sorted_candidates.loc[mask, ['state', 'score', 'score_state', 'score_treatment']]}"
    )
    return sorted_candidates.index[mask]


@batch_treatment_rule
def treatment_rule_random(candidates, remaining_capacity, **kwargs):
    """
    Select individuals randomly.
    """
    # Shuffle candidates randomly
    shuffled = candidates.sample(frac=1)

    # Compute cumsum of dosages
    cumsum_dosages = np.cumsum(shuffled["dosage"].values)

    # Select where cumsum <= remaining_capacity
    mask = cumsum_dosages <= remaining_capacity
    return shuffled.index[mask]


# @batch_treatment_rule
# def treatment_rule_priority_fluid(
#     candidates, remaining_capacity, prob_vector, state_column="state", **kwargs
# ):
#     """
#     Priority policy, but may not exhaust the capacity.
#         The policy is motivated by the fluid model, where we preset a probability vector for
#         prob_vector = {state s: probability p_s}, s \in S
#     and each person (in state s) is assigned according to a coin flip with the probability vector.

#     Args:
#         candidates: DataFrame of qualifying individuals not yet in treatment
#         remaining_capacity: maximum number of individuals to select
#         prob_vector: dict mapping state tuple -> probability (0 to 1)
#         state_column: column name containing the state (default "state")
#     Returns:
#         indices of selected individuals (may be fewer than remaining_capacity)
#     """
#     selected = []
#     for idx, row in candidates.iterrows():
#         if len(selected) >= remaining_capacity:
#             break
#         # Get state as tuple
#         state = row[state_column]
#         if hasattr(state, "value"):
#             state_key = tuple(state.value)
#         else:
#             state_key = tuple(state) if hasattr(state, "__iter__") else (state,)
#         # Look up probability for this state (default 0 if not in prob_vector)
#         prob = prob_vector.get(state_key, 0.0)
#         # Coin flip with probability
#         if np.random.random() < prob:
#             selected.append(idx)
#     return selected


# ------------------------------------------------------------------------------
# heterogeneous treatment effect functions
# ------------------------------------------------------------------------------
def treatment_effect_type_1(row, med=-0.3604, mt_high=TR_HIGH, mt_low=TR_LOW):
    """
    Heterogeneous treatment effect based on individual characteristics.
        "higher score than current risk score, less treatment effect."
        "H vs. L., baseline = -0.3425"
    Args:
        row: pandas Series representing an individual's data

    Returns:
        float: treatment effect value (0-1)
    """
    delta = row["score"] - med
    highr = delta >= 0
    mt = mt_high if highr else mt_low
    return -0.3425 * mt


def treatment_effect_type_2(row, med=-0.3604, mt_high=0.1, mt_low=6.0):
    """
    Heterogeneous treatment effect based on individual characteristics.
      "higher state score than the initial mean, less treatment effect."
    Args:
        row: pandas Series representing an individual's data

    Returns:
        float: treatment effect value (0-1)
    """
    # TODO: implement type-1 heterogeneous treatment effect logic
    delta = row["score_state"] + 0.3604
    highr = delta >= 0
    mt = mt_high if highr else mt_low
    return -0.3425 * mt


def treatment_effect_type_3(row, med=-0.3604, mt_high=0.1, mt_low=6.0):
    """
    Heterogeneous treatment effect based on individual characteristics.
     "more offenses, less treatment effect."

    Args:
        row: pandas Series representing an individual's data

    Returns:
        float: treatment effect value (0-1)
    """
    # TODO: implement type-1 heterogeneous treatment effect logic
    delta = row["offenses"] - 1
    highr = delta >= 0
    mt = mt_high if highr else mt_low
    return -0.3425 * mt


# ------------------------------------------------------------------------------
# heterogeneous treatment dosage functions
# ------------------------------------------------------------------------------
def treatment_dosage_default(idx, dfi):
    """
    Homogeneous treatment dosage.
    """
    return 1.0


def treatment_dosage_type_1(idx, dfi):
    """
    Heterogeneous treatment dosage based on individual characteristics.
        more offenses, more dosage.

    Args:
        row: pandas Series representing an individual's data

    Returns:
        float: treatment dosage value :>=1
    """

    # return np.sqrt(dfi.at[idx, "offenses"] + 1)
    return (dfi.at[idx, "offenses"] ** 2) + 1

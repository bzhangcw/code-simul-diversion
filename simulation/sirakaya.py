import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif font family
        "font.serif": [
            "Computer Modern Roman"
        ],  # Specify specific serif font (default LaTeX font)
        # 'axes.labelsize': 12,  # Font size for labels
        "font.size": 24,  # General font size
        "legend.fontsize": 22,  # Font size for legend
        "xtick.labelsize": 22,  # Font size for x-axis ticks
        "ytick.labelsize": 22,  # Font size for y-axis ticks
    }
)

ucr_to_category = {
    100: 1,
    110: 1,
    120: 1,
    130: 1,
    140: 1,
    200: 2,
    210: 2,
    220: 2,
    300: 3,
    310: 3,
    320: 3,
    400: 4,
    430: 4,
    500: 5,
    510: 5,
    520: 5,
    530: 5,
    600: 6,
    700: 7,
    710: 7,
    720: 7,
    730: 7,
    800: 8,
    810: 8,
    820: 8,
    830: 8,
    840: 8,
    850: 8,
    860: 8,
    870: 8,
    880: 8,
    998: 8,
}

# precompute constants Sirakaya's scores
_SCORE_AGE_DIST_VALS = [0.0, -0.270, -0.456, -0.709, -1.282, -1.808]
_SCORE_SEX_VALS = [0.0, -0.534]
_SCORE_EMPLOY_VALS = [0.0, -0.206, -0.488]
_SCORE_DRUG_VALS = [0.0, 0.039, 0.403]
_SCORE_FELONY_PRIOR_CONV_VALS = [0.0, 0.369, 0.463]
_SCORE_RACE_VALS = [0.0, 0.593, 0.245, -0.537, -0.534]
_SCORE_ETHNICITY_VALS = [0.0, -0.336]
_SCORE_OFFENSE_TYPE_VALS = [0.0, -0.128, 0.739, 0.386, 0.868, 0.936, 0.606, 0.776]
_SCORE_SUPERVISION_VALS = [0.0, 0.220, -0.172, 0.238, 0.586, 0.697]

# Group (II) community-level coefficients (continuous percentages).
# Source: tab.sirakaya.coeffs in the journal manuscript (Sirakaya 2003 estimates).
_SCORE_HOUSE_FEMALE_COEF = -0.034  # Female-head household (% of households)
_SCORE_PROPERTY_TAX_COEF = 0.012  # Property taxes (% of income)


# ------------------------------------------------------------------------------
# age updates
# ------------------------------------------------------------------------------
# age_dist to age map (I take the lower bound of the age_dist)
# see the codebook for more details
age_dist_map = {
    0: 18,
    1: 20,
    2: 25,
    3: 30,
    4: 40,
    5: 50,
    6: 55,
}


def age_dist_to_age(age_dist):
    # support float, use interpolation
    # random between the lower and upper bound
    # if already 6 then set 55
    if int(age_dist) == 6:
        return 55.0
    return float(
        np.random.randint(age_dist_map[int(age_dist) - 1], age_dist_map[int(age_dist)])
    )


def age_to_age_dist(age):
    # find the age_dist of the age
    # if it is between two dist, use the lower one
    for k, v in age_dist_map.items():
        if age < v:
            return k
    return 6


# ------------------------------------------------------------------------------
# scoring rules
# ------------------------------------------------------------------------------
def score_levels(v, vals):
    n = len(vals)
    for i in range(n):
        if v <= i + 1.0:
            return vals[i]
    return vals[n - 1]


def score_age_dist(v):
    return score_levels(v, _SCORE_AGE_DIST_VALS)


def score_age(age):
    """
    Compute age score using linear interpolation based on continuous age value.

    Uses age_dist_map boundaries and _SCORE_AGE_DIST_VALS to interpolate.
    The mapping is:
    - age in [18, 20) -> interpolate between 0.0 and -0.270
    - age in [20, 25) -> interpolate between -0.270 and -0.456
    - age in [25, 30) -> interpolate between -0.456 and -0.709
    - age in [30, 40) -> interpolate between -0.709 and -1.282
    - age in [40, 50) -> interpolate between -1.282 and -1.808
    - age >= 50       -> extrapolate using the trend from [40, 50)

    For example, if age=21, it lies between 20 and 25, so the score is
    interpolated between -0.270 and -0.456.

    Args:
        age: continuous age value

    Returns:
        interpolated score value
    """
    # Handle edge case: below minimum age
    if age < age_dist_map[0]:
        return _SCORE_AGE_DIST_VALS[0]

    # Handle edge case: above maximum age - extrapolate using last segment's trend
    if age >= age_dist_map[5]:  # age >= 50
        # Use the slope from the last segment [40, 50)
        age_lower = age_dist_map[4]  # 40
        age_upper = age_dist_map[5]  # 50
        score_lower = _SCORE_AGE_DIST_VALS[4]  # -1.282
        score_upper = _SCORE_AGE_DIST_VALS[5]  # -1.808
        slope = (score_upper - score_lower) / (age_upper - age_lower)
        # Extrapolate: score = score_upper + slope * (age - age_upper)
        return score_upper + slope * (age - age_upper)

    # Find the bracket: age is in [age_dist_map[i], age_dist_map[i+1])
    # and should interpolate between _SCORE_AGE_DIST_VALS[i] and _SCORE_AGE_DIST_VALS[i+1]
    for i in range(5):  # Only go up to i=4, so we access indices 0-5
        age_lower = age_dist_map[i]
        age_upper = age_dist_map[i + 1]

        if age_lower <= age < age_upper:
            score_lower = _SCORE_AGE_DIST_VALS[i]
            score_upper = _SCORE_AGE_DIST_VALS[i + 1]

            # Linear interpolation
            t = (age - age_lower) / (age_upper - age_lower)
            return score_lower + t * (score_upper - score_lower)

    # Should not reach here, but return the last score if we do
    return _SCORE_AGE_DIST_VALS[5]


def eval_score_fixed(row):
    score = 0.0
    # sex
    score += score_levels(row["sex"], _SCORE_SEX_VALS)
    # employment
    score += score_levels(row["employ"], _SCORE_EMPLOY_VALS)
    # drug
    score += score_levels(row["drug_abuse"], _SCORE_DRUG_VALS)
    # felony_prior_conviction
    score += score_levels(row["felony_prior_conviction"], _SCORE_FELONY_PRIOR_CONV_VALS)
    # race
    score += score_levels(row["race"], _SCORE_RACE_VALS)
    # ethnicity
    score += score_levels(row["ethnicity"], _SCORE_ETHNICITY_VALS)
    # offense_type
    score += score_levels(row["offense_type"], _SCORE_OFFENSE_TYPE_VALS)
    # supervision_level
    score += score_levels(row["supervision"], _SCORE_SUPERVISION_VALS)
    # ------------------------------------------------------------
    # fixed community level features (group II, continuous percentages)
    # -------------------------------------------------------------
    score += row["house_female"] * _SCORE_HOUSE_FEMALE_COEF
    # property tax column is not yet loaded in input.py:cols_attrs;
    # uncomment once available:
    score += row["tax_property"] * _SCORE_PROPERTY_TAX_COEF
    # -------------------------------------------------------------
    # there are other features that are not significant, see Table 3 of
    # Sirakaya's thesis, e.g., education:
    # score += score_levels(row["education"], [0.0, 0.57, -0.086, -0.188, -0.301])
    # ... and so on...
    # -------------------------------------------------------------
    return score


SIRAKAYA_COEFFS_COMM = [-0.003, 0.057]


def eval_score_comm(row, coeffs=SIRAKAYA_COEFFS_COMM):
    score = 0.0
    score += row["mean_re_time"] * coeffs[0]
    score += row["percent_re"] * coeffs[1]
    # ------------------------------------------------------------
    return score


# ------------------------------------------------------------------------------
# fill missing values
# ------------------------------------------------------------------------------
def fill_default(df):
    df["survival"] = df["survival"].fillna(df["survival"].max())
    df["supervision"] = df["supervision_level"].fillna(6).apply(lambda x: max(6 - x, 1))
    df["offense_type"] = df["offense_type"].map(ucr_to_category).fillna(8)
    cols_fill_1 = [
        "age_dist",
        "education",
        "sex",
        "race",
        "drug_abuse",
        "felony_prior_conviction",
        "ethnicity",
        "employ",
    ]
    df[cols_fill_1] = df[cols_fill_1].fillna(1)
    return df


def fill_emis_impute_df(
    df,
    cols_fill,
    n_imputations=5,
    n_components=3,
    max_iter=50,
    random_state=0,
):
    """
    EMis-like imputation on selected columns of a DataFrame.

    Args:
        df: pandas DataFrame
        cols_fill: list of column names to impute
        n_imputations: number of complete DataFrames to return
        n_components: number of GMM components
        max_iter: EM iterations
        random_state: random seed

    Returns:
        List of DataFrames, each with selected columns imputed
    """
    np.random.seed(random_state)
    # Subset the columns to fill
    X = df[cols_fill].to_numpy(dtype=float)

    # Initial mean imputation
    X_obs = np.copy(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        # The arithmetic mean is the sum of the non-NaN elements along the axis
        #   divided by the number of non-NaN elements.
        mean_val = np.nanmean(col)
        col[np.isnan(col)] = mean_val
        X_obs[:, j] = col

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=n_components, max_iter=max_iter, random_state=random_state
    )
    gmm.fit(X_obs)

    imputations = []
    for _ in range(n_imputations):
        X_imp = np.copy(X)
        for i in range(X.shape[0]):
            missing = np.isnan(X[i])
            if np.any(missing):
                mu = gmm.means_
                sigma = gmm.covariances_
                weights = gmm.predict_proba(X_obs[i].reshape(1, -1)).flatten()
                component = np.random.choice(np.arange(n_components), p=weights)

                mu_i = mu[component]
                sigma_i = sigma[component]
                sampled = np.random.multivariate_normal(mu_i, sigma_i)

                X_imp[i, missing] = sampled[missing]

        # Replace only the selected columns
        df_new = df.copy()
        df_new[cols_fill] = X_imp
        imputations.append(df_new)

    return imputations, df_new, gmm

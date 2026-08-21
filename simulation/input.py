import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sirakaya
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter
from simulation import Event

# Cache directory for pickle files
CACHE_DIR = os.path.join(os.path.dirname(__file__), "bin")

cols_attrs = [
    "sex",
    "age_dist",
    "race",
    "ethnicity",
    "marital",
    "education",
    "felony_prior_conviction",
    "drug_abuse",
    "offense_type",
    "employ",
    "probation_term",
    "pstatus",
    "population",
    "nonwhite",
    "hispanic",
    "house_female",
    "house_owner",
    "persons_hs",
    "persons_college",
    "persons_poverty",
    "rate_vote_republican",
    "crime_per_capita",
    "time",
    "observed",
    "supervision",
    "supervision_level",
]

cols_fill_1 = [
    "t_recid",
    "t_since_sentence",
    "observed",
    "age_dist",
    "education",
    "felony_arrest",
    "sex",
    "race",
    "drug_abuse",
    "felony_prior_conviction",
    "ethnicity",
    "employ",
    "supervision_level",
    "supervision",
]


def load_data(datadir="./", use_cache=True):
    """
    Load and process felony 1989 data.

    Args:
        datadir: Directory containing the felony-1989-final.xlsx file
        use_cache: If True, load from pickle cache if available, otherwise save to cache

    Returns:
        Tuple of (dfr, df, df_individual, df_community)
    """
    # Define cache file paths
    cache_files = {
        "dfr": os.path.join(CACHE_DIR, "dfr.pkl"),
        "df": os.path.join(CACHE_DIR, "df.pkl"),
        "df_individual": os.path.join(CACHE_DIR, "df_individual.pkl"),
        "df_community": os.path.join(CACHE_DIR, "df_community.pkl"),
    }

    # Check if all cache files exist
    all_cached = use_cache and all(os.path.exists(f) for f in cache_files.values())

    if all_cached:
        print("Loading data from pickle cache...")
        dfr = pd.read_pickle(cache_files["dfr"])
        df = pd.read_pickle(cache_files["df"])
        df_individual = pd.read_pickle(cache_files["df_individual"])
        df_community = pd.read_pickle(cache_files["df_community"])
        return dfr, df, df_individual, df_community

    # Load from Excel
    print("Loading data from Excel file...")
    ff = pd.ExcelFile(f"{datadir}/felony-1989-final.xlsx", engine="openpyxl")
    dfr = ff.parse("main", index_col="id")
    dfr_at = ff.parse("artificial", index_col="id")

    dfr = pd.merge(dfr, dfr_at, how="left", on="id")

    dfr = dfr.assign(
        na_t_recid=lambda df: df["t_recid"].isna(),
        na_code_recid=lambda df: df["code_recid"].isna(),
        na_num_recid=lambda df: df["num_recid"].isna(),
        # status = 1 means still on probation
        unknown_st=lambda df: df["pstatus"].isna(),
        end_status=lambda df: 1 - df["pstatus"].fillna(1),
        # you recidivate or you did not and your probation ends
        observed=lambda df: df["num_recid"],
        supervision=lambda df: df.apply(
            lambda row: (
                np.nan
                if np.isnan(row["supervision_level"])
                else 7 - row["supervision_level"]
            ),
            axis=1,
        ),
    )
    dfr["probation_term"] = dfr["probation_term"].fillna(36)
    dfr["offense_type"] = dfr["offense_type"].map(sirakaya.ucr_to_category).fillna(8)

    # use gaussian mixture model to impute missing values
    new_cols, df, gmm = sirakaya.fill_emis_impute_df(
        dfr.query("bool_include==True").reset_index(), cols_fill_1
    )
    # use time since sentence if observed is 0
    df = df.assign(
        time=lambda df: df.apply(
            lambda row: (
                row["t_recid"] if row["observed"] else row["t_since_sentence"]
            ),
            axis=1,
        ),
        # this part is generally unchanged
        score_fixed=lambda df: df.apply(sirakaya.eval_score_fixed, axis=1),
        score_age_dist=lambda df: df["age_dist"].apply(sirakaya.score_age_dist),
    ).dropna(subset=["score_fixed"])

    # community data
    df_community = (
        df.groupby("code_county")
        .agg({"time": "mean", "observed": "mean", "age_dist": "count"})
        .rename(columns={"age_dist": "size", "time": "mean_re_time"})
        .assign(
            percent_re=lambda x: x["observed"],
            score_comm=lambda x: x.apply(sirakaya.eval_score_comm, axis=1),
        )
    )
    dfc_dict = df_community["score_comm"].to_dict()
    min_pstart = df["pstart"].min()
    df_individual = df.assign(
        rel_pstart=lambda df: df["pstart"] - min_pstart,
        rel_probation=lambda df: df["probation_term"] * 30,
        offenses=lambda df: df["felony_arrest"].apply(np.round),
    )[
        [
            "observed",
            "time",
            "code_county",
            "rel_pstart",
            "rel_probation",
            "offenses",
            "age_dist",
            "score_fixed",
            "score_age_dist",
            "prison_rate",
        ]
    ].assign(
        score_comm=lambda x: x["code_county"].map(dfc_dict),
        offset=lambda x: x["score_fixed"] + x["score_comm"] + x["score_age_dist"],
        score=lambda x: x["offset"],
        time=lambda x: x["time"] + 1,
        age=lambda x: x["age_dist"].apply(sirakaya.age_dist_to_age),
    )
    # avoid float after imputation
    df_individual["offenses"] = df_individual["offenses"].apply(np.round)
    df_individual["score_offenses"] = df_individual["offenses"]

    # Save to cache if use_cache is enabled
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"Saving data to pickle cache in {CACHE_DIR}...")
        dfr.to_pickle(cache_files["dfr"])
        df.to_pickle(cache_files["df"])
        df_individual.to_pickle(cache_files["df_individual"])
        df_community.to_pickle(cache_files["df_community"])

    return dfr, df, df_individual, df_community


from matplotlib.colors import ListedColormap

rbgs = [
    (0.0, 0.6056031611752245, 0.9786801175696073),
    (0.8888735002725197, 0.4356491903481899, 0.2781229361419437),
    (0.2422242978521988, 0.6432750931576305, 0.3044486515341153),
    (0.7644401754934357, 0.44411177946877667, 0.8242975359232757),
    (0.6755439572114058, 0.5556623322045815, 0.09423433626639476),
    (4.821181565883848e-7, 0.6657589812923558, 0.6809969518707946),
    (0.930767491919665, 0.3674771896571418, 0.5757699667547833),
    (0.7769816661712935, 0.5097431319944512, 0.1464252569555494),
    (3.8077343912812365e-7, 0.6642678029460113, 0.5529508754522481),
    (0.558464964115081, 0.5934846564332881, 0.11748125233232112),
    (5.947623876556563e-7, 0.6608785231434255, 0.7981787608414301),
    (0.6096707676128643, 0.49918492100827794, 0.9117812665042643),
    (0.3800016049820355, 0.5510532724353505, 0.9665056985227145),
    (0.9421816479542178, 0.37516423354097606, 0.4518168202944591),
    (0.8684020893043973, 0.3959893639954848, 0.7135147524811882),
    (0.423146743646308, 0.6224954944199984, 0.1987706025213047),
]
juliacmaps = ListedColormap(rbgs)
# Configure rcParams to use LaTeX and serif font
colors = juliacmaps(np.linspace(0, 1, len(rbgs)))  # Get all colors from the colormap
# plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
# plt.rcParams.update(
#     {
#         "text.usetex": True,  # Use LaTeX for text rendering
#         "font.family": "serif",  # Use serif font family
#         "font.serif": [
#             "Computer Modern Roman"
#         ],  # Specify specific serif font (default LaTeX font)
#         "font.size": 18,  # General font size
#         "axes.labelsize": 12,  # Font size for labels
#         "legend.fontsize": 18,  # Font size for legend
#         "xtick.labelsize": 18,  # Font size for x-axis ticks
#         "ytick.labelsize": 18,  # Font size for y-axis ticks
#         # "image.cmap": "juliacmaps",  # Set custom colormap
#     }
# )

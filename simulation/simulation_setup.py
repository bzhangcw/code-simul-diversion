"""
Simulation setup module - provides common configuration for running policies.

Usage:
    from simulation_setup import default_settings, get_tests

    # Access settings
    default_settings.p_freeze
    default_settings.treatment_capacity

    # Get test configurations
    tests = get_tests(default_settings)

Note:
    run_sim() and run_name() functions have been moved to run_policy.py

Environment variables:
    DBG: 0 = formal run (default), 1 = debug mode with more arrivals
"""

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import simulation
import simulation_treatment as smt
import sirakaya
import input

# here we will implicit allow treatment in the middle of the probation
QUAL_STRING_ANYTIME = (
    "bool_left == 0 "
    + "& stage == 'p'"
    + "& bool_can_be_treated == 1"
    + "& has_been_treated == 0"
)
# here we will only allow treatment upon arrival to probation
# this means, the decision is not made yet
QUAL_STRING_ENTRY_ONLY = (
    "bool_left == 0 "
    + "& stage == 'p'"
    + "& bool_can_be_treated == 1"
    + "& has_been_treated == 0"
    + "& bool_decision_made == 0"
)


class SimulationSetup:
    """Configuration class for simulation parameters."""

    # Class-level data (shared across all instances to avoid reloading)
    _data_loaded = False
    _dfr = None
    _df = None
    _df_individual = None
    _df_community = None
    _simulator = None

    def __init__(self, print_summary=True):
        # Debug mode from environment
        self.dbg = int(os.environ.get("DBG", "0"))

        # Default values
        self.p_freeze = 2
        self.p_freeze_policy = 10  # number of periods to run before updating the policy
        # @note: previously we use 400 x 40000 is quite stable.
        self.p_length = 100
        self.T_max = 40000
        self.treatment_capacity = 80
        self.treatment_effect = lambda _: 0.5
        """@note: try not change to 1
        1: vital for memory need about 5G RAM
        2: 1 will be very slow to converge (need many epochs))
        """
        self.beta_arrival = 5
        self.beta_initial = 5
        self.communities = {1}
        self.rel_off_probation = 2000
        self.fit_kwargs = {
            "new_col": "offenses",
            "bool_use_cph": False,
            "baseline": "exponential",
        }
        self.func_to_call = simulation.arrival_constant
        self.state_defs = simulation.StateDefs(
            ["offenses", "age_dist", "has_been_treated", "stage"]
        )

        # Treatment eligibility and enrollment settings
        self.max_returns = 25
        self.max_offenses = 25
        # @note: decide who can be sent to treatment
        # treatment_timing: "upon_arrival" (default) or "anytime"
        self.treatment_timing = "upon_arrival"
        self.str_qualifying_for_treatment = (
            QUAL_STRING_ENTRY_ONLY
            if self.treatment_timing == "upon_arrival"
            else QUAL_STRING_ANYTIME
        )
        # @note: decide who is currently enrolled in treatment
        self.str_current_enroll = "bool_left == 0 & bool_treat == 1 & stage == 'p'"

        # Boolean flags
        self.bool_return_can_be_treated = 1
        self.bool_keep_treatment_effect = 1
        self.bool_has_incarceration = 1
        self.bool_keep_idiosyncratic_effect = 1
        # @note, this is more stable.
        self.bool_mean_use_accumulated_offenses = 1

        # Scalers
        self.prison_rate_scaler = 1.0
        self.length_scaler = 1.0

        # Instance-specific data (derived from class-level shared data)
        self.dfc = None
        self.dfi = None
        self.dfpop0 = None

        # Adjust settings based on debug mode
        if self.dbg == 1:
            self.beta_arrival = 10
            self.p_length = 250
            self.T_max = 1000

        self._load_data()

        if print_summary:
            self._print_summary()

    def _print_summary(self):
        """Print summary of main statistics and settings."""
        n_episodes = self.T_max // self.p_length
        state_dims = self.state_defs.state_keys

        print("=" * 140)
        print("SimulationSetup Summary")
        print("=" * 140)
        print(
            f"  Mode:              {'DEBUG' if self.dbg else 'FORMAL'} (DBG={self.dbg})"
        )
        print("-" * 140)
        print("Simulation Parameters:")
        print(f"  T_max:             {self.T_max:,} time units")
        print(f"  p_length:          {self.p_length} (episode length)")
        print(f"  p_freeze:          {self.p_freeze} (freeze period)")
        print(f"  p_freeze_policy:   {self.p_freeze_policy} (policy freeze period)")
        print(f"  off_prob:          {self.rel_off_probation} (off-probation term)")
        print(f"  n_episodes:        ~{n_episodes}")
        print("-" * 140)
        print("Treatment Settings:")
        print(f"  capacity:          {self.treatment_capacity} per episode")
        effect_text = getattr(self, "treatment_help_text", str(self.treatment_effect))
        # Handle multi-line docstrings
        effect_lines = effect_text.strip().split("\n")
        print(f"  effect:            {effect_lines[0].strip()}")
        for line in effect_lines[1:]:
            print(f"                     {line.strip()}")
        dosage_text = getattr(
            self,
            "treatment_dosage_help_text",
            str(getattr(self, "treatment_dosage", "default")),
        )
        # Handle multi-line docstrings
        dosage_lines = dosage_text.strip().split("\n")
        print(f"  dosage:            {dosage_lines[0].strip()}")
        for line in dosage_lines[1:]:
            print(f"                     {line.strip()}")
        print(f"  max_returns:       {self.max_returns}")
        print(f"  max_offenses:      {self.max_offenses}")
        print(f"  return_can_treat:  {self.bool_return_can_be_treated}")
        print(f"  has_incarceration: {self.bool_has_incarceration}")
        print(
            f"  keep_idiosync:     {self.bool_keep_idiosyncratic_effect} ({'individual score_fixed' if self.bool_keep_idiosyncratic_effect else 'mean score_fixed for all'})"
        )
        print(
            f"  use_accum_stats:   {self.bool_mean_use_accumulated_offenses} ({'cumulative stats' if self.bool_mean_use_accumulated_offenses else 'current-episode stats'} for community score)"
        )
        print(
            f"  prison_scaler:     {self.prison_rate_scaler} (scaler for prison rate)"
        )
        print(
            f"  length_scaler:     {self.length_scaler} (scaler for probation/off-probation length)"
        )
        print(f"  qualifying:        {self.str_qualifying_for_treatment}")
        print(f"  current_enroll:    {self.str_current_enroll}")
        print("-" * 140)
        print("Population & Arrivals:")
        print(f"  beta_arrival:      {self.beta_arrival} (arrival rate)")
        print(f"  beta_initial:      {self.beta_initial} (initial population size)")
        print(f"  communities:       {self.communities}")
        print(f"  initial pop:       {len(self.dfpop0)} individuals")
        print(f"  available pool:    {len(self.dfi)} individuals")
        print(
            self.dfi[
                ["age", "weight", "prison_rate", "score_fixed", "score_comm"]
            ].head(5)
        )
        print("-" * 140)
        print("State Space:")
        print(f"  dimensions:        {state_dims}")
        print(f"  n_states:          {len(self.state_defs.state_key_range)}")
        _fmt_ssw = {
            k: f"{v:.4f}" for k, v in self.simulator.state_defs.scoring_weights.items()
        }
        _fmt_sw = {k: f"{v:.4f}" for k, v in self.simulator._scoring_weights.items()}
        print(f"  state weights:     {_fmt_ssw}")
        print(f"  final weights:     {_fmt_sw}")
        print("-" * 140)
        sw = self.simulator._scoring_weights
        ssw = self.simulator.state_defs.scoring_weights
        print("Scoring Equation:")
        print(f"  score = score_fixed     × {sw['score_fixed']:.4f}")
        print(f"        + score_comm      × {sw['score_comm']:.4f}")
        print(f"        + score_state     × {sw['score_state']:.4f}")
        print(f"        + score_treatment × {sw['score_treatment']:.4f}")
        print()
        state_terms = [
            f"score_{k} × {ssw[f'score_{k}']:.4f}"
            for k in self.state_defs.state_keys
            if f"score_{k}" in ssw
        ]
        if state_terms:
            print(f"    where score_state = {state_terms[0]}")
            for term in state_terms[1:]:
                print(f"                      + {term}")
        print()
        print(f"          score_comm  = mean_re_time × (-0.003)")
        print(f"                      + percent_re   × 0.057")
        print()
        print(f"  S(t | score) = S₀(t) ^ exp(score)")
        print("=" * 140)

    def _load_data(self, ignore_communities=True):
        """Load data and initialize simulator."""
        if not SimulationSetup._data_loaded:
            # Load raw data once at class level
            (
                SimulationSetup._dfr,
                SimulationSetup._df,
                SimulationSetup._df_individual,
                SimulationSetup._df_community,
            ) = input.load_data()

            # Create initial simulator for scoring
            SimulationSetup._simulator = simulation.Simulator(
                eval_score_fixed=sirakaya.eval_score_fixed,
                eval_score_comm=sirakaya.eval_score_comm,
                func_arrival=self.arrival_func,
                func_leaving=None,
                state_defs=self.state_defs,
            )
            simulation.refit_baseline(
                SimulationSetup._simulator,
                SimulationSetup._df_individual,
                **self.fit_kwargs,
            )
            SimulationSetup._data_loaded = True

        # Reference class-level data in instance
        self.dfr = SimulationSetup._dfr
        self.df = SimulationSetup._df
        self.df_individual = SimulationSetup._df_individual
        self.df_community = SimulationSetup._df_community
        self.simulator = SimulationSetup._simulator
        # Filter by communities
        self.dfc = self.df_community[
            self.df_community.index.isin(self.communities)
        ].copy()

        # if ignore_community, we change the community to {communities}
        # and then we can sample from the entire population
        if ignore_communities:
            _valid_population = self.df_individual.assign(
                code_county=list(self.communities)[0]
            )
        else:
            _valid_population = self.df_individual[
                self.df_individual["code_county"].isin(self.communities)
            ]

        # here we should assign a few needed new items
        self.dfi = (
            _valid_population.reset_index(drop=True)
            .copy()
            .assign(
                stage="p",
                has_been_treated=0,  # whether the individual has been treated
                state=lambda x: x.apply(
                    lambda row: self.state_defs.get_state(row), axis=1
                ),
                score_state=lambda x: x.apply(
                    lambda row: self.state_defs.get_score(row), axis=1
                ),
                score_treatment=0.0,
            )
            .assign(
                score_fixed=lambda df: (
                    df["score_fixed"]
                    if self.bool_keep_idiosyncratic_effect
                    else df["score_fixed"].mean()
                ),
                score=lambda df: df.apply(
                    lambda row: self.simulator.produce_score(row.name, df), axis=1
                ),
            )
        )

        # Sample initial population
        self.assign_weights()
        self.dfpop0 = self.dfi.sample(self.beta_initial)

    def assign_weights(self):
        """Assign weights to the individuals so I sample according to the weights."""
        self.dfi["weight"] = 1.0

        return self.dfi

    def arrival_func(self, *args, **kwargs):
        return self.func_to_call(self.beta_arrival, *args, **kwargs)


# Create default settings instance
# Note: Call _print_summary() manually if you need to see the configuration
default_settings = SimulationSetup(print_summary=False)


# ------------------------------------------------------------
# define policy configurations to test
# ------------------------------------------------------------
def create_lowj_lowa_prob_vector(j_max=4, a_max=2):
    pr = {}
    for j in range(j_max):
        for a in range(a_max):
            pr[(j, a, 1.0)] = 1.0
    return pr


def create_threshj_lowa_prob_vector(j_min=2, a_max=2, j_max=20):
    pr = {}
    for j in range(j_min, j_max):
        for a in range(a_max):
            pr[(j, a, 1.0)] = 1.0
    return pr


pr_lowj_lowa = create_lowj_lowa_prob_vector()
pr_threshj_lowa = create_threshj_lowa_prob_vector()
thres_cutoff = {
    1: 4,  # if in age-group 1, cut-off below 1
    2: 2,  # if in age-group 2, cut-off below 2
    3: 1,  # if in age-group 3, cut-off below 3
    4: 1,
    5: 0,
    6: 0,
}


# ------------------------------------------------------------
# get test configurations
# ------------------------------------------------------------
def get_tests(settings=None):
    """Return test configurations using given settings."""
    if settings is None:
        settings = default_settings

    return {
        "null": (smt.treatment_null, dict()),
        "random": (
            smt.treatment_rule_random,
            dict(capacity=settings.treatment_capacity),
        ),
        "high-risk": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=False,
            ),
        ),
        "low-risk": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=True,
            ),
        ),
        "age-first": (
            smt.treatment_rule_priority,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                ascending=True,
                effect=settings.treatment_effect,
                to_compute=lambda df: df.apply(
                    lambda row: (row["age"], row["offenses"]), axis=1
                ),
            ),
        ),
        "age-first-high-risk": (
            smt.treatment_rule_priority,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                ascending=True,
                effect=settings.treatment_effect,
                to_compute=lambda df: df.apply(
                    lambda row: (row["age"], -row["score_state"]), axis=1
                ),
            ),
        ),
        "low-risk-young-first": (
            smt.treatment_rule_priority,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                ascending=True,
                effect=settings.treatment_effect,
                to_compute=lambda df: df.apply(
                    lambda row: (row["score"], row["age"]), axis=1
                ),
            ),
        ),
        "high-risk-young-first": (
            smt.treatment_rule_priority,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                ascending=True,
                effect=settings.treatment_effect,
                to_compute=lambda df: df.apply(
                    lambda row: (-row["score"], row["age"]), axis=1
                ),
            ),
        ),
        "high-risk-cutoff": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=False,
                exclude=lambda row: row["score"] >= -1.414,
            ),
        ),
        "age-tolerance": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=False,
                exclude=lambda row: row["offenses"] < thres_cutoff[row["age_dist"]],
            ),
        ),
        "high-risk-lean-young": (
            smt.treatment_rule_priority,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                ascending=True,
                effect=settings.treatment_effect,
                to_compute=lambda df: df.apply(
                    lambda row: row["score"] + row["age_dist"] * 0.05, axis=1
                ),
            ),
        ),
        # ------------------------------------------------------------
        # fluid/probabilistic policies
        # ------------------------------------------------------------
        # "fluid-low-age-low-prev": (
        #     smt.treatment_rule_priority_fluid,
        #     dict(
        #         key="_new_prior",
        #         capacity=settings.treatment_capacity,
        #         prob_vector=pr_lowj_lowa,
        #     ),
        # ),
        # "fluid-low-age-threshold-offenses": (
        #     smt.treatment_rule_priority_fluid,
        #     dict(
        #         key="_new_prior",
        #         capacity=settings.treatment_capacity,
        #         prob_vector=pr_threshj_lowa,
        #     ),
        # ),
    }


# Default tests using default_settings
tests = get_tests()

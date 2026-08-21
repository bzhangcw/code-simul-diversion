"""
Run simulation policies with configurable parameters.

Example usage:
    # Basic usage with default parameters
    python run_policy.py high-risk 0 results

    # With custom parameters
    python run_policy.py high-risk 0 results --T_max 20000 --treatment_capacity 100

    # Show help and all available options
    python run_policy.py -h
"""

import matplotlib.pyplot as plt

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns

import json
import plotly

import simulation, sirakaya

# Import what we need from simulation_setup
from simulation_setup import (
    SimulationSetup,
    get_tests,
    default_settings,
    QUAL_STRING_ANYTIME,
    QUAL_STRING_ENTRY_ONLY,
)
from simulation_summary import *
import simulation_treatment as smt


def create_parser():
    """Create argument parser with default parameters from SimulationSetup."""
    parser = argparse.ArgumentParser(
        description="Run a simulation policy with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Get available policies from default settings
    available_policies = ", ".join(get_tests(default_settings).keys())

    # Required arguments
    parser.add_argument(
        "policy_name",
        type=str,
        help=f"Policy name to run. Available: {available_policies}",
    )
    parser.add_argument(
        "rep",
        type=int,
        help="Repetition number; also used as the random seed",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Base directory for output files",
    )

    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument(
        "--T_max",
        type=int,
        default=default_settings.T_max,
        help="Maximum simulation time in days",
    )
    sim_group.add_argument(
        "--p_length",
        type=int,
        default=default_settings.p_length,
        help="Length of each episode in days",
    )
    sim_group.add_argument(
        "--p_freeze",
        type=int,
        default=default_settings.p_freeze,
        help="Number of periods to freeze before updating scores",
    )
    sim_group.add_argument(
        "--p_freeze_policy",
        type=int,
        default=default_settings.p_freeze_policy,
        help="Number of periods to run before updating the policy",
    )
    sim_group.add_argument(
        "--rel_off_probation",
        type=int,
        default=default_settings.rel_off_probation,
        help="Off-probation term in days",
    )
    sim_group.add_argument(
        "--beta_arrival",
        type=float,
        default=default_settings.beta_arrival,
        help="Arrival rate parameter",
    )
    sim_group.add_argument(
        "--beta_initial",
        type=int,
        default=default_settings.beta_initial,
        help="Initial population size",
    )

    # Treatment parameters
    treat_group = parser.add_argument_group("Treatment Parameters")
    treat_group.add_argument(
        "--treatment_capacity",
        type=int,
        default=default_settings.treatment_capacity,
        help="Treatment capacity per episode",
    )
    treat_group.add_argument(
        "--treatment_effect",
        type=str,
        default="0.5",
        help="Treatment effect: a number (0-1) for homogeneous effect, or 'type-1' for heterogeneous effect",
    )
    treat_group.add_argument(
        "--treatment_dosage",
        type=str,
        default="default",
        help="Treatment dosage: 'default' (1.0 for all) or 'type-1' for heterogeneous",
    )
    treat_group.add_argument(
        "--max_returns",
        type=int,
        default=default_settings.max_returns,
        help="Maximum number of returns allowed",
    )

    treat_group.add_argument(
        "--max_offenses",
        type=int,
        default=default_settings.max_offenses,
        help="Maximum number of offenses to track",
    )

    treat_group.add_argument(
        "--treatment_timing",
        type=str,
        choices=["upon_arrival", "anytime"],
        default=default_settings.treatment_timing,
        help="When treatment can be assigned: 'upon_arrival' (only at entry to probation) or 'anytime' (during probation)",
    )

    # Boolean flags
    flag_group = parser.add_argument_group("Boolean Flags")
    flag_group.add_argument(
        "--bool_return_can_be_treated",
        type=int,
        choices=[0, 1],
        default=default_settings.bool_return_can_be_treated,
        help="Whether returning individuals can be treated",
    )
    flag_group.add_argument(
        "--bool_has_incarceration",
        type=int,
        choices=[0, 1],
        default=default_settings.bool_has_incarceration,
        help="Whether incarceration is enabled",
    )
    flag_group.add_argument(
        "--bool_keep_idiosyncratic_effect",
        type=int,
        choices=[0, 1],
        default=default_settings.bool_keep_idiosyncratic_effect,
        help="Keep individual score_fixed (1) or replace with population mean (0)",
    )
    flag_group.add_argument(
        "--bool_mean_use_accumulated_offenses",
        type=int,
        choices=[0, 1],
        default=default_settings.bool_mean_use_accumulated_offenses,
        help="Use cumulative stats (1) or current-episode stats (0) for community score",
    )

    # Scalers
    scaler_group = parser.add_argument_group("Scalers")
    scaler_group.add_argument(
        "--prison_rate_scaler",
        type=float,
        default=default_settings.prison_rate_scaler,
        help="Scaler for prison rate probability",
    )
    scaler_group.add_argument(
        "--length_scaler",
        type=float,
        default=default_settings.length_scaler,
        help="Scaler for probation and off-probation term lengths",
    )

    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Verbosity level (0=silent, 1=progress, 2=detailed)",
    )
    other_group.add_argument(
        "--option_weights",
        type=str,
        choices=["younger"],
        default=None,
        help="Option for assigning weights to population (e.g., 'younger' for age-based weights)",
    )

    return parser


def run_sim(
    name,
    func_treatment=None,
    func_treatment_kwargs=None,
    seed=1,
    verbosity=2,
    output_dir=None,
    settings=None,
):
    """Run a single simulation with given treatment function."""
    if settings is None:
        settings = default_settings

    simulator = simulation.Simulator(
        eval_score_fixed=sirakaya.eval_score_fixed,
        eval_score_comm=sirakaya.eval_score_comm,
        func_arrival=settings.arrival_func,
        func_leaving=None,
        state_defs=settings.state_defs,
    )
    simulation.refit_baseline(simulator, settings.df_individual, **settings.fit_kwargs)

    # Determine log path
    log_path = f"{output_dir}/log.txt" if output_dir else f"results/log.dyn.{name}.log"

    simulator.run(
        seed=seed,
        dfi=settings.dfpop0.copy(),
        dfpop=settings.dfi.copy(),
        fo=open(log_path, "w"),
        opt_verbosity=verbosity,
        T_max=settings.T_max,
        p_length=settings.p_length,
        p_freeze=settings.p_freeze,
        p_freeze_policy=settings.p_freeze_policy,
        rel_off_probation=settings.rel_off_probation,
        treatment_effect=settings.treatment_effect,
        func_treatment=func_treatment,
        func_treatment_kwargs=func_treatment_kwargs,
        func_dosage=settings.treatment_dosage,
        str_qualifying_for_treatment=settings.str_qualifying_for_treatment,
        str_current_enroll=settings.str_current_enroll,
        bool_return_can_be_treated=settings.bool_return_can_be_treated,
        max_returns=settings.max_returns,
        max_offenses=settings.max_offenses,
        bool_has_incarceration=settings.bool_has_incarceration,
        bool_mean_use_accumulated_offenses=settings.bool_mean_use_accumulated_offenses,
        prison_rate_scaler=settings.prison_rate_scaler,
        length_scaler=settings.length_scaler,
    )
    if verbosity > 0:
        print(
            "people appeared/new",
            simulator.n_persons_appeared,
            simulator.n_persons_appeared - simulator.n_persons_initial,
        )
    return simulator


def run_name(name, rep, output_dir="results", settings=None, verbosity=2):
    """
    Run one policy repetition and save it under its repetition number.

    Args:
        name: policy name (must be a key in tests dict)
        rep: repetition number, also used as the random seed
        output_dir: base directory for outputs
        settings: SimulationSetup instance (default: default_settings)
        verbosity: simulation output level

    Returns:
        The completed Simulator instance.
    """
    if settings is None:
        settings = default_settings

    from simulation_summary import dump_rep_metrics

    policy_tests = get_tests(settings)
    policy_dir = f"{output_dir}/{name}"
    os.makedirs(policy_dir, exist_ok=True)

    rep_dir = f"{policy_dir}/{rep}"
    os.makedirs(rep_dir, exist_ok=True)

    sim = run_sim(
        name,
        policy_tests[name][0],
        policy_tests[name][1],
        seed=rep,
        verbosity=verbosity,
        output_dir=rep_dir,
        settings=settings,
    )
    dump_rep_metrics(sim, rep_dir, settings.p_freeze)

    return sim


def assign_weights_younger_population(settings):
    """Sample a younger population from the initial population."""
    settings.dfi["weight"] = settings.dfi["age"].apply(lambda x: 100 - x)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Create custom settings (data will be reused from class-level cache)
    # Don't print summary during init - we'll print after updating parameters
    settings = SimulationSetup(print_summary=False)

    # Update settings with command-line arguments
    settings.T_max = args.T_max
    settings.p_length = args.p_length
    settings.p_freeze = args.p_freeze
    settings.p_freeze_policy = args.p_freeze_policy
    settings.rel_off_probation = args.rel_off_probation
    settings.beta_arrival = args.beta_arrival
    settings.beta_initial = args.beta_initial
    settings.bool_return_can_be_treated = args.bool_return_can_be_treated
    settings.bool_has_incarceration = args.bool_has_incarceration
    settings.bool_keep_idiosyncratic_effect = args.bool_keep_idiosyncratic_effect
    settings.bool_mean_use_accumulated_offenses = (
        args.bool_mean_use_accumulated_offenses
    )
    if not args.bool_keep_idiosyncratic_effect:
        settings.dfi["score_fixed"] = settings.dfi["score_fixed"].median()
        settings.dfi["score"] = settings.dfi.apply(
            lambda row: settings.simulator.produce_score(row.name, settings.dfi),
            axis=1,
        )
    settings.length_scaler = args.length_scaler
    settings.prison_rate_scaler = args.prison_rate_scaler
    settings.max_offenses = args.max_offenses
    settings.max_returns = args.max_returns
    settings.treatment_capacity = args.treatment_capacity
    settings.treatment_dosage = args.treatment_dosage

    # Set treatment timing mode
    settings.treatment_timing = args.treatment_timing
    settings.str_qualifying_for_treatment = (
        QUAL_STRING_ENTRY_ONLY
        if args.treatment_timing == "upon_arrival"
        else QUAL_STRING_ANYTIME
    )

    # Convert treatment_effect arg to a callable
    if args.treatment_effect == "type-1":
        settings.treatment_effect = smt.treatment_effect_type_1
        settings.treatment_help_text = smt.treatment_effect_type_1.__doc__
    elif args.treatment_effect == "type-2":
        settings.treatment_effect = smt.treatment_effect_type_2
        settings.treatment_help_text = smt.treatment_effect_type_2.__doc__
    elif args.treatment_effect == "type-3":
        settings.treatment_effect = smt.treatment_effect_type_3
        settings.treatment_help_text = smt.treatment_effect_type_3.__doc__
    else:
        # Assume it's a number for homogeneous effect
        effect_value = float(args.treatment_effect)
        settings.treatment_effect = lambda row, e=effect_value, **kwargs: np.log(e)
        settings.treatment_help_text = f"Homogeneous treatment effect: input {effect_value}, translate to score as +log({effect_value}) = {np.log(effect_value)}"

    # Convert treatment_effect arg to a callable
    if args.treatment_dosage == "type-1":
        settings.treatment_dosage = smt.treatment_dosage_type_1
        settings.treatment_dosage_help_text = smt.treatment_dosage_type_1.__doc__
    else:
        settings.treatment_dosage = smt.treatment_dosage_default
        settings.treatment_dosage_help_text = smt.treatment_dosage_default.__doc__

    # Resample initial population with updated beta_initial
    settings.dfpop0 = settings.dfi.sample(settings.beta_initial)

    if args.option_weights == "younger":
        assign_weights_younger_population(settings)

    # Print summary with final parameters
    settings._print_summary()

    # Validate policy name
    available_policies = get_tests(settings)
    if args.policy_name not in available_policies:
        print(f"Error: Unknown policy '{args.policy_name}'")
        print(f"Available policies: {', '.join(available_policies.keys())}")
        sys.exit(1)

    # Run the simulation
    run_name(
        args.policy_name,
        rep=args.rep,
        output_dir=args.output_dir,
        settings=settings,
        verbosity=args.verbosity,
    )

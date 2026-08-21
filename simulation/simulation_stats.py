import queue
from collections import defaultdict
from enum import IntEnum, Enum
import warnings
import numpy as np
import pandas as pd
import itertools
from lifelines import CoxPHFitter
from pyexcelerate import Workbook
from scipy.interpolate import interp1d

import sirakaya


def summarize_trajectory(
    self,
    p_freeze,
    state_lst_columns=["state_lst"],
    state_columns=["state"],
    window_for_ratios=10,
):
    """
    Compute trajectory statistics using arrest count differences between periods;
        this is a double-checking mechanism to ensure the correctness of the statistics.

    For each period p (starting from 1):
    - population_start: population at start of period (from snapshot p-1), grouped by state at p-1
    - population_end: population at end of period (from snapshot p), grouped by state at p
    - total_offenses/total_offenses_out: offense counts (j_p - j_{p-1}), grouped by state at p / p-1
    - total_offenders/total_offenders_out: distinct individuals who offended, grouped by state at p / p-1
    """
    snaps = len(self.t_dfi)
    results = []
    results_df = []
    results_flow = []
    results_retention = []
    error_accumulated = 0.0
    # start from p=1 since we need p-1 for comparison
    for p in range(1, snaps):
        # df_prev: snapshot at p-1 (start of period)
        # df_curr: snapshot at p (end of period)
        df_prev = (
            self.t_dfi[p - 1]
            .copy()
            .dropna(subset=["state", "state_lst"])
            .assign(
                state=lambda df: df["state"].apply(lambda x: x.value),
                state_lst=lambda df: df["state_lst"].apply(lambda x: x.value),
            )
        )
        df_curr = (
            self.t_dfi[p]
            .copy()
            .dropna(subset=["state", "state_lst"])
            .assign(
                state=lambda df: df["state"].apply(lambda x: x.value),
                state_lst=lambda df: df["state_lst"].apply(lambda x: x.value),
            )
        )

        lst_cols_avail = [c for c in state_lst_columns if c in df_prev.columns]
        cur_cols_avail = [c for c in state_columns if c in df_curr.columns]
        grp_keys_lst = lst_cols_avail
        grp_keys_cur = cur_cols_avail

        # _boundary is the latest _episode value reachable inside df_curr.
        # The summary for _episode = k+1 fires at t = (k+1)*p_length and snapshots
        # dfi as t_dfi[k]; by that moment, all events that have fired carry
        # _episode in {0, ..., k} (events with _episode = k+1 haven't fired yet,
        # they're scheduled for t >= (k+1)*p_length). So new events visible in
        # df_curr but not in df_prev = t_dfi[p-1] are exactly those with
        # _episode == p. p_freeze is retained only for API back-compat.
        _boundary = p
        _boundary_prev = p - 1

        # x0: present at start of period (using p-1 snapshot), grouped by state at p-1
        _present_start_mask = (df_prev["ep_arrival"] <= _boundary_prev) & (
            df_prev["ep_leaving"] > _boundary_prev
        )
        df_presence_begin = df_prev.loc[_present_start_mask]
        x0 = df_presence_begin.groupby(grp_keys_cur)["index"].count()

        # x: present at end of period (using p snapshot), grouped by state at p
        _present_end_mask = (df_curr["ep_arrival"] <= _boundary) & (
            df_curr["ep_leaving"] > _boundary
        )
        x = df_curr.loc[_present_end_mask].groupby(grp_keys_cur)["index"].count()

        # -----------------------------------------------------------------
        # Compute y using arrest count difference: j_p - j_{p-1}
        # -----------------------------------------------------------------
        # Get individuals present in both periods
        idx_common = df_presence_begin.index.intersection(df_curr.index)

        # Create merged df with state at p-1, state at p, and offenses at both
        df_merged = pd.DataFrame(
            {
                "index": idx_common,
                "state_prev": df_prev.loc[idx_common, "state"],
                "state_curr": df_curr.loc[idx_common, "state"],
                "j_prev": df_prev.loc[idx_common, "offenses"],
                "j_curr": df_curr.loc[idx_common, "offenses"],
            }
        ).assign(offenses=lambda df: df["j_curr"] - df["j_prev"])

        # y_out: offenses aggregated by state at p-1 (outflow from state)
        _df_offended = df_merged[df_merged["offenses"] > 0]
        yout = (
            _df_offended.groupby("state_prev")["offenses"]
            .sum()
            .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
        )

        # y_in: offenses aggregated by state at p (inflow to state)
        yin = (
            _df_offended.groupby("state_curr")["offenses"]
            .sum()
            .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
        )

        # observed_yout/observed_yin: number of distinct individuals who offended
        # (at least once this period), grouped by state at p-1 / p respectively
        observed_yout = (
            _df_offended.groupby("state_prev")["index"]
            .count()
            .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
        )
        observed_yin = (
            _df_offended.groupby("state_curr")["index"]
            .count()
            .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
        )

        # arrivals and departures
        # arrivals: people arriving at end of period (ep_arrival == _boundary)
        _arrive_now_mask = df_curr["ep_arrival"] == _boundary
        lmd = df_curr.loc[_arrive_now_mask].groupby(grp_keys_cur)["index"].count()

        # lv: people present at start who leave at boundary due to END_OF_PROCESS (type_left == 1)
        # departures: use df_curr for ep_leaving (set when leaving event processed)
        _leave_now_mask = df_curr["ep_leaving"] == _boundary
        _leave_end_of_process_indices = df_presence_begin.index.intersection(
            df_curr[_leave_now_mask & (df_curr["type_left"] == 1)].index
        )
        lv = (
            df_presence_begin.loc[_leave_end_of_process_indices]
            .groupby(grp_keys_cur)["index"]
            .count()
        )

        # inc: incarcerated during the period (type_left == 3 means INCARCERATION).
        # Note: this is the present-at-start-filtered count (intersected with
        # df_presence_begin), so it excludes individuals who arrived AND were
        # incarcerated within the same review period, plus the entire first window
        # (loop starts at p=1). The unfiltered event-firing total lives in
        # simulator.df_episode_stats.cum_n_incarcerations.
        _incarcerated_indices = df_presence_begin.index.intersection(
            df_curr[_leave_now_mask & (df_curr["type_left"] == 3)].index
        )
        inc = (
            df_presence_begin.loc[_incarcerated_indices]
            .groupby(grp_keys_cur)["index"]
            .count()
        )

        # nr: number of returns during the period (ep_return == _boundary)
        if "ep_return" in df_curr.columns:
            _return_now_mask = df_curr["ep_return"] == _boundary
            nr = df_curr.loc[_return_now_mask].groupby(grp_keys_cur)["index"].count()
        else:
            nr = pd.Series(dtype=float)

        # tau: treatment in period
        tau = df_presence_begin.groupby(grp_keys_cur)["bool_treat"].sum()

        df_traj = (
            pd.DataFrame(
                {
                    "population_start": x0,
                    "population_end": x,
                    "total_arrivals": lmd,
                    "total_departures": lv,
                    "total_offenses": yin,
                    "total_offenses_out": yout,
                    "total_offenders": observed_yin,
                    "total_offenders_out": observed_yout,
                    "total_enrollment": tau,
                    "total_incarcerated": inc,
                    "total_returns": nr,
                }
            )
            .fillna(0)
            .rename_axis(index=grp_keys_cur)
            .assign(
                enrollment_rate=lambda df: df["total_enrollment"]
                / (df["population_start"] + 1e-3),
                population_check=lambda df: df.apply(
                    lambda row: row["population_start"]
                    + row["total_arrivals"]
                    + row["total_returns"]
                    - row["total_departures"]
                    - row["total_incarcerated"]
                    + row["total_offenders"]
                    - row["total_offenders_out"],
                    axis=1,
                ),
                error=lambda df: df["population_check"] - df["population_end"],
            )
        )
        error = df_traj["error"].abs().max()
        error_accumulated += error

        # Store y flow information (state_prev -> state_curr for offenders)
        y_flow = (
            df_merged[df_merged["offenses"] > 0]
            .groupby(["state_prev", "state_curr"])["offenses"]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "state_prev": "state_lst",
                    "state_curr": "state",
                    "offenses": "count",
                }
            )
        )

        rr = {
            "y": (
                y_flow.set_index(["state_lst", "state"])["count"].to_dict()
                if len(y_flow) > 0
                else {}
            ),
            **df_traj.to_dict(),
        }
        results.append(rr)
        results_df.append(df_traj)

        # compute exact flows from previous state(s) to current state(s)
        try:
            # x_flow: individuals present at start who did NOT have offenses
            x_flow_df = (
                df_merged[df_merged["offenses"] == 0]
                .groupby(["state_prev", "state_curr"])["index"]
                .count()
                .rename("count")
                .reset_index()
                .rename(columns={"state_prev": "state_lst", "state_curr": "state"})
            )
            x_flow_df["source"] = "x"
            _xtot = x_flow_df.groupby(["state_lst"])["count"].transform("sum")
            x_flow_df["ratio"] = np.where(_xtot > 0, x_flow_df["count"] / _xtot, 0.0)

            # y_flow: individuals present at start who had offenses
            y_flow_df = (
                df_merged[df_merged["offenses"] > 0]
                .groupby(["state_prev", "state_curr"])["offenses"]
                .sum()
                .rename("count")
                .reset_index()
                .rename(columns={"state_prev": "state_lst", "state_curr": "state"})
            )
            y_flow_df["source"] = "y"
            _ytot = y_flow_df.groupby(["state_lst"])["count"].transform("sum")
            y_flow_df["ratio"] = np.where(_ytot > 0, y_flow_df["count"] / _ytot, 0.0)

            # arrivals as separate source with previous-state marked 'NEW'
            arrival_flow_df = (
                df_curr.loc[_arrive_now_mask]
                .groupby(grp_keys_cur)["index"]
                .count()
                .rename("count")
                .reset_index()
            )
            arrival_flow_df["state_lst"] = "NEW"
            arrival_flow_df["source"] = "new"
            _ntot_new = arrival_flow_df.groupby(["state_lst"])["count"].transform("sum")
            arrival_flow_df["ratio"] = np.where(
                _ntot_new > 0, arrival_flow_df["count"] / _ntot_new, 0.0
            )

            flow_df = pd.concat(
                [x_flow_df, y_flow_df, arrival_flow_df], ignore_index=True
            ).assign(
                state_key=lambda df: df["state"].apply(
                    self.state_defs.state_key_range.get
                ),
                state_lst_key=lambda df: df["state_lst"].apply(
                    self.state_defs.state_key_range.get
                ),
            )
        except Exception:
            flow_df = pd.DataFrame(
                columns=["state_lst", "state", "count", "source", "ratio", "ratio_all"]
            )

        # compute ratios: within-source and across all sources per origin
        if len(flow_df) > 0:
            _src_tot = flow_df.groupby(["state_lst", "source"], group_keys=False)[
                "count"
            ].transform("sum")
            flow_df["ratio"] = np.where(_src_tot > 0, flow_df["count"] / _src_tot, 0.0)
            _orig_tot = flow_df.groupby(["state_lst"], group_keys=False)[
                "count"
            ].transform("sum")
            flow_df["ratio_all"] = np.where(
                _orig_tot > 0, flow_df["count"] / _orig_tot, 0.0
            )

        # sort by source, previous-state columns, then current-state columns
        sort_cols = ["source", "state_lst", "state"]
        sort_cols = [c for c in sort_cols if c in flow_df.columns]
        if len(sort_cols) > 0:
            flow_df = flow_df.sort_values(by=sort_cols).reset_index(drop=True)

        flow_df["snap"] = p
        results_flow.append(flow_df)

        # compute retention per origin based on starts (including arrivals) and leaves
        _start_mask_union = _present_start_mask | (
            df_prev["ep_arrival"] == _boundary_prev
        )
        _starts = df_prev.loc[_start_mask_union].groupby(grp_keys_cur)["index"].count()
        # _leaves: people who started and left at boundary (use df_curr for leave mask)
        _start_indices = df_prev[_start_mask_union].index
        _leave_start_indices = _start_indices.intersection(
            df_curr[_leave_now_mask].index
        )
        _leaves = (
            df_prev.loc[_leave_start_indices].groupby(grp_keys_cur)["index"].count()
        )
        _ret_df = (
            pd.DataFrame({"total": _starts, "left": _leaves})
            .fillna(0)
            .assign(
                stay=lambda d: d["total"] - d["left"],
                retention_ratio=lambda d: np.where(
                    d["total"] > 0, (d["total"] - d["left"]) / d["total"], 0.0
                ),
            )
            .reset_index()
        )
        _ret_df["snap"] = p
        # rename index column to match expected structure
        if grp_keys_cur:
            _ret_df = _ret_df.rename(columns={grp_keys_cur[0]: "state_lst"})
        retention_df = _ret_df.assign(
            state_lst_key=lambda df: df["state_lst"].apply(
                self.state_defs.state_key_range.get
            ),
        )

        results_retention.append(retention_df)

    # compute moving-average counts first, then ratios over last N episodes
    if len(results_flow) > 0:
        _all_flow = pd.concat(results_flow, ignore_index=True)
        needed_cols = ["state_lst", "state", "source", "snap", "count"]
        if all(c in _all_flow.columns for c in needed_cols):
            _all_flow = _all_flow.sort_values(
                by=["state_lst", "state", "source", "snap"]
            )
            # rolling average counts per origin+source+destination
            _all_flow["count_ma"] = (
                _all_flow.groupby(["state_lst", "state", "source"], group_keys=False)[
                    "count"
                ]
                .rolling(window=window_for_ratios, min_periods=10)
                .mean()
                .reset_index(level=[0, 1, 2], drop=True)
            )
            # per-source normalization across destinations
            _all_flow["origin_source_total_ma"] = _all_flow.groupby(
                ["state_lst", "source", "snap"], group_keys=False
            )["count_ma"].transform("sum")
            _all_flow["ratio_ma"] = np.where(
                _all_flow["origin_source_total_ma"] > 0,
                _all_flow["count_ma"] / _all_flow["origin_source_total_ma"],
                0.0,
            )
            # across all sources normalization
            _all_flow["origin_total_ma"] = _all_flow.groupby(
                ["state_lst", "snap"], group_keys=False
            )["count_ma"].transform("sum")
            _all_flow["ratio_all_ma"] = np.where(
                _all_flow["origin_total_ma"] > 0,
                _all_flow["count_ma"] / _all_flow["origin_total_ma"],
                0.0,
            )
            results_flow = [
                _all_flow[_all_flow["snap"] == snap].reset_index(drop=True)
                for snap in sorted(_all_flow["snap"].unique())
            ]

    # compute retention
    try:
        if len(results_retention) > 0:
            _all_ret = pd.concat(results_retention, ignore_index=True)
            if all(
                c in _all_ret.columns for c in ["state_lst", "snap", "stay", "left"]
            ):
                _all_ret = _all_ret.sort_values(by=["state_lst", "snap"])
                # rolling average counts then compute ratio
                _all_ret["stay_ma"] = (
                    _all_ret.groupby(["state_lst"], group_keys=False)["stay"]
                    .rolling(window=window_for_ratios, min_periods=10)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                _all_ret["left_ma"] = (
                    _all_ret.groupby(["state_lst"], group_keys=False)["left"]
                    .rolling(window=window_for_ratios, min_periods=10)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                _all_ret["total_ma"] = _all_ret["stay_ma"] + _all_ret["left_ma"]
                _all_ret["retention_ratio_ma"] = np.where(
                    _all_ret["total_ma"] > 0,
                    _all_ret["stay_ma"] / _all_ret["total_ma"],
                    0.0,
                )
                results_retention = [
                    _all_ret[_all_ret["snap"] == snap].reset_index(drop=True)
                    for snap in sorted(_all_ret["snap"].unique())
                ]
    except Exception:
        pass

    # sum of last 20 results
    from functools import reduce

    if len(results_df) >= window_for_ratios:
        mean_df = reduce(
            lambda x, y: x.add(y, fill_value=0), results_df[-window_for_ratios:]
        ).astype(float)
        mean_df = mean_df / window_for_ratios
    elif len(results_df) > 0:
        mean_df = reduce(lambda x, y: x.add(y, fill_value=0), results_df).astype(float)
        mean_df = mean_df / len(results_df)
    else:
        mean_df = pd.DataFrame()

    # build transition matrices Px and Py (state_lst_key -> state_key)
    if len(results_flow) > 0:
        _all_flow = pd.concat(results_flow, ignore_index=True)
        cols_needed = {
            "state_key",
            "state_lst_key",
            "source",
            "snap",
            "ratio_ma",
        }
        if cols_needed.issubset(_all_flow.columns):
            num_states = max(self.state_defs.state_key_range.values()) + 1
            Px = np.zeros((num_states, num_states), dtype=float)
            Py = np.zeros((num_states, num_states), dtype=float)
            last_snap = int(sorted(_all_flow["snap"].unique())[-1])
            flow_last = _all_flow[_all_flow["snap"] == last_snap]

            def _fill_matrix(df_src, M):
                dfw = df_src.copy().fillna(0.0)
                dfw = dfw.dropna(subset=["state_key", "state_lst_key"])
                if len(dfw) == 0:
                    return
                grouped = dfw.groupby(["state_lst_key", "state_key"], as_index=False)[
                    "ratio_ma"
                ].mean()
                row_sum = grouped.groupby("state_lst_key")["ratio_ma"].sum().to_dict()
                for _, r in grouped.iterrows():
                    i = int(r["state_lst_key"])
                    j = int(r["state_key"])
                    s = float(row_sum.get(i, 0.0))
                    p = (float(r["ratio_ma"]) / s) if s > 0 else 0.0
                    if 0 <= i < num_states and 0 <= j < num_states:
                        M[i, j] = p

            _fill_matrix(flow_last[flow_last["source"] == "x"], Px)
            _fill_matrix(flow_last[flow_last["source"] == "y"], Py)

            self.Px = Px
            self.Py = Py
            self.Px_vec = Px.flatten()
            self.Py_vec = Py.flatten()

    # save x, x0, lmd, lv, yout, yin, tau, inc, nr from mean_df
    try:
        _md = mean_df.reset_index()
        # use "state" column for mapping (current state at end of period)
        if "state" in _md.columns:
            _md["state_key"] = _md["state"].map(self.state_defs.state_key_range)
        num_states = max(self.state_defs.state_key_range.values()) + 1

        def _to_vec(column_name: str) -> np.ndarray:
            vec = np.zeros(num_states, dtype=float)
            if column_name in _md.columns and "state_key" in _md.columns:
                for _, r in _md.dropna(subset=["state_key"]).iterrows():
                    k = int(r["state_key"]) if not pd.isna(r["state_key"]) else -1
                    if 0 <= k < num_states:
                        vec[k] = (
                            float(r[column_name])
                            if not pd.isna(r[column_name])
                            else 0.0
                        )
            return vec

        self.x0_vec = _to_vec("x0")
        self.x_vec = _to_vec("x")
        self.lmd_vec = _to_vec("lmd")
        self.lv_vec = _to_vec("lv")
        self.yout_vec = _to_vec("yout")
        self.yin_vec = _to_vec("yin")
        self.tau_vec = _to_vec("tau")
        self.inc_vec = _to_vec("inc")
        self.nr_vec = _to_vec("nr")
    except Exception:
        pass

    try:
        # retention vector from latest retention snapshot
        if len(results_retention) > 0:
            last_ret = results_retention[-1]
            num_states = max(self.state_defs.state_key_range.values()) + 1
            self.retention_vec = np.zeros(num_states, dtype=float)
            col = (
                "retention_ratio_ma"
                if "retention_ratio_ma" in last_ret.columns
                else "retention_ratio"
            )
            if "state_lst_key" in last_ret.columns and col in last_ret.columns:
                for _, r in last_ret.dropna(subset=["state_lst_key"]).iterrows():
                    k = (
                        int(r["state_lst_key"])
                        if not pd.isna(r["state_lst_key"])
                        else -1
                    )
                    if 0 <= k < num_states:
                        self.retention_vec[k] = max(
                            float(r[col]) if not pd.isna(r[col]) else 0.0, 0.1
                        )
    except Exception:
        pass

    return (
        mean_df.reset_index().assign(
            state_key=lambda df: (
                df["state"].map(self.state_defs.state_key_range)
                if "state" in df.columns
                else None
            ),
        ),
        results,
        results_df,
        results_flow,
        results_retention,
    )


def evaluation_metrics(results_df):
    """
    Compute evaluation metrics from trajectory results.

    Each result in results_df is a dataframe with columns:
        population_start, population_end, total_arrivals, total_departures,
        total_offenses, total_offenses_out, total_offenders, total_offenders_out,
        total_enrollment, total_incarcerated, total_returns, enrollment_rate, population_check, error

        - total_offenses/total_offenses_out: offense counts (sum of j_curr - j_prev, counts repeat offenses)
        - total_offenders/total_offenders_out: distinct individuals who offended (at least once)

    Example:
        state_lst     population_start  ...  total_offenses  total_offenders  total_enrollment  ...
    (25.0, 4.0, 1.0)              1.0  ...             1.0              1.0               1.0  ...
    ...

    Returns dict with vectors (one value per episode):
        - total_population: total individuals present (sum of x0)
        - total_offenses: total recidivism events (sum of yin, counts repeat offenses)
        - total_offenders: number of distinct individuals who offended (sum of observed_yin)
        - total_offenses_treated: offenses by individuals with has_been_treated==1
        - total_enrollment: total treated individuals (sum of tau)
        - total_arrivals: new arrivals (sum of lmd)
        - total_departures: left system (sum of lv)
        - total_incarcerated: incarcerated during period (sum of inc)
        - total_returns: number of returns during period (sum of nr)
        - offense_rate: fraction of population who offended (total_offenders / population)
        - offense_per_capita: total offenses per capita (total_offenses / population)
        - offense_rate_treated: offenses by treated / population
        - enrollment_rate: enrolled / population
        - incarceration_rate: incarcerated / population
        - return_rate: returns / population
    """
    metrics = {
        "total_population": [],
        "total_offenses": [],
        "total_offenders": [],
        "total_offenses_treated": [],
        "total_enrollment": [],
        "total_arrivals": [],
        "total_departures": [],
        "total_incarcerated": [],
        "total_returns": [],
        "offense_rate": [],
        "offense_per_capita": [],
        "offense_rate_treated": [],
        "enrollment_rate": [],
        "incarceration_rate": [],
        "return_rate": [],
    }

    for df in results_df:
        pop = df["population_start"].sum()
        offenses = df["total_offenses"].sum()
        offenders = (
            df["total_offenders"].sum() if "total_offenders" in df.columns else 0.0
        )

        # Offenses by treated individuals (filter by state where has_been_treated == 1)
        # State tuple structure: (offenses, age_dist, has_been_treated, stage)
        # has_been_treated is at index 2
        try:
            offenses_treated = df[df.index.map(lambda s: s[2] == 1)][
                "total_offenses"
            ].sum()
        except (IndexError, TypeError, KeyError):
            # If state structure is different or has_been_treated not in state
            offenses_treated = 0.0

        enrolled = df["total_enrollment"].sum()
        arrivals = df["total_arrivals"].sum()
        departures = df["total_departures"].sum()
        incarcerated = (
            df["total_incarcerated"].sum()
            if "total_incarcerated" in df.columns
            else 0.0
        )
        returns = df["total_returns"].sum() if "total_returns" in df.columns else 0.0

        metrics["total_population"].append(pop)
        metrics["total_offenses"].append(offenses)
        metrics["total_offenders"].append(offenders)
        metrics["total_offenses_treated"].append(offenses_treated)
        metrics["total_enrollment"].append(enrolled)
        metrics["total_arrivals"].append(arrivals)
        metrics["total_departures"].append(departures)
        metrics["total_incarcerated"].append(incarcerated)
        metrics["total_returns"].append(returns)
        metrics["offense_rate"].append(offenders / pop if pop > 0 else 0.0)
        metrics["offense_per_capita"].append(offenses / pop if pop > 0 else 0.0)
        metrics["offense_rate_treated"].append(
            offenses_treated / pop if pop > 0 else 0.0
        )
        metrics["enrollment_rate"].append(enrolled / pop if pop > 0 else 0.0)
        metrics["incarceration_rate"].append(incarcerated / pop if pop > 0 else 0.0)
        metrics["return_rate"].append(returns / pop if pop > 0 else 0.0)

    # Convert to numpy arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])

    # Save per-state data per episode (flattened for HDF5 storage)
    # Indices are shared between treatment and population data
    # Collect per-episode arrays
    index_list = []
    tau_list = []
    tau_rel_list = []
    x0_list = []  # population size per state
    yin_list = []  # offenses per state
    inc_list = []  # incarcerated per state
    lv_list = []  # exits/departures per state
    lengths = []

    for df in results_df:
        # Index is state tuple - convert to 2D array (n_states x n_dims)
        # Handle mixed types (numbers and strings) by converting strings to codes
        idx_raw = [list(i) if hasattr(i, "__iter__") else [i] for i in df.index]

        # Convert each element to numeric (encode strings as numbers)
        idx_numeric = []
        for state_tuple in idx_raw:
            numeric_tuple = []
            for val in state_tuple:
                if isinstance(val, str):
                    # Encode string values: 'p' -> 0, 'f' -> 1 (for stage)
                    # This is a simple encoding scheme
                    if val == "p":
                        numeric_tuple.append(0)
                    elif val == "f":
                        numeric_tuple.append(1)
                    else:
                        # For any other string, use hash or ordinal
                        numeric_tuple.append(ord(val[0]) if len(val) > 0 else 0)
                else:
                    numeric_tuple.append(float(val))
            idx_numeric.append(numeric_tuple)

        idx_arr = np.array(idx_numeric)
        tau = (
            df["total_enrollment"].values
            if "total_enrollment" in df.columns
            else np.zeros(len(df))
        )
        tau_rel = (
            df["enrollment_rate"].values
            if "enrollment_rate" in df.columns
            else np.zeros(len(df))
        )
        x0 = (
            df["population_start"].values
            if "population_start" in df.columns
            else np.zeros(len(df))
        )
        yin = (
            df["total_offenses"].values
            if "total_offenses" in df.columns
            else np.zeros(len(df))
        )
        inc = (
            df["total_incarcerated"].values
            if "total_incarcerated" in df.columns
            else np.zeros(len(df))
        )
        lv = (
            df["total_departures"].values
            if "total_departures" in df.columns
            else np.zeros(len(df))
        )
        index_list.append(idx_arr)
        tau_list.append(tau)
        tau_rel_list.append(tau_rel)
        x0_list.append(x0)
        yin_list.append(yin)
        inc_list.append(inc)
        lv_list.append(lv)
        lengths.append(len(df))

    # Flatten into arrays for storage (indices shared across all per-state metrics)
    metrics["state_index_flat"] = (
        np.vstack(index_list) if index_list else np.array([])
    )  # shape: (total_states, n_dims)
    metrics["state_lengths"] = np.array(lengths)  # to reconstruct per-episode arrays

    # Treatment data (uses shared indices)
    metrics["treatment_tau_flat"] = (
        np.concatenate(tau_list) if tau_list else np.array([])
    )
    metrics["treatment_tau_rel_flat"] = (
        np.concatenate(tau_rel_list) if tau_rel_list else np.array([])
    )

    # Population data (uses shared indices)
    metrics["population_x0_flat"] = np.concatenate(x0_list) if x0_list else np.array([])
    metrics["population_yin_flat"] = (
        np.concatenate(yin_list) if yin_list else np.array([])
    )

    # Incarcerated and exits data (uses shared indices)
    metrics["population_inc_flat"] = (
        np.concatenate(inc_list) if inc_list else np.array([])
    )
    metrics["population_lv_flat"] = np.concatenate(lv_list) if lv_list else np.array([])

    # Keep legacy names for backward compatibility
    metrics["treatment_index_flat"] = metrics["state_index_flat"]
    metrics["treatment_lengths"] = metrics["state_lengths"]

    return metrics


def recover_per_state_as_df(metrics, rep=None, columns=None):
    """
    Recover per-state data as a single DataFrame with shared indices.

    Available columns: total_enrollment, enrollment_rate, population_start, total_offenses,
                       total_incarcerated, total_departures

    Args:
        metrics: dict from evaluation_metrics() or all_metrics[policy_name]
        rep: repetition index (required if metrics has multiple reps, i.e. from all_metrics)
        columns: list of columns to include (default: all available)
                 e.g., ['total_enrollment', 'enrollment_rate'] for treatment only,
                       ['population_start', 'total_offenses'] for population only,
                       None for all columns

    Returns:
        pd.DataFrame with requested columns and MultiIndex (state, p)
        If multiple reps and rep=None, includes 'rep' in the index: (rep, state, p)
    """
    import pandas as pd

    # Use shared index (fall back to legacy name if not present)
    index_flat = metrics.get("state_index_flat", metrics.get("treatment_index_flat"))
    lengths = metrics.get("state_lengths", metrics.get("treatment_lengths"))

    # Available data columns
    available_data = {
        "total_enrollment": metrics.get("treatment_tau_flat"),
        "enrollment_rate": metrics.get("treatment_tau_rel_flat"),
        "population_start": metrics.get("population_x0_flat"),
        "total_offenses": metrics.get("population_yin_flat"),
        "total_incarcerated": metrics.get("population_inc_flat"),
        "total_departures": metrics.get("population_lv_flat"),
    }

    # Determine which columns to include
    if columns is None:
        # Include all available columns
        columns = [k for k, v in available_data.items() if v is not None]
    else:
        # Validate requested columns
        for col in columns:
            if col not in available_data:
                raise ValueError(
                    f"Unknown column '{col}'. Available: {list(available_data.keys())}"
                )
            if available_data[col] is None:
                raise ValueError(
                    f"Column '{col}' not found in metrics. Ensure data was saved with updated evaluation_metrics()."
                )

    # Check if this is multi-rep data (list of arrays) or single-rep (single array)
    is_multi_rep = isinstance(lengths, list)

    if is_multi_rep:
        if rep is not None:
            # Extract single repetition
            index_flat = index_flat[rep]
            lengths = lengths[rep]
            col_data = {col: available_data[col][rep] for col in columns}
        else:
            # Combine all reps with rep index
            dfs = []
            for r in range(len(lengths)):
                idx = index_flat[r]
                lens = lengths[r]

                states = [tuple(row) for row in idx]
                episodes = np.repeat(np.arange(len(lens)), lens)
                reps = np.full(len(states), r)

                # Validate data lengths match index length
                index_len = len(states)
                data = {}
                for col in columns:
                    col_data = available_data[col][r]
                    if len(col_data) != index_len:
                        raise ValueError(
                            f"Length mismatch for column '{col}' in rep {r}: "
                            f"data has {len(col_data)} entries but index has {index_len}. "
                            f"This may indicate the data was saved with a different state structure."
                        )
                    data[col] = col_data

                multi_idx = pd.MultiIndex.from_arrays(
                    [reps, states, episodes], names=["rep", "state", "p"]
                )
                df = pd.DataFrame(data, index=multi_idx)
                dfs.append(df)
            return pd.concat(dfs)
    else:
        col_data = {col: available_data[col] for col in columns}

    # Single rep case (or extracted single rep from multi-rep)
    states = [tuple(row) for row in index_flat]
    episodes = np.repeat(np.arange(len(lengths)), lengths)

    multi_idx = pd.MultiIndex.from_arrays([states, episodes], names=["state", "p"])
    df = pd.DataFrame(col_data, index=multi_idx)

    return df


# Backward compatibility aliases
def recover_treatment_decision_as_df(metrics, rep=None):
    """
    Recover treatment decisions as a DataFrame with MultiIndex.

    DEPRECATED: Use recover_per_state_as_df(metrics, rep, columns=['tau', 'tau_rel']) instead.
    """
    return recover_per_state_as_df(metrics, rep=rep, columns=["tau", "tau_rel"])


def recover_population_as_df(metrics, rep=None):
    """
    Recover population data as a DataFrame with MultiIndex.

    DEPRECATED: Use recover_per_state_as_df(metrics, rep, columns=['x0', 'yin']) instead.
    """
    return recover_per_state_as_df(metrics, rep=rep, columns=["x0", "yin"])


def compute_equilibrium_stats(all_metrics, metric_key="tau", last_n=10):
    """
    Compute equilibrium average of a metric over the last N episodes across all reps.
    Aggregates by (offenses, age_dist), summing across has_been_treated and stage.

    Args:
        all_metrics: dict from read_metrics_from_h5 containing per-state fields
        metric_key: one of "tau", "tau_rel", "x0", "yin" (default "tau")
        last_n: number of episodes from the end to average over (default 10)

    Returns:
        (df_mean, df_std): two DataFrames with:
            - rows = age_dist (second dim of state)
            - columns = offenses (first dim of state)
            - df_mean: raw mean values (sum across all states with same offenses, age_dist)
            - df_std: raw std values

    Note: State structure is (offenses, age_dist, has_been_treated, stage_encoded)
    """
    # Recover per-state data as DataFrame using shared function
    df = recover_per_state_as_df(all_metrics, rep=None, columns=[metric_key])

    # df has MultiIndex (rep, state, p) where state is a tuple
    # Filter to last_n episodes per rep
    n_reps = df.index.get_level_values("rep").nunique()

    # Collect per-state data across all reps
    # state_data[state_key] = [val1, val2, ...] where each entry is one rep's SUM for that episode window
    state_data = defaultdict(list)

    for rep in range(n_reps):
        rep_df = df.xs(rep, level="rep")
        n_episodes = rep_df.index.get_level_values("p").max() + 1

        if n_episodes < last_n:
            ep_start = 0
        else:
            ep_start = n_episodes - last_n

        # Filter to last_n episodes
        rep_df = rep_df[rep_df.index.get_level_values("p") >= ep_start]

        # First, sum across all states per episode to get total per (offenses, age_dist)
        # Then take mean across episodes
        episode_sums = defaultdict(lambda: defaultdict(float))
        for state, group in rep_df.groupby(level="state"):
            # State structure: (offenses, age_dist, has_been_treated, stage_encoded)
            # Use (offenses, age_dist) as key, summing across has_been_treated and stage
            state_key = (state[0], state[1])
            for p, val in group[metric_key].items():
                episode_idx = p[1]  # p is (state, episode_idx)
                episode_sums[state_key][episode_idx] += val

        # Now compute mean across episodes for this rep
        for state_key, ep_vals in episode_sums.items():
            mean_val = np.mean(list(ep_vals.values()))
            state_data[state_key].append(mean_val)

    # Build result: rows = age_dist, columns = offenses
    # state_key = (offenses, age_dist)
    offenses_set = set()
    age_dist_set = set()
    for state_key in state_data:
        offenses_set.add(state_key[0])
        age_dist_set.add(state_key[1])

    offenses_sorted = sorted(offenses_set)
    age_dist_sorted = sorted(age_dist_set)

    # Create mean and std matrices
    mean_data = {off: {} for off in offenses_sorted}
    std_data = {off: {} for off in offenses_sorted}

    for state_key, values in state_data.items():
        offenses, age_dist = state_key
        arr = np.array(values)
        mean_data[offenses][age_dist] = np.mean(arr)
        std_data[offenses][age_dist] = np.std(arr)

    # Build DataFrames: rows = age_dist, columns = offenses
    df_mean = pd.DataFrame(
        mean_data, index=age_dist_sorted, columns=offenses_sorted
    ).fillna(0)
    df_std = pd.DataFrame(
        std_data, index=age_dist_sorted, columns=offenses_sorted
    ).fillna(0)

    df_mean.index.name = "age_dist"
    df_mean.columns.name = "offenses"
    df_std.index.name = "age_dist"
    df_std.columns.name = "offenses"

    return df_mean, df_std


# Backward compatibility alias
def compute_equilibrium_treatment_stats(all_metrics, metric_key="tau", last_n=10):
    """
    DEPRECATED: Use compute_equilibrium_stats() instead.
    """
    return compute_equilibrium_stats(all_metrics, metric_key=metric_key, last_n=last_n)

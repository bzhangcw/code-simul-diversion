# Step 2: Run the policy simulation


## 1. Start in the simulation directory

Change the working directory to simulation, and create a result directory to store your simulation statistics.

```bash
cd ./simulation
mkdir -p results
```


## 2. The python script to run simulation: `run_policy.py`

The Python script for running the simulation is the following

```bash
python -u run_policy.py -h
```

The general form is:

```text
python -u run_policy.py [OPTIONS] POLICY_NAME REP OUTPUT_DIR
```

The three positional arguments are:

| Argument | Meaning |
| --- | --- |
| `POLICY_NAME` | Treatment-assignment policy. The help output lists every currently registered policy. |
| `REP` | Repetition number. The same integer is used as the random seed and output-directory name. |
| `OUTPUT_DIR` | Directory beneath which policy and repetition folders are created. |

One Python invocation always runs exactly one repetition. 
The optional arguments are grouped in the help output. 

| Group | Important options |
| --- | --- |
| Simulation horizon | `--T_max`, `--p_length`, `--p_freeze`, `--p_freeze_policy` |
| Population flow | `--beta_arrival`, `--beta_initial`, `--rel_off_probation` |
| Treatment | `--treatment_capacity`, `--treatment_effect`, `--treatment_dosage`, `--treatment_timing` |
| Limits | `--max_returns`, `--max_offenses` |
| Model switches | `--bool_return_can_be_treated`, `--bool_has_incarceration`, `--bool_keep_idiosyncratic_effect`, `--bool_mean_use_accumulated_offenses` |
| Scaling | `--prison_rate_scaler`, `--length_scaler` |
| Output | `--verbosity` |

You can pass `-h` to see details.
Defaults are printed beside the options. Most of the arguments can be omitted in the input and defaults will be applied.

### Example smoke-test output

For example, we can run one repetition of the simulation, without any treatment policy by the following command,
```bash
python -u run_policy.py \
  --T_max 1000 \
  --p_length 100 \
  --prison_rate_scaler 0.02 \
  --rel_off_probation 1000 \
  --treatment_capacity 100 \
  --treatment_effect 0.6 \
  --bool_return_can_be_treated 1 \
  --bool_keep_idiosyncratic_effect 1 \
  null 0 results/smoke
```

The console first confirms the effective configuration, then reports one row
per episode. The console will report log including:

- header information and configurations
- summary of each episode
- data paths

Most of the output should be self-explanatory. For example, the above command will output:

> header information and configurations

```text
Loading data from pickle cache...
============================================================================================================================================
SimulationSetup Summary
============================================================================================================================================
  Mode:              FORMAL (DBG=0)
--------------------------------------------------------------------------------------------------------------------------------------------
Simulation Parameters:
  T_max:             1,000 time units
  p_length:          100 (episode length)
  p_freeze:          2 (freeze period)
  p_freeze_policy:   10 (policy freeze period)
  off_prob:          1000 (off-probation term)
  n_episodes:        ~10
--------------------------------------------------------------------------------------------------------------------------------------------
Treatment Settings:
  capacity:          100 per episode
  effect:            Homogeneous treatment effect: input 0.6, translate to score as +log(0.6) = -0.5108256237659907
  dosage:            Homogeneous treatment dosage.
  max_returns:       25
  max_offenses:      25
  return_can_treat:  1
  has_incarceration: 1
  keep_idiosync:     1 (individual score_fixed)
  use_accum_stats:   1 (cumulative stats for community score)
  prison_scaler:     0.02 (scaler for prison rate)
  length_scaler:     1.0 (scaler for probation/off-probation length)
  qualifying:        bool_left == 0 & stage == 'p'& bool_can_be_treated == 1& has_been_treated == 0& bool_decision_made == 0
  current_enroll:    bool_left == 0 & bool_treat == 1 & stage == 'p'
--------------------------------------------------------------------------------------------------------------------------------------------
Population & Arrivals:
  beta_arrival:      5 (arrival rate)
  beta_initial:      5 (initial population size)
  communities:       {1}
  initial pop:       5 individuals
  available pool:    9374 individuals
    age  weight  prison_rate  score_fixed  score_comm
0  55.0     1.0         0.12     0.857657   -2.078157
1  45.0     1.0         0.12     0.323657   -2.078157
2  55.0     1.0         0.12     0.244657   -2.078157
3  18.0     1.0         0.12     1.279657   -2.078157
4  33.0     1.0         0.12     1.459657   -2.078157
--------------------------------------------------------------------------------------------------------------------------------------------
State Space:
  dimensions:        ['offenses', 'age_dist', 'has_been_treated', 'stage']
  n_states:          264
  state weights:     {'score_age_dist': '0.7452', 'score_offenses': '0.1089'}
  final weights:     {'score_fixed': '0.7452', 'score_comm': '0.7452', 'score_state': '1.0000', 'score_treatment': '1.0000'}
--------------------------------------------------------------------------------------------------------------------------------------------
Scoring Equation:
  score = score_fixed     × 0.7452
        + score_comm      × 0.7452
        + score_state     × 1.0000
        + score_treatment × 1.0000

    where score_state = score_offenses × 0.1089
                      + score_age_dist × 0.7452

          score_comm  = mean_re_time × (-0.003)
                      + percent_re   × 0.057

  S(t | score) = S₀(t) ^ exp(score)
```

> summary of an episode

```
--------------------------------------------------------------------------------------------------------------------------------------------
   episode  n_total  n_present_at_start  n_present_at_end  ep_n_offenses  ep_offense_time  ep_percent_re  ep_mean_re_time  cum_n_offenses  cum_offense_time  cum_percent_re  cum_mean_re_time  ep_n_incarcerations  cum_n_incarcerations  score_comm
2        3       64                  64                64            5.0       363.155406       0.109375        72.631081             9.0        722.084674        0.109375         80.231630                  0.0                   0.0   -0.234461
3        4       85                  85                85            5.0       866.770826       0.129412       173.354165            14.0       1588.855500        0.129412        113.489679                  0.0                   0.0   -0.333093
4        5      104                 104               104           16.0      3176.993312       0.230769       198.562082            30.0       4765.848812        0.230769        158.861627                  0.0                   0.0   -0.463431
5        6      125                 125               125           15.0      3874.901978       0.280000       258.326799            45.0       8640.750789        0.280000        192.016684                  0.0                   0.0   -0.560090
6        7      144                 144               144           20.0      4178.399315       0.340278       208.919966            65.0      12819.150105        0.340278        197.217694                  0.0                   0.0   -0.572257
current global measures: percent_re        0.340278
mean_re_time    197.217694
dtype: float64
[update_community_score] elapsed: 0.0026s
[update_state] elapsed: 0.0864s
[treatment_selection] elapsed: 0.0000s
episode 8/10 (t=800/1000)
```

> output

A successful run here will save `results/smoke/null/0/metrics.h5` plus the
Excel summaries and log files.


## (Optional) 3. Using the template zsh script for batch run

The Python script supports exactly one repetition of the simulation with the corresponding parameters and selected repetition number.
For convenience, it will be helpful to use it as a tool and call it by a batch script, so we can test **different parameters** 
and **automatically organize the simulation data by repetitions**. Such a template script is provided under `./sbin`.

!!! note
    It will be helpful to have a basic understanding of bash scripts. In the
    zsh file, you can edit the `policies`, `scale_factors`, `term_lengths`,
    `beta`, and `cap` assignments before generating commands when a different
    experiment is intended.

### Introduction to the template script
```zsh
zsh run_all_params-ofpl.zsh REPEAT START OUTPUT EFFECT ALLOW_RETURN IDIOSYNCRASY
```

| Argument | Meaning |
| --- | --- |
| `REPEAT` | Number of repetitions to generate. |
| `START` | First repetition number and random seed. |
| `OUTPUT` | Root result directory. Avoid spaces in this path because the script does not quote every generated path. |
| `EFFECT` | Treatment effect accepted by `run_policy.py`, such as `0.71` or `type-1`. A numeric value must be positive. |
| `ALLOW_RETURN` | `1` allows returning individuals to receive treatment; `0` does not. |
| `IDIOSYNCRASY` | `1` retains individual fixed-risk scores; `0` replaces them with the population median. |

For example, you will generate 5 repeations for each configuration under `results/` using the following command:

```bash
OUTPUT=results/result-ofpl-scl-0.6
zsh sbin/run_all_params-ofpl.zsh 5 0 "$OUTPUT" 0.6 1 1
```

After running it, it outputs
```bash
usage: run_all_params-ofpl.zsh <repetitions> <start> <output> <effect> <allowrtn> <idio>
 |- repetitions: number of repetitions
 |- start: first repetition number and random seed
 |- output: output directory
 |- effect: treatment effect
 |- allowrtn: whether to allow returning individuals to be treated
 |- idio: whether to allow idiosyncrasies
Generating commands for 5 repetitions # 0 => 4
	 with effect=0.6
	 and allowrtn=1
	 and idio=1 individuals
Commands generated in cmd.sh.
see:  results/result-ofpl-scl-0.6/cmd.sh
parallel run with:

cat results/result-ofpl-scl-0.6/cmd.sh | xargs -I {} -P 30 bash -c "{}" &
```

!!! note 
    The zsh script will not direct start the Python processes, instead, it create the commands in a cmd file
    ```
    results/$OUTPUT/cmd.sh
    ```
    in which, each line should call `python -u run_policy.py`, contain the intended
    parameters, end with one repetition number, and write to the intended
    `tl-<term>-sf-<scale>` directory.
      Check the count and inspect several lines:
      ```bash
      wc -l "$OUTPUT/cmd.sh"
      sed -n '1,5p' "$OUTPUT/cmd.sh"
      ```

### Starting the jobs
You can directly run it line-by-line via
  ```bash
  zsh "$OUTPUT/cmd.sh"
  ```
or, instead, use tools like `xargs` or `parallel` to run it in a concurrent mode.
  For examples, run with 30 jobs in parallel using `xargs`:
  ```bash
  cat results/result-ofpl-scl-0.6/cmd.sh | xargs -I {} -P 30 bash -c "{}" &
  ```

### Monitoring the job status
In this specific example, you can watch one job log:

  ```bash
  tail -f "$OUTPUT/tl-1000-sf-0.02/logs/null_0_tl-1000-sf-0.02.log"
  ```
  Scan all logs for failures:

  ```bash
  rg -n "Traceback|Exception|Error" "$OUTPUT" -g '*.log'
  ```


### Understand the result layout

Each completed repetition automatically produces `metrics.h5` together with
Excel summaries and a log. `metrics.h5` is the simulation output consumed in
Step 3; no separate conversion command is needed.

A successful run (as we used above) will generate a directory with the files as follows:

```text
results/
└── result-ofpl-scl-0.6/
    ├── cmd.sh
    └── tl-1000-sf-0.02/
        ├── logs/
        │   └── null_0_tl-1000-sf-0.02.log
        ├── null/
        │   └── 0/
        │       ├── metrics.h5
        │       ├── dfi.xlsx
        │       ├── df_episode_stats.xlsx
        │       ├── df_result_last.xlsx
        │       └── log.txt
        ├── high-risk/
        ├── low-risk/
        └── age-first/
```

The files have different purposes:

| File | Purpose |
| --- | --- |
| `metrics.h5` | Per-episode metrics consumed by the sensitivity-analysis code. This is the essential analysis input. |
| `dfi.xlsx` | Final individual-level simulator state. |
| `df_episode_stats.xlsx` | Episode-level simulator statistics. |
| `df_result_last.xlsx` | State-level summary for the final episode. |
| `log.txt` | Simulator log for one repetition. |
| `logs/*.log` | Captured stdout and stderr for each generated command. |
| `cmd.sh` | Exact commands for the sweep; retain this as the experiment manifest. |

### Add repetitions without replacing metrics

As you may find 5 repetitions insufficient, you can append repetitions `5` through `9` to an existing five-repetition result:

```bash
zsh sbin/run_all_params-ofpl.zsh 5 5 "$OUTPUT" 0.6 1 1
xargs -I {} -P 4 bash -c "{}" < "$OUTPUT/cmd.sh"
```

The generator preserves existing policy/repetition result directories, but it
replaces `cmd.sh` and removes existing command-log directories. Save the old
`cmd.sh` and logs first if they are part of the experiment record.


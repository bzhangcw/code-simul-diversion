# Step 3: Collect statistics and generate plots

The statistics workflow uses three layers:

1. `simulation_summary.py` reads each repetition's `metrics.h5` file;
2. `simulation_sensanaly.py` supplies shared windowing and plotting utilities;
3. `sensanaly/run_sensanaly_prscale.py` is the runnable driver for the
   off-probation-length and prison-scale experiment.

Run the driver rather than `simulation_sensanaly.py` directly. The latter only
defines helper functions and has no command-line entry point.

## 1. Start in the simulation directory

Again, let us use the output directory defined in [Step 2](simulation_run.md)

```bash
cd ./simulation
OUTPUT=results/result-ofpl-scl-0.6
```

The module imports assume this working directory.

## 2. Verify the HDF5 results produced in Step 2

Each completed `run_policy.py` repetition from Step 2 automatically produces
one HDF5 file in a directory named exactly like:

```text
$OUTPUT/tl-<term_length>-sf-<scale_factor>/<policy>/<repetition>/metrics.h5
```

Check the files before starting the analysis:

```bash
find "$OUTPUT" -name metrics.h5 -type f | sort | sed -n '1,20p'
find "$OUTPUT" -name metrics.h5 -type f | wc -l
```

Use canonical numeric directory names such as `sf-0.7`. The parser converts
the directory component to a float and later reconstructs the name; a spelling
such as `sf-0.70` may therefore fail to reload.

The input to Step 3 is the `$OUTPUT` directory created under `results/` by Step
2, including all of its
`tl-...-sf-.../<policy>/<repetition>/metrics.h5` files.

Stop here if the HDF5 count is smaller than the expected number of simulation
commands or if any file is empty. Return to Step 2, inspect the corresponding
job logs, and rerun the failed commands. Analysis should begin only after the
raw result matrix is complete.

## 3. Review the analysis configuration

Open `sensanaly/run_sensanaly_prscale.py` and confirm its configuration before
running. For example, 

| Setting | Current value |
| --- | --- |
| Policies | `null`, `high-risk`, `low-risk`, `age-first` |
| Equilibrium window | Last `20` episodes, used as equilibrium statistics |
| Early window | `40` episodes starting at episode index `40` |
| Metrics | Offense rate, offenses per capita, incarceration rate, population, offenses, departures |

Despite the output directory name `first`, the current early window is
episodes `40` through `79`, not episodes `0` through `39`. Set
`summary_wd_start = 0` if the first episodes are intended.

The configured metrics are:

```text
offense_rate
offense_per_capita
incarceration_rate
total_population
total_offenses
total_departures
```

For ordinary metrics, the code sums each repetition over the selected window
and divides by the window length, giving a per-episode average. 

## 4. Run the analysis driver

Because the driver uses a relative package import, run it as a module:

```bash
python -m sensanaly.run_sensanaly_prscale "$OUTPUT"
```

Do not use `python sensanaly/run_sensanaly_prscale.py`; that invocation can
fail because its relative import has no package context.

The driver automatically scans all `tl-...-sf-...` directories, loads every
numeric repetition directory containing `metrics.h5`, aggregates the requested
policies, writes CSV files, and creates both absolute and null-relative plots.

## 5. Understand the generated outputs

The driver writes:

```text
$OUTPUT/sensitivity_analysis/
├── equilibrium/
│   ├── sensitivity_offense_rate.csv
│   ├── offense_rate_term_length_1000.png
│   ├── offense_rate_term_length_1000.pgf
│   ├── offense_rate_term_length_1000_rel_null.png
│   └── offense_rate_term_length_1000_rel_null.pgf
└── first/
    └── ...
```

Each CSV row is indexed by `term_length`, `scale_factor`, and `policy`, with:

| Column | Meaning |
| --- | --- |
| `mean` | Mean across repetitions after applying the selected episode window. |
| `std` | Standard deviation across repetitions. |
| `n_reps` | Number of successfully loaded repetitions. |
| `enrollment_mean` | Mean treatment enrollment per episode over the same window. |
| `enrollment_std` | Standard deviation of enrollment across repetitions. |

Inspect the CSV before trusting the plots:

```bash
sed -n '1,20p' \
  "$OUTPUT/sensitivity_analysis/equilibrium/sensitivity_offense_rate.csv"
```

Confirm that every intended parameter-policy combination appears and that
`n_reps` is constant across rows.

# (Optimals) Tips

## Keep all experiment families under `results/`

Use `results/` as the common root for raw simulation output and generated
analysis. Different treatment-effect specifications get separate experiment
directories:

```text
results/
├── result-ofpl-scl-0.6/
├── result-ofpl-scl-0.71/
├── result-ofpl-scl-0.71-hete/
├── result-ofpl-scl-0.9/
├── result-ofpl-scl-type1+/
└── result-ofpl-scl-type1++/
```

Within each family, keep the `tl-...-sf-...` raw simulation directories and the
generated `sensitivity_analysis/equilibrium` and
`sensitivity_analysis/first` directories together.

For a fully reproducible record, retain:

- the complete `tl-...-sf-...` raw result directories;
- every repetition's `metrics.h5`;
- the `cmd.sh` used for the run;
- command logs;
- the code revision, input-data version, and Python environment;
- a copy of the analysis configuration.

If the raw result tree is moved to separate storage after analysis, put a text
manifest in the compact archive recording its location, directory layout, and
checksum.

## Optional comparisons across experiment families

`compare_sensitivity.py` compares already-generated sensitivity CSVs across
multiple result folders. Its folder list and base directory are configured in
the source file, so review those values before running it. It is a downstream
comparison step and does not replace the `run_sensanaly_prscale` driver.

# Code for "The Dynamic and Endogenous Behavior of Re-Offense Risk..."

This repository contains a Python discrete-event simulator for probation,
recidivism, incarceration, and treatment-assignment policies.

## Environment setup

The regression, simulation, and analysis workflow is tested with **Python
3.10**. It uses a standard Python virtual environment and `pip`; Conda is not
required.

From the repository root, create one isolated environment and install the
simulation, notebook, and documentation dependencies:

```bash
python3.10 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
```

The prompt may show `(.venv)` while the environment is active. This name is
local to the repository and does not depend on a personally configured Python
or Jupyter environment.

Verify the interpreter and the main libraries:

```bash
python --version
python -c "import numpy, pandas, scipy, lifelines, h5py; print('Dependencies OK')"
cd ./simulation
python -u run_policy.py -h
```

The first command should report Python 3.10.x, and the second should print
`Dependencies OK`. The final command displays the simulation command-line
options and also confirms that the simulation modules can be loaded.

Keep the environment active while following all three steps below. In a new
terminal, reactivate it from the repository root with:

```bash
source ./.venv/bin/activate
```

The root `./requirements.txt` installs both component lists:
`./simulation/requirements.txt` for the research workflow and
`./docs/requirements.txt` for this documentation website.

## Workflow

Follow the workflow in order:

1. [Fit and check the regression models](simulation_regression.md).
2. [Run the policy simulation](simulation_run.md).
3. [Collect statistics and generate plots](simulation_results.md).

## Optional steps

1. [Prepare the dataset from raw data](data_preparation.md).
2. [Fit the regression from scratch](regression_without_sirakaya.md).

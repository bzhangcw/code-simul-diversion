# Step 1: Fit the regression model

The regression steps are recorded in `./simulation/py-sirakaya-fitting.ipynb`. See the comments for explaining the fitting and validation steps.

```bash
cd ./simulation
python -m jupyter lab py-sirakaya-fitting.ipynb
```

If the fit leads to a model or code change, record that change before continuing
to [Step 2: Run the simulation](simulation_run.md).

## Data


The data used for regression and simulation is kept under
```bash
./simulation/felony-1989-final.xlsx
```

Generally, you do not have to reproduce this dataset. For the steps to produce
the dataset from raw data files collected from multiple places, see
[Prepare the dataset from raw data](data_preparation.md).

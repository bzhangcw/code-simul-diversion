# Code for "The Dynamic and Endogenous Behavior of Re-Offense Risk..."

This repository contains the Python agent-based simulation and its
documentation.

## Start here: launch the documentation

From the repository root, create the shared Python 3.10 environment and
install all simulation and documentation dependencies:

```bash
python3.10 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
```

Then launch the documentation website:

```bash
./docs/serve.sh
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and follow the workflow
shown there. The server automatically rebuilds the website and reloads the
browser when a documentation source file changes. Press `Ctrl+C` in the
terminal to stop it.

For later sessions, reactivate the existing environment before working with
the simulation or notebooks:

```bash
source ./.venv/bin/activate
```

## Repository layout

- `./simulation/` contains the simulation, notebooks, batch scripts, and
  analysis code.
- `./docs/` contains the Markdown documentation and MkDocs configuration.
- `./requirements.txt` installs both the simulation and documentation
  dependencies into the shared environment.


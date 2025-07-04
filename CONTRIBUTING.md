# Contributing to `mosses`
We welcome contributions in the form of feedback via email, requests for changes/fixes via `GitHub Issues`, or direct contribution using best practices.

## Setting up your development environment
The `pyproject.toml` already contains the optional dependencies needed for development. Follow these steps to set up the environment.
```bash
# Make sure you have got Python >= 3.10
python --version
> Python 3.10.16

# Installs `mosses` in editable mode and with dev dependencies
pip install -e .[dev]
> ...
> Successfully installed cfgv-3.4.0 distlib-0.3.9 filelock-3.18.0 identify-2.6.10 mosses-0.1.0 nodeenv-1.9.1 pre-commit-4.2.0 virtualenv-20.31.1

# Setup pre-commit hooks
pre-commit install
> pre-commit installed at .git/hooks/pre-commit
```

You are ready to go! Please make sure you always work on a branch and merge through pull requests.

## Pushing packages to Pypi
Currently packages are just pushed directly using `twine`. See `Makefile`. You need the correct permissions upstream to push to the server.

# Contributing to `mosses`
We welcome contributions in the form of feedback via email, requests for changes/fixes via `GitHub Issues`, or direct contribution using best practices.

## Setting up your development environment
The `pyproject.toml` already contains a `dev` dependency group with the tools needed for development. Follow these steps to set up the environment.
```bash
# Make sure you have got Python >= 3.10
python --version
> Python 3.10.16

# Installs `mosses` in editable mode together with the `dev` dependency group (needs pip >= 25.1)
pip install -e . --group dev
> ...

# Setup pre-commit hooks
pre-commit install
> pre-commit installed at .git/hooks/pre-commit
```

You are ready to go! Please make sure you always work on a branch and merge through pull requests.

## Pushing packages to Pypi
Currently packages are just pushed directly using `twine`. See `Makefile`. You need the correct permissions upstream to push to the server.

![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# Mosses - Model Assessment Toolkit

## Description
`Mosses` is a library that provides a set of functionalities to assess molecular property prediction models, e.g., QSAR/QSPR models. The library currently includes:
- A model validation module (`predictive_validity.py`) built on top of the concept of *predictive validity* described by Scannell et al. Nat Rev Drug Discov. 2022;21(12):915-931. [doi:10.1038/s41573-022-00552-x](https://www.nature.com/articles/s41573-022-00552-x). The function `predictive_validity.evaluate_pv()` allows analysing the quality of predictions on a given data set (e.g., a prospective test set of compounds), according to a desired threshold. The analysis can be used to determine whether the model used to generate the predictions is suitable for the data of interest (e.g., the validation can be done on a new series of compounds), and if so, to configure optimal thresholds for maximising enrichment of compounds with the desired property.

## Software requirements
The library is written in Python and requires a version >= 3.10 for runtime. The dependencies required by the library are defined in `pyproject.toml` and are automatically installed when installing the library.

## How to install `mosses`
You can install the library using `pip install mosses`, or you can clone this repository then run `make build && make install`.

## Examples of usage
Jupyter notebooks with examples can be found in the folder `examples`. We recommend following those to adapt your data, configs, and code to work with the modules in `mosses`.

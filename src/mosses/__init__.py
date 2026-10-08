"""
MOSSES - MOlecular propertyS prediction aSsESSment toolkit.

A library for assessing molecular property prediction models with tools for:
- Predictive validity analysis
- Heatmap visualizations
- Multi-Parameter Optimization (MPO)
- Headless, JSON-serialisable computation API

Modules
-------
predictive_validity
    Functions for validating prediction models
heatmap
    Heatmap visualization tools
mpo
    Multi-Parameter Optimization scoring and analysis
data_api
    Data-only versions of the evaluations, without plotting

Example
-------
>>> import mosses
>>> from mosses import mpo
>>> 
>>> # Use MPO scoring
>>> result = mpo.compute_scores(df, config)
"""

from mosses import data_api, heatmap, mpo, predictive_validity

__all__ = [
    "predictive_validity",
    "heatmap",
    "mpo",
    "data_api",
]

__version__ = "0.6.0"

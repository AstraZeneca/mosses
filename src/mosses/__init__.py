"""
MOSSES - MOlecular propertyS prediction aSsESSment toolkit.

A library for assessing molecular property prediction models with tools for:
- Predictive validity analysis
- Heatmap visualizations
- Multi-Parameter Optimization (MPO)

Modules
-------
predictive_validity
    Functions for validating prediction models
heatmap
    Heatmap visualization tools
mpo
    Multi-Parameter Optimization scoring and analysis

Example
-------
>>> import mosses
>>> from mosses import mpo
>>> 
>>> # Use MPO scoring
>>> result = mpo.compute_scores(df, config)
"""

from mosses import heatmap, mpo, predictive_validity

__all__ = [
    "predictive_validity",
    "heatmap",
    "mpo",
]

__version__ = "0.2.13"

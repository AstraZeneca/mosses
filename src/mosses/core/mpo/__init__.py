"""
MPO (Multi-Parameter Optimization) submodule for the mosses library.

This module provides functionality for calculating and optimizing
Multi-Parameter Optimization scores for drug discovery applications.

Submodules:
    - scoring: Sigmoid-based scoring functions for parameter desirability
    - weights: Weight normalization and management utilities
    - statistics: Enrichment, correlation, and performance metrics
    - optimizers: Various weight optimization algorithms
    - ml_estimators: Machine learning-based weight estimation
    - feature_selection: Feature selection
    - plotter: MPO-specific visualization functions
"""

from mosses.core.mpo.scoring import (
    sigmoid,
    reverse_sigmoid,
    double_sigmoid,
    create_scoring_columns,
    ScoringConfig,
    ParameterConfig,
)
from mosses.core.mpo.weights import (
    normalize_weights,
    update_dict_keys,
    WeightConfiguration,
)
from mosses.core.mpo.statistics import (
    calculate_enrichment,
    calculate_spearman_correlation,
    find_top_n_percent_ids,
    custom_f1,
    create_class_label,
    collect_stats,
    MPOStatistics,
)

__all__ = [
    # Scoring
    "sigmoid",
    "reverse_sigmoid",
    "double_sigmoid",
    "create_scoring_columns",
    "ScoringConfig",
    "ParameterConfig",
    # Weights
    "normalize_weights",
    "update_dict_keys",
    "WeightConfiguration",
    # Statistics
    "calculate_enrichment",
    "calculate_spearman_correlation",
    "find_top_n_percent_ids",
    "custom_f1",
    "create_class_label",
    "collect_stats",
    "MPOStatistics",
]

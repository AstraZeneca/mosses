"""
Multi-Parameter Optimization (MPO) Module.

This module provides a high-level API for Multi-Parameter Optimization
analysis of compound data. MPO combines multiple molecular properties
into a single score using sigmoid-based desirability functions.

The module includes:
    - Sigmoid-based scoring functions
    - Weight optimization algorithms
    - ML-based weight estimation
    - Feature selection utilities
    - Statistics and enrichment analysis
    - Visualization tools

Example
-------
>>> from mosses import mpo
>>> 
>>> # Define parameters and thresholds
>>> config = {
...     "LogD": {"preference": "middle", "threshold": (0.0, 3.0), "weight": 1.0},
...     "Sol": {"preference": "maximize", "threshold": 50, "weight": 1.0},
...     "CL": {"preference": "minimize", "threshold": 100, "weight": 1.0},
... }
>>> 
>>> # Calculate MPO scores
>>> result = mpo.compute_scores(df, config)
>>> print(result.mpo_scores.describe())
"""

from typing import Any

import numpy as np
import pandas as pd

from mosses.core.mpo.feature_selection import (
    FeatureSelectionResult,
    analyze_feature_importance,
    select_features,
)
from mosses.core.mpo.ml_estimators import (
    ML_ESTIMATORS,
    MLEstimatorResult,
    estimate_weights_ml,
    logistic_classifier,
    rf_classifier,
    rf_regression,
    ridge_classifier,
)
from mosses.core.mpo.optimizers import (
    OPTIMIZERS,
    OptimizerMethod,
    OptimizationResult,
    optimize_weights,
    optimize_weights_differential_evolution,
    optimize_weights_dual_annealing,
    optimize_weights_least_squares,
    optimize_weights_minimize,
    optimize_weights_powell,
    optimize_weights_pygad,
)
from mosses.core.mpo.plotter import (
    plot_best_fit_scatter,
    plot_comparison,
    plot_experimental_correlation_matrix,
    plot_id_occurrences,
    plot_mpo_histogram,
    plot_mutual_info,
    plot_parameter_correlation_matrix,
    plot_predicted_correlation_matrix,
    plot_roc_curve,
    plot_scoring_curves,
)
from mosses.core.mpo.scoring import (
    ParameterConfig,
    ScoringConfig,
    compute_mpo_scores,
    create_scoring_columns,
    double_sigmoid,
    reverse_sigmoid,
    sigmoid,
)
from mosses.core.mpo.statistics import (
    MPOStatistics,
    calculate_enrichment,
    calculate_spearman_correlation,
    check_stats,
    collect_stats,
    create_class_label,
    custom_f1,
    find_top_n_percent_ids,
)
from mosses.core.mpo.weights import (
    WeightConfiguration,
    normalize_coefficients_abs,
    normalize_weights,
    quick_median_thresholds,
    update_dict_keys,
)

# Re-export core classes and functions for convenient access
__all__ = [
    # Main API functions
    "compute_scores",
    "optimize_mpo_weights",
    "evaluate_mpo",
    "build_mpo_pipeline",
    # Scoring
    "sigmoid",
    "reverse_sigmoid",
    "double_sigmoid",
    "create_scoring_columns",
    "compute_mpo_scores",
    "ParameterConfig",
    "ScoringConfig",
    # Weights
    "normalize_weights",
    "update_dict_keys",
    "normalize_coefficients_abs",
    "quick_median_thresholds",
    "WeightConfiguration",
    # Optimizers
    "optimize_weights",
    "optimize_weights_least_squares",
    "optimize_weights_minimize",
    "optimize_weights_dual_annealing",
    "optimize_weights_differential_evolution",
    "optimize_weights_powell",
    "optimize_weights_pygad",
    "OptimizerMethod",
    "OptimizationResult",
    "OPTIMIZERS",
    # ML Estimators
    "estimate_weights_ml",
    "rf_regression",
    "rf_classifier",
    "logistic_classifier",
    "ridge_classifier",
    "MLEstimatorResult",
    "ML_ESTIMATORS",
    # Feature Selection
    "select_features",
    "analyze_feature_importance",
    "FeatureSelectionResult",
    # Statistics
    "find_top_n_percent_ids",
    "calculate_enrichment",
    "calculate_spearman_correlation",
    "custom_f1",
    "create_class_label",
    "collect_stats",
    "check_stats",
    "MPOStatistics",
    # Plotting
    "plot_scoring_curves",
    "plot_mpo_histogram",
    "plot_comparison",
    "plot_mutual_info",
    "plot_parameter_correlation_matrix",
    "plot_experimental_correlation_matrix",
    "plot_predicted_correlation_matrix",
    "plot_best_fit_scatter",
    "plot_roc_curve",
    "plot_id_occurrences",
]


def compute_scores(
    df: pd.DataFrame,
    config: dict[str, dict[str, Any]],
    steepness: float = 2.0,
    id_column: str = "Compound Name",
    return_intermediate: bool = False,
) -> pd.DataFrame:
    """
    Compute MPO scores for a DataFrame of compounds.

    This is a high-level convenience function that handles the full
    scoring pipeline including:
    - Creating scoring columns for each parameter
    - Computing weighted MPO scores
    - Optionally returning intermediate score columns

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with compound data.
    config : dict[str, dict[str, Any]]
        Configuration dictionary where keys are parameter names and values
        are dictionaries with:
        - "column": (optional) actual column name if different from key
        - "preference": "maximize", "minimize", or "middle"
        - "threshold": float or tuple[float, float] for "middle"
        - "weight": float weight for this parameter (default: 1.0)
    steepness : float, optional
        Steepness of sigmoid functions (default: 2.0).
    id_column : str, optional
        Name of identifier column (default: "Compound Name").
    return_intermediate : bool, optional
        If True, include individual score columns (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with MPO_Score column (and individual scores if requested).

    Examples
    --------
    >>> config = {
    ...     "LogD": {"preference": "middle", "threshold": (0, 3), "weight": 1.0},
    ...     "Solubility": {"preference": "maximize", "threshold": 50, "weight": 1.5},
    ...     "Clearance": {"preference": "minimize", "threshold": 100, "weight": 1.0},
    ... }
    >>> result = compute_scores(df, config)
    >>> print(result["MPO_Score"].describe())
    """
    result_df = df.copy()

    # Extract configuration
    preferences = {}
    thresholds = {}
    weights = {}
    param_columns = {}

    for param_name, param_config in config.items():
        column_name = param_config.get("column", param_name)
        preferences[param_name] = param_config.get("preference", "maximize")
        thresholds[param_name] = param_config["threshold"]
        weights[param_name] = param_config.get("weight", 1.0)
        param_columns[param_name] = column_name

    # Create scoring columns
    score_column_names = []
    for param_name, column_name in param_columns.items():
        if column_name not in result_df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        score_col = f"{param_name}_score"
        preference = preferences[param_name]
        threshold = thresholds[param_name]

        if preference == "maximize":
            result_df[score_col] = sigmoid(
                result_df[column_name], threshold=threshold, steepness=steepness
            )
        elif preference == "minimize":
            result_df[score_col] = reverse_sigmoid(
                result_df[column_name], threshold=threshold, steepness=steepness
            )
        elif preference == "middle":
            if isinstance(threshold, tuple) and len(threshold) == 2:
                result_df[score_col] = double_sigmoid(
                    result_df[column_name],
                    lower_threshold=threshold[0],
                    upper_threshold=threshold[1],
                    steepness=2 * steepness,
                )
            else:
                raise ValueError(
                    f"Parameter '{param_name}' with preference 'middle' "
                    "requires a tuple of (lower, upper) thresholds."
                )
        else:
            raise ValueError(
                f"Unknown preference '{preference}' for parameter '{param_name}'."
            )

        score_column_names.append(score_col)

    # Normalize weights
    weight_values = [weights[param] for param in param_columns.keys()]
    total_weight = sum(weight_values)
    normalized_weights = [w / total_weight for w in weight_values]

    # Compute weighted MPO score
    mpo_score = np.zeros(len(result_df))
    for score_col, norm_weight in zip(score_column_names, normalized_weights):
        mpo_score += result_df[score_col].values * norm_weight

    result_df["MPO_Score"] = mpo_score

    # Return requested columns
    if return_intermediate:
        output_cols = [id_column] if id_column in result_df.columns else []
        output_cols.extend(score_column_names)
        output_cols.append("MPO_Score")
        return result_df[output_cols]

    output_cols = [id_column] if id_column in result_df.columns else []
    output_cols.append("MPO_Score")
    return result_df[output_cols]


def optimize_mpo_weights(
    df: pd.DataFrame,
    score_columns: list[str],
    target_column: str,
    method: OptimizerMethod | str = OptimizerMethod.LEAST_SQUARES,
    normalize: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, float], OptimizationResult]:
    """
    Optimize MPO weights to match a target score.

    Uses various optimization algorithms to find weights that minimize
    the difference between computed MPO scores and target values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing score columns and target.
    score_columns : list[str]
        Names of columns containing individual parameter scores.
    target_column : str
        Name of target column to optimize against.
    method : str, optional
        Optimization method (default: "least_squares").
        Options: "least_squares", "minimize", "dual_annealing",
        "differential_evolution", "powell", "pygad".
    normalize : bool, optional
        Whether to normalize resulting weights (default: True).
    verbose : bool, optional
        Print optimization details (default: False).
    **kwargs
        Additional arguments passed to optimizer.

    Returns
    -------
    tuple[dict[str, float], OptimizationResult]
        Dictionary mapping column names to weights, and full result object.

    Examples
    --------
    >>> score_cols = ["LogD_score", "Sol_score", "CL_score"]
    >>> weights, result = optimize_mpo_weights(
    ...     df, score_cols, "Experimental_Activity",
    ...     method="differential_evolution"
    ... )
    >>> print(weights)
    {"LogD_score": 0.25, "Sol_score": 0.45, "CL_score": 0.30}
    """
    # Run optimization
    result = optimize_weights(
        df, score_columns, target_column,
        method=method, verbose=verbose, **kwargs
    )

    # Get weights dictionary from result (already normalized by optimizer)
    weights_dict = result.weights

    # Further normalize if requested
    if normalize:
        total = sum(abs(w) for w in weights_dict.values())
        if total > 0:
            weights_dict = {k: v / total for k, v in weights_dict.items()}

    return weights_dict, result


def evaluate_mpo(
    df: pd.DataFrame,
    mpo_column: str,
    reference_column: str,
    id_column: str = "Compound Name",
    top_percent: float = 10.0,
    verbose: bool = True,
) -> MPOStatistics:
    """
    Evaluate MPO performance against a reference.

    Computes enrichment, correlation, and F1 statistics comparing
    the MPO scores against a reference column (e.g., experimental data).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing MPO and reference columns.
    mpo_column : str
        Name of the MPO score column.
    reference_column : str
        Name of the reference/target column.
    id_column : str, optional
        Name of identifier column (default: "Compound Name").
    top_percent : float, optional
        Percentage for top compound analysis (default: 10.0).
    verbose : bool, optional
        Print evaluation results (default: True).

    Returns
    -------
    MPOStatistics
        Dataclass containing all evaluation metrics.

    Examples
    --------
    >>> stats = evaluate_mpo(df, "MPO_Score", "Experimental_Activity")
    >>> print(f"Enrichment: {stats.enrichment:.2f}")
    >>> print(f"Correlation: {stats.spearman_correlation:.3f}")
    """
    stats = collect_stats(
        method_name=mpo_column,
        results_df=df,
        reference_column=reference_column,
        method_column=mpo_column,
        percent_top=top_percent,
        id_column=id_column,
        verbose=verbose,
    )

    return stats


def build_mpo_pipeline(
    df: pd.DataFrame,
    experimental_columns: list[str],
    target_column: str | None = None,
    preferences: dict[str, str] | None = None,
    thresholds: dict[str, float | tuple[float, float]] | None = None,
    weights: dict[str, float] | None = None,
    auto_threshold: bool = True,
    optimize_weights_method: str | None = None,
    steepness: float = 2.0,
    id_column: str = "Compound Name",
) -> pd.DataFrame:
    """
    Build a complete MPO scoring pipeline.

    This is a convenience function that orchestrates the full MPO workflow:
    1. Automatically determine thresholds (if not provided)
    2. Create scoring columns for each parameter
    3. Optionally optimize weights against a target
    4. Compute final MPO scores

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with compound data.
    experimental_columns : list[str]
        List of experimental measurement columns.
    target_column : str | None, optional
        Target column for weight optimization (default: None).
    preferences : dict[str, str] | None, optional
        Parameter optimization directions. If None, defaults to "maximize".
    thresholds : dict[str, float | tuple] | None, optional
        Parameter thresholds. If None and auto_threshold=True, uses median.
    weights : dict[str, float] | None, optional
        Parameter weights. If None, uses equal weights.
    auto_threshold : bool, optional
        Automatically calculate thresholds from data (default: True).
    optimize_weights_method : str | None, optional
        If provided, optimize weights using this method.
    steepness : float, optional
        Sigmoid steepness parameter (default: 2.0).
    id_column : str, optional
        Name of identifier column (default: "Compound Name").

    Returns
    -------
    pd.DataFrame
        DataFrame with added score columns and final MPO_Score.

    Examples
    --------
    >>> # Simple usage with auto thresholds
    >>> result = build_mpo_pipeline(
    ...     df,
    ...     experimental_columns=["LogD", "Solubility", "Clearance"],
    ...     preferences={"LogD": "middle", "Solubility": "maximize", "Clearance": "minimize"},
    ... )
    >>> 
    >>> # With optimization against experimental target
    >>> result = build_mpo_pipeline(
    ...     df,
    ...     experimental_columns=["LogD", "Solubility", "Clearance"],
    ...     target_column="Activity",
    ...     optimize_weights_method="differential_evolution",
    ... )
    """
    result_df = df.copy()

    # Set default preferences
    if preferences is None:
        preferences = {col: "maximize" for col in experimental_columns}

    # Auto-calculate thresholds if needed
    if thresholds is None and auto_threshold:
        # Simple median-based thresholds for each column
        thresholds = {col: df[col].median() for col in experimental_columns}

    # Set default equal weights
    if weights is None:
        weights = {col: 1.0 for col in experimental_columns}

    # Create scoring columns
    score_columns = []
    for col in experimental_columns:
        score_col = f"{col}_score"
        preference = preferences.get(col, "maximize")
        threshold = thresholds.get(col) if thresholds else None

        if threshold is None:
            threshold = df[col].median()

        if preference == "maximize":
            result_df[score_col] = sigmoid(
                result_df[col], threshold=threshold, steepness=steepness
            )
        elif preference == "minimize":
            result_df[score_col] = reverse_sigmoid(
                result_df[col], threshold=threshold, steepness=steepness
            )
        elif preference == "middle":
            if isinstance(threshold, tuple) and len(threshold) == 2:
                result_df[score_col] = double_sigmoid(
                    result_df[col],
                    lower_threshold=threshold[0],
                    upper_threshold=threshold[1],
                    steepness=2 * steepness,
                )
            else:
                median_val = df[col].median()
                std_val = df[col].std()
                result_df[score_col] = double_sigmoid(
                    result_df[col],
                    lower_threshold=median_val - std_val,
                    upper_threshold=median_val + std_val,
                    steepness=2 * steepness,
                )

        score_columns.append(score_col)

    # Optimize weights if target provided
    if optimize_weights_method and target_column:
        weights_dict, _ = optimize_mpo_weights(
            result_df,
            score_columns,
            target_column,
            method=optimize_weights_method,
            normalize=True,
        )
        # Map back to original column names
        weights = {
            col: weights_dict.get(f"{col}_score", 1.0) for col in experimental_columns
        }

    # Compute final MPO score
    normalized_weights = normalize_weights(weights)
    mpo_score = np.zeros(len(result_df))

    for col in experimental_columns:
        score_col = f"{col}_score"
        weight = normalized_weights.get(col, 1.0 / len(experimental_columns))
        mpo_score += result_df[score_col].values * weight

    result_df["MPO_Score"] = mpo_score

    return result_df

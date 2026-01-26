"""
Statistics and metrics for Multi-Parameter Optimization (MPO).

This module provides functions for calculating enrichment, correlation,
and other performance metrics for MPO score evaluation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class MPOStatistics:
    """
    Container for MPO evaluation statistics.

    Attributes
    ----------
    method : str
        Name of the MPO method used.
    weights : dict[str, float]
        Dictionary of parameter weights.
    thresholds : dict[str, float | tuple[float, float]]
        Dictionary of parameter thresholds.
    rmse : float
        Root Mean Squared Error to reference.
    enrichment : float
        Enrichment percentage.
    spearman_correlation : float
        Spearman correlation coefficient.
    f1_score : float
        F1 score for top compound identification.
    top_compounds : list[str]
        List of top compound identifiers.
    shared_with_reference : list[str]
        Compounds shared with reference top set.
    """

    method: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    rmse: float = 0.0
    enrichment: float = 0.0
    spearman_correlation: float = 0.0
    f1_score: float = 0.0
    top_compounds: list = field(default_factory=list)
    shared_with_reference: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "Method": self.method,
            "RMSE to Reference": self.rmse,
            "Enrichment": self.enrichment,
            "Spearman Correlation": self.spearman_correlation,
            "F1 score": self.f1_score,
            "Top cmpds": self.top_compounds,
            "Top shared with Reference": self.shared_with_reference,
        }
        # Add weights
        for key, value in self.weights.items():
            result[f"{key} weight"] = value
        # Add thresholds
        for key, value in self.thresholds.items():
            result[f"{key} threshold"] = value
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


def find_top_n_percent_ids(
    percent_top: float,
    df: pd.DataFrame,
    score_column: str,
    id_column: str = "Compound Name",
) -> list:
    """
    Find compound IDs in the top N percent of scores.

    Uses range-based percentile calculation: compounds are selected
    if their score >= min + (1 - percent_top/100) * range.

    Parameters
    ----------
    percent_top : float
        Percentage of top compounds to select (e.g., 10 for top 10%).
    df : pd.DataFrame
        DataFrame containing the scores.
    score_column : str
        Name of the column containing scores.
    id_column : str, optional
        Name of the column containing compound identifiers
        (default: "Compound Name").

    Returns
    -------
    list
        List of unique compound identifiers in the top N percent.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Compound Name': ['A', 'B', 'C', 'D', 'E'],
    ...     'score': [0.1, 0.3, 0.5, 0.7, 0.9]
    ... })
    >>> find_top_n_percent_ids(20, df, 'score')
    ['E']  # Top 20% by score range
    """
    score_range = df[score_column].max() - df[score_column].min()

    if score_range == 0:
        # All scores are the same, return all IDs
        return list(df[id_column].unique())

    cutoff_score = df[score_column].min() + (1 - percent_top / 100) * score_range

    top_ids = df[df[score_column] >= cutoff_score][id_column]
    return list(set(top_ids))


def calculate_enrichment(
    percent_top: float,
    df: pd.DataFrame,
    reference_column: str,
    method_column: str,
    id_column: str = "Compound Name",
) -> float:
    """
    Calculate enrichment of a method compared to a reference.

    Enrichment measures how well the method identifies the same
    top compounds as the reference.

    Parameters
    ----------
    percent_top : float
        Percentage of top compounds to consider (e.g., 10 for top 10%).
    df : pd.DataFrame
        DataFrame containing both score columns.
    reference_column : str
        Name of the reference score column.
    method_column : str
        Name of the method score column to evaluate.
    id_column : str, optional
        Name of the compound identifier column
        (default: "Compound Name").

    Returns
    -------
    float
        Enrichment percentage (0-100). Higher is better.
        100% means perfect overlap with reference top compounds.

    Examples
    --------
    >>> enrichment = calculate_enrichment(10, df, 'ref_score', 'pred_score')
    """
    top_ref = find_top_n_percent_ids(percent_top, df, reference_column, id_column)
    top_method = find_top_n_percent_ids(percent_top, df, method_column, id_column)

    if len(top_method) == 0:
        return 0.0

    overlap = len(set(top_ref) & set(top_method))
    enrichment = 100 * (overlap / len(top_method))

    return enrichment


def calculate_spearman_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    verbose: bool = False,
) -> float:
    """
    Calculate the Spearman correlation between two columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    col1 : str
        Name of the first column.
    col2 : str
        Name of the second column.
    verbose : bool, optional
        If True, print the correlation value (default: False).

    Returns
    -------
    float
        Spearman correlation coefficient, rounded to 3 decimal places.

    Raises
    ------
    ValueError
        If either column does not exist in the DataFrame.

    Examples
    --------
    >>> corr = calculate_spearman_correlation(df, 'actual', 'predicted')
    """
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(
            f"Columns '{col1}' and/or '{col2}' do not exist in the DataFrame."
        )

    correlation, _ = spearmanr(df[col1], df[col2])
    correlation = round(correlation, 3)

    if verbose:
        print(f"Spearman correlation: {correlation:.3f}")

    return correlation


def custom_f1(
    true_positives: list,
    predictions: list,
) -> float:
    """
    Calculate F1 score for set overlap (compound identification).

    This is a set-based F1 calculation, useful for comparing
    which compounds are identified as "top" by two methods.

    Parameters
    ----------
    true_positives : list
        List of true positive identifiers (reference top compounds).
    predictions : list
        List of predicted positive identifiers (method top compounds).

    Returns
    -------
    float
        F1 score in range [0, 1]. Higher is better.

    Examples
    --------
    >>> custom_f1(['A', 'B', 'C'], ['B', 'C', 'D'])
    0.666...  # 2 overlap out of 3 each
    """
    true_set = set(true_positives)
    pred_set = set(predictions)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def create_class_label(
    values: list | np.ndarray | pd.Series,
    percent_top: float,
) -> np.ndarray:
    """
    Create binary class labels based on top N percent threshold.

    Labels compounds as 1 ("good") if they are in the top N percent
    of the score range, and 0 ("bad") otherwise.

    Parameters
    ----------
    values : list | np.ndarray | pd.Series
        Score values to classify.
    percent_top : float
        Percentage of range to consider as "good" (e.g., 20 for top 20%).

    Returns
    -------
    np.ndarray
        Binary array with 1 for "good" compounds and 0 for "bad".

    Examples
    --------
    >>> scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    >>> create_class_label(scores, 20)
    array([0, 0, 0, 0, 1])
    """
    values = np.array(values)

    score_range = values.max() - values.min()

    if score_range == 0:
        # All values are the same - return all zeros
        return np.zeros(len(values), dtype=int)

    cutoff_score = values.min() + (1 - percent_top / 100) * score_range

    return (values >= cutoff_score).astype(int)


def collect_stats(
    method_name: str,
    results_df: pd.DataFrame,
    reference_column: str,
    method_column: str,
    weights: dict[str, float] | None = None,
    thresholds: dict | None = None,
    percent_top: float = 10.0,
    id_column: str = "Compound Name",
    verbose: bool = False,
) -> MPOStatistics:
    """
    Collect comprehensive statistics for an MPO method.

    Parameters
    ----------
    method_name : str
        Name of the method being evaluated.
    results_df : pd.DataFrame
        DataFrame containing both reference and method scores.
    reference_column : str
        Name of the reference score column.
    method_column : str
        Name of the method score column.
    weights : dict[str, float] | None, optional
        Dictionary of parameter weights used.
    thresholds : dict | None, optional
        Dictionary of parameter thresholds used.
    percent_top : float, optional
        Percentage for top compound selection (default: 10.0).
    id_column : str, optional
        Name of compound identifier column (default: "Compound Name").
    verbose : bool, optional
        If True, print statistics during calculation (default: False).

    Returns
    -------
    MPOStatistics
        Dataclass containing all computed statistics.

    Examples
    --------
    >>> stats = collect_stats(
    ...     'my_method', df, 'ref_score', 'my_score',
    ...     weights={'logP': 0.5, 'MW': 0.5},
    ...     percent_top=20
    ... )
    >>> print(stats.f1_score)
    """
    weights = weights or {}
    thresholds = thresholds or {}

    # Calculate RMSE
    rmse = np.sqrt(
        ((results_df[reference_column] - results_df[method_column]) ** 2).mean()
    )

    # Calculate enrichment
    enrichment = calculate_enrichment(
        percent_top, results_df, reference_column, method_column, id_column
    )

    # Calculate Spearman correlation
    spearman = calculate_spearman_correlation(
        results_df, reference_column, method_column, verbose=verbose
    )

    # Find top compounds
    ref_top = find_top_n_percent_ids(
        percent_top, results_df, reference_column, id_column
    )
    method_top = find_top_n_percent_ids(
        percent_top, results_df, method_column, id_column
    )

    # Calculate F1
    f1 = custom_f1(ref_top, method_top)

    # Find shared compounds
    shared = list(set(ref_top) & set(method_top))

    return MPOStatistics(
        method=method_name,
        weights=weights,
        thresholds=thresholds,
        rmse=rmse,
        enrichment=enrichment,
        spearman_correlation=spearman,
        f1_score=f1,
        top_compounds=method_top,
        shared_with_reference=shared,
    )


def check_stats(
    df: pd.DataFrame,
    reference_methods: list[str],
    parameter_columns: list[str],
    n_best_methods: int = 3,
    max_std: float = 0.5,
) -> list:
    """
    Check statistics and identify the best method.

    Analyzes method performance and warns if there are issues
    with result reliability (e.g., high weight variance with
    similar F1 scores).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with method statistics (from collect_stats).
    reference_methods : list[str]
        Names of reference methods to compare against.
    parameter_columns : list[str]
        Names of parameter weight columns to analyze.
    n_best_methods : int, optional
        Number of top methods to consider (default: 3).
    max_std : float, optional
        Maximum acceptable standard deviation for weights
        when F1 scores are similar (default: 0.5).

    Returns
    -------
    list
        List containing the best method name and its weights,
        or empty list if results are unreliable.
    """
    other_rows = df[~df["Method"].isin(reference_methods)]

    metric_1 = "F1 score"
    metric_2 = "Enrichment"

    # Get reference metrics
    if len(reference_methods) > 0 and reference_methods[-1] in df["Method"].values:
        ref_metric_1 = df.loc[df["Method"] == reference_methods[-1], metric_1].iloc[0]
        ref_metric_2 = df.loc[df["Method"] == reference_methods[-1], metric_2].iloc[0]
    else:
        ref_metric_1 = 0
        ref_metric_2 = 0

    # Identify top methods
    top_methods = other_rows.sort_values(
        [metric_1, metric_2], ascending=[False, False]
    ).head(n_best_methods)

    # Compare and print warnings
    for idx, row in top_methods.iterrows():
        if row[metric_1] < ref_metric_1:
            print(
                f"Warning: Method {row['Method']} has lower {metric_1} "
                f"({row[metric_1]}) than reference ({ref_metric_1})"
            )
        if row[metric_2] < ref_metric_2:
            print(
                f"Warning: Method {row['Method']} has lower {metric_2} "
                f"({row[metric_2]}) than reference ({ref_metric_2})"
            )

    # Check for weight stability
    existing_param_cols = [c for c in parameter_columns if c in df.columns]
    if len(existing_param_cols) > 0:
        summary_stats_weights = df[existing_param_cols].describe()
        summary_stats_metrics = top_methods[[metric_1]].describe()

        weights_std = summary_stats_weights.loc["std"].mean()
        metrics_std = summary_stats_metrics.loc["std"].mean()

        if weights_std > max_std and metrics_std <= max_std:
            print(
                f"WARNING: Average weight std ({weights_std:.2f}) is larger than "
                f"{max_std} for similar F1 performance. Results may be unreliable!"
            )
            return []

    # Return best method
    if len(top_methods) > 0:
        best_method = top_methods.head(1).reset_index(drop=True)
        method_name = best_method["Method"].iloc[0]
        print(f"The best method is {method_name}.")

        result = [method_name]
        print("Recommended weights for each parameter:")
        for p in existing_param_cols:
            weight_val = best_method[p].iloc[0]
            print(f"  {p}: {weight_val}")
            result.append(weight_val)

        return result

    return []

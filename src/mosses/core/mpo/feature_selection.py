"""
Feature selection utilities for Multi-Parameter Optimization (MPO).

This module provides functions for selecting relevant parameters
based on mutual information and correlation analysis.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


@dataclass
class FeatureSelectionResult:
    """
    Result of feature selection analysis.

    Attributes
    ----------
    selected_features : list[str]
        List of selected feature names.
    mutual_information : pd.DataFrame
        DataFrame with feature names and MI (Mutual Information) scores.
    noise_threshold : float
        The noise threshold used for selection.
    correlation_threshold : float
        The correlation threshold used for filtering.
    """

    selected_features: list[str] = field(default_factory=list)
    mutual_information: pd.DataFrame = field(default_factory=pd.DataFrame)
    noise_threshold: float = 0.0
    correlation_threshold: float = 0.9


def mutual_information_score(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_shuffles: int = 100,
    random_state: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Calculate mutual information scores with noise floor estimation.

    Uses permutation testing to establish a noise floor, helping
    to identify which features have significant mutual information.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Feature matrix.
    y : pd.Series | np.ndarray
        Target variable.
    n_shuffles : int, optional
        Number of permutations for noise estimation (default: 100).
    random_state : int, optional
        Random seed for reproducibility (default: 0).

    Returns
    -------
    tuple[np.ndarray, float]
        - Array of mutual information scores for each feature
        - 95th percentile noise threshold

    Examples
    --------
    >>> mi, threshold = mutual_information_score(X, y)
    >>> significant = mi > threshold
    """
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    mi_noise_scores = []

    for _ in range(n_shuffles):
        y_shuffled = np.random.permutation(y_arr)
        mi_noise = mutual_info_regression(X_arr, y_shuffled, random_state=random_state)
        mi_noise_scores.append(mi_noise)

    mi_noise_scores = np.array(mi_noise_scores)  # shape: (n_shuffles, n_features)
    noise_percentile = np.percentile(mi_noise_scores, 95)

    # Compute actual mutual information scores
    mi = mutual_info_regression(X_arr, y_arr, random_state=random_state)

    return mi, noise_percentile


def select_by_correlation(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    threshold: float = 0.8,
    verbose: bool = False,
) -> list[str]:
    """
    Select features by removing highly correlated pairs.

    For each pair of features with correlation above threshold,
    keeps the one more correlated with the target column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and target.
    features : list[str]
        List of feature column names to consider.
    target_col : str
        Name of the target column for comparison.
    threshold : float, optional
        Correlation threshold (default: 0.8).
        Features with higher inter-correlation will be filtered.
    verbose : bool, optional
        Print selection decisions (default: False).

    Returns
    -------
    list[str]
        List of selected feature names.

    Examples
    --------
    >>> selected = select_by_correlation(df, ['logP', 'MW', 'tPSA'], 'mpo_score')
    """
    if len(features) == 0:
        return []

    features_to_keep = list(features)

    while True:
        # Build correlation matrix for current features
        corr_matrix = df[features_to_keep].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)

        # Find pairs above threshold
        mask = corr_matrix >= threshold
        if not mask.any().any():
            if verbose:
                print("No correlated pairs above threshold remaining.")
            break

        # Process pairs
        to_remove = set()
        pairs = np.argwhere(mask.values)
        checked_features = set()

        for i, j in pairs:
            f1 = corr_matrix.index[i]
            f2 = corr_matrix.columns[j]

            if f1 in checked_features or f2 in checked_features:
                continue

            if verbose:
                print(f"Comparing {f1} and {f2} (corr={corr_matrix.loc[f1, f2]:.2f})")

            # Keep the one more correlated with target
            corr_with_target = df[[f1, f2]].corrwith(df[target_col]).abs()
            remove_f = f1 if corr_with_target[f1] < corr_with_target[f2] else f2

            to_remove.add(remove_f)
            if verbose:
                print(f"  Removing {remove_f}")

            checked_features.add(f1)
            checked_features.add(f2)

        features_to_keep = [f for f in features_to_keep if f not in to_remove]

        if not features_to_keep:
            raise ValueError(
                "All features removed. Consider increasing the correlation threshold."
            )

    return features_to_keep


def select_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    correlation_threshold: float = 0.9,
    n_shuffles: int = 100,
    parameter_columns_dict: dict[str, tuple[str, str]] | None = None,
    verbose: bool = False,
) -> FeatureSelectionResult:
    """
    Select relevant features using mutual information and correlation.

    Two-step selection process:
    1. Filter features with MI above noise threshold
    2. Remove redundant features based on inter-correlation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names to consider.
    reference_col : str
        Name of the reference column.
    correlation_threshold : float, optional
        Threshold for inter-feature correlation (default: 0.9).
    n_shuffles : int, optional
        Number of permutations for MI noise estimation (default: 100).
    parameter_columns_dict : dict[str, tuple[str, str]] | None, optional
        Mapping of parameter names to (exp, pred) column tuples.
        Used to map back selected columns to parameter names.
    verbose : bool, optional
        Print selection information (default: False).

    Returns
    -------
    FeatureSelectionResult
        Result containing selected features and analysis details.

    Examples
    --------
    >>> result = select_features(df, ['logP_pred', 'MW_pred'], 'mpo_ref')
    >>> print(result.selected_features)
    """
    # Validate inputs
    valid_cols = [c for c in feature_cols if c in df.columns]
    if len(valid_cols) == 0:
        return FeatureSelectionResult()

    X = df[valid_cols]
    y = df[reference_col]

    # Calculate mutual information
    mi, noise_percentile = mutual_information_score(X, y, n_shuffles=n_shuffles)

    # Create MI dataframe
    mi_df = pd.DataFrame({
        "Feature": valid_cols,
        "Mutual Information": mi,
    })

    # Select features above noise threshold
    selected_features = mi_df[mi_df["Mutual Information"] > noise_percentile][
        "Feature"
    ].tolist()

    if verbose:
        print(f"Features above noise threshold ({noise_percentile:.4f}):")
        print(f"  {selected_features}")

    if len(selected_features) == 0:
        if verbose:
            print("No features passed MI threshold. Using all features.")
        selected_features = valid_cols

    # Filter by correlation
    selected_params = select_by_correlation(
        df,
        selected_features,
        target_col=reference_col,
        threshold=correlation_threshold,
        verbose=verbose,
    )

    # Map back to parameter names if mapping provided
    if parameter_columns_dict is not None:
        mapped_params = []
        for param, cols in parameter_columns_dict.items():
            if any(col in selected_params for col in cols):
                mapped_params.append(param)
        if verbose:
            print(f"Selected parameters: {mapped_params}")

    return FeatureSelectionResult(
        selected_features=selected_params,
        mutual_information=mi_df,
        noise_threshold=noise_percentile,
        correlation_threshold=correlation_threshold,
    )


def get_feature_correlation_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Get correlation matrix for specified features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features.
    features : list[str]
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    valid_features = [f for f in features if f in df.columns]
    return df[valid_features].corr()


def analyze_feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Analyze feature importance using multiple methods.

    Combines mutual information and correlation analysis
    to provide a comprehensive view of feature importance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    verbose : bool, optional
        Print analysis results (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with feature analysis results including:
        - Feature name
        - Mutual information score
        - Correlation with reference
        - Above noise threshold flag
    """
    valid_cols = [c for c in feature_cols if c in df.columns]

    X = df[valid_cols]
    y = df[reference_col]

    # Mutual information
    mi, noise_threshold = mutual_information_score(X, y)

    # Correlation with reference
    correlations = df[valid_cols].corrwith(y).abs()

    # Build result dataframe
    result = pd.DataFrame({
        "Feature": valid_cols,
        "Mutual Information": mi,
        "Correlation": correlations.values,
        "Above Noise": mi > noise_threshold,
    }).sort_values("Mutual Information", ascending=False)

    if verbose:
        print(f"Noise threshold: {noise_threshold:.4f}")
        print(result.to_string(index=False))

    return result

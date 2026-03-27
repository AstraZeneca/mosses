"""
Weight management utilities for Multi-Parameter Optimization (MPO).

This module provides functions for normalizing and managing parameter
weights used in MPO score calculations.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WeightConfiguration:
    """
    Configuration for parameter weights in MPO calculation.

    Attributes
    ----------
    weights : dict[str, float]
        Dictionary mapping parameter names to their weights.
    normalized : bool
        Whether the weights have been normalized to sum to 1.

    Examples
    --------
    >>> wc = WeightConfiguration(
    ...     weights={'logP': 1, 'MW': 2, 'solubility': 3}
    ... )
    >>> wc.normalize()
    >>> wc.weights
    {'logP': 0.166..., 'MW': 0.333..., 'solubility': 0.5}
    """

    weights: dict[str, float] = field(default_factory=dict)
    normalized: bool = False

    def normalize(self) -> None:
        """Normalize weights so they sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.normalized = True

    def get_normalized(self) -> dict[str, float]:
        """Return normalized weights without modifying the original."""
        total = sum(self.weights.values())
        if total == 0:
            return {k: 1.0 / len(self.weights) for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}

    def to_array(self, columns: list[str]) -> np.ndarray:
        """
        Convert weights to numpy array in the order of specified columns.

        Parameters
        ----------
        columns : list[str]
            List of column names defining the order.

        Returns
        -------
        np.ndarray
            Array of weights in the specified order.
        """
        normalized = self.get_normalized()
        return np.array([normalized.get(col, 0.0) for col in columns])


def normalize_weights(weights: dict[str, float | int]) -> dict[str, float]:
    """
    Normalize weights so they sum to 1.

    Parameters
    ----------
    weights : dict[str, float | int]
        Dictionary of weights for each parameter.
        Example: {"logP": 1, "tPSA": 2, "solubility": 3}

    Returns
    -------
    dict[str, float]
        Dictionary of normalized weights where the weights sum to 1.

    Examples
    --------
    >>> normalize_weights({"logP": 1, "tPSA": 2, "solubility": 3})
    {'logP': 0.166..., 'tPSA': 0.333..., 'solubility': 0.5}
    """
    total = sum(weights.values())
    if total == 0:
        # Avoid division by zero - return equal weights
        n = len(weights)
        return {param: 1.0 / n for param in weights} if n > 0 else {}
    return {param: weight / total for param, weight in weights.items()}


def update_dict_keys(
    original_dict: dict,
    parameter_columns_dict: dict[str, tuple[str, str]],
    selection: str,
) -> dict:
    """
    Update dictionary keys based on parameter columns mapping.

    This function is useful when you have parameter names as keys
    but need to map them to actual column names in a DataFrame
    (either experimental or predicted columns).

    Parameters
    ----------
    original_dict : dict
        Original dictionary with parameter names as keys.
    parameter_columns_dict : dict[str, tuple[str, str]]
        Dictionary mapping parameter names to (experimental, predicted)
        column name tuples.
    selection : str
        Either "exp" for experimental columns or "pred" for predicted.

    Returns
    -------
    dict
        New dictionary with updated keys.

    Raises
    ------
    ValueError
        If selection is not "exp" or "pred".

    Examples
    --------
    >>> params = {'RH': (0.5, 1.0)}
    >>> mapping = {'RH': ('RH_exp', 'RH_pred')}
    >>> update_dict_keys(params, mapping, 'pred')
    {'RH_pred': (0.5, 1.0)}
    """
    if selection not in ("exp", "pred"):
        raise ValueError("Selection must be either 'exp' or 'pred'")

    # Determine index based on selection
    index = 0 if selection == "exp" else 1

    new_dict = {}

    for original_key, value in original_dict.items():
        if original_key in parameter_columns_dict:
            new_key = parameter_columns_dict[original_key][index]
            new_dict[new_key] = value
        else:
            # Keep original key if not found in mapping
            new_dict[original_key] = value

    return new_dict


def normalize_coefficients_abs(coef_dict: dict[str, float]) -> dict[str, float]:
    """
    Normalize coefficients by their absolute values.

    Useful for converting ML model coefficients (which can be negative)
    into positive weights that sum to 1.

    Parameters
    ----------
    coef_dict : dict[str, float]
        Dictionary of coefficients (can include negative values).

    Returns
    -------
    dict[str, float]
        Dictionary of normalized positive weights summing to 1.

    Examples
    --------
    >>> normalize_coefficients_abs({'a': -2.0, 'b': 1.0, 'c': 1.0})
    {'a': 0.5, 'b': 0.25, 'c': 0.25}
    """
    abs_coefs = {k: abs(v) for k, v in coef_dict.items()}
    total = sum(abs_coefs.values())
    if total == 0:
        n = len(abs_coefs)
        return {k: 1.0 / n for k in abs_coefs} if n > 0 else {}
    return {k: v / total for k, v in abs_coefs.items()}


def quick_median_thresholds(
    df,
    parameter_columns_dict: dict[str, tuple[str, str]],
    data_type: str = "exp",
) -> dict[str, float]:
    """
    Calculate median-based thresholds for parameters.

    A quick way to establish threshold values based on the
    median of parameter values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the parameter columns.
    parameter_columns_dict : dict[str, tuple[str, str]]
        Dictionary mapping parameter names to (experimental, predicted)
        column name tuples.
    data_type : str, optional
        Either "exp" for experimental or "pred" for predicted
        columns (default: "exp").

    Returns
    -------
    dict[str, float]
        Dictionary of median threshold values for each parameter.

    Examples
    --------
    >>> thresholds = quick_median_thresholds(df, params_dict, 'pred')
    """
    thresholds = {}
    index = 0 if data_type == "exp" else 1

    for param, cols in parameter_columns_dict.items():
        col = cols[index]
        if col in df.columns:
            thresholds[param] = df[col].median()

    return thresholds

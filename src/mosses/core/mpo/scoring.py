"""
Scoring functions for Multi-Parameter Optimization (MPO).

This module provides sigmoid-based scoring functions that transform
raw parameter values into desirability scores (0-1 range).

Functions:
    sigmoid: Standard sigmoid for maximization (higher is better)
    reverse_sigmoid: Reversed sigmoid for minimization (lower is better)
    double_sigmoid: Double sigmoid for middle optimization (optimal range)
    create_scoring_columns: Apply scoring functions to DataFrame columns
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Preference(str, Enum):
    """Optimization preference for parameter scoring."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    MIDDLE = "middle"


@dataclass
class ParameterConfig:
    """
    Configuration for a single parameter in MPO scoring.

    Attributes
    ----------
    name : str
        The parameter name (used as key identifier).
    preference : Preference
        The optimization direction:
        - MAXIMIZE: Higher values are better (scores increase with value)
        - MINIMIZE: Lower values are better (scores decrease with value)
        - MIDDLE: Values in a range are optimal (bell-shaped scoring)
    threshold : float | tuple[float, float]
        The threshold value(s):
        - Single float for MAXIMIZE and MINIMIZE preferences
        - Tuple (lower, upper) for MIDDLE preference
    experimental_column : str | None
        Column name for experimental/observed values.
    predicted_column : str | None
        Column name for predicted values.
    weight : float
        The weight for this parameter in MPO calculation (default: 1.0).
    """

    name: str
    preference: Preference
    threshold: float | tuple[float, float]
    experimental_column: str | None = None
    predicted_column: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        if self.preference == Preference.MIDDLE:
            if not isinstance(self.threshold, tuple) or len(self.threshold) != 2:
                raise ValueError(
                    f"Parameter '{self.name}' with 'middle' preference requires "
                    f"a tuple (lower, upper) threshold, got {type(self.threshold)}"
                )
        elif self.preference in (Preference.MAXIMIZE, Preference.MINIMIZE):
            if isinstance(self.threshold, tuple):
                raise ValueError(
                    f"Parameter '{self.name}' with '{self.preference.value}' preference "
                    f"requires a single threshold value, got tuple"
                )


@dataclass
class ScoringConfig:
    """
    Configuration for MPO scoring across multiple parameters.

    Attributes
    ----------
    parameters : dict[str, ParameterConfig]
        Dictionary mapping parameter names to their configurations.
    steepness : float
        Global steepness parameter for sigmoid functions (default: 2.0).
        Higher values create sharper transitions at thresholds.

    Examples
    --------
    >>> config = ScoringConfig(
    ...     parameters={
    ...         'logP': ParameterConfig(
    ...             name='logP',
    ...             preference=Preference.MIDDLE,
    ...             threshold=(1.0, 3.0),
    ...             weight=1.0
    ...         ),
    ...         'solubility': ParameterConfig(
    ...             name='solubility',
    ...             preference=Preference.MAXIMIZE,
    ...             threshold=50.0,
    ...             weight=2.0
    ...         ),
    ...     },
    ...     steepness=2.0
    ... )
    """

    parameters: dict[str, ParameterConfig] = field(default_factory=dict)
    steepness: float = 2.0

    def add_parameter(self, param: ParameterConfig) -> None:
        """Add a parameter configuration."""
        self.parameters[param.name] = param

    def get_weights(self) -> dict[str, float]:
        """Get normalized weights for all parameters."""
        weights = {name: p.weight for name, p in self.parameters.items()}
        total = sum(weights.values())
        if total == 0:
            return {name: 1.0 / len(weights) for name in weights}
        return {name: w / total for name, w in weights.items()}

    def get_preferences(self) -> dict[str, str]:
        """Get preferences for all parameters."""
        return {name: p.preference.value for name, p in self.parameters.items()}

    def get_thresholds(self) -> dict[str, float | tuple[float, float]]:
        """Get thresholds for all parameters."""
        return {name: p.threshold for name, p in self.parameters.items()}


def sigmoid(
    x: float | np.ndarray,
    threshold: float = 0,
    steepness: float = 2,
) -> float | np.ndarray:
    """
    Standard sigmoid function for maximization scoring.

    Values above the threshold get high scores (approaching 1),
    values below the threshold get low scores (approaching 0).

    Parameters
    ----------
    x : float | np.ndarray
        Input value(s) to score.
    threshold : float, optional
        The inflection point of the sigmoid (default: 0).
        At this value, the score is 0.5.
    steepness : float, optional
        Controls the sharpness of the transition (default: 2).
        Higher values create steeper transitions.

    Returns
    -------
    float | np.ndarray
        Score(s) in range [0, 1].

    Examples
    --------
    >>> sigmoid(5, threshold=3, steepness=2)
    0.9820...
    >>> sigmoid(1, threshold=3, steepness=2)
    0.0179...
    """
    return 1 / (1 + np.exp(-steepness * (x - threshold)))


def reverse_sigmoid(
    x: float | np.ndarray,
    threshold: float = 0,
    steepness: float = 2,
) -> float | np.ndarray:
    """
    Reverse sigmoid function for minimization scoring.

    Values below the threshold get high scores (approaching 1),
    values above the threshold get low scores (approaching 0).

    Parameters
    ----------
    x : float | np.ndarray
        Input value(s) to score.
    threshold : float, optional
        The inflection point of the sigmoid (default: 0).
        At this value, the score is 0.5.
    steepness : float, optional
        Controls the sharpness of the transition (default: 2).
        Higher values create steeper transitions.

    Returns
    -------
    float | np.ndarray
        Score(s) in range [0, 1].

    Examples
    --------
    >>> reverse_sigmoid(1, threshold=3, steepness=2)
    0.9820...
    >>> reverse_sigmoid(5, threshold=3, steepness=2)
    0.0179...
    """
    return 1 / (1 + np.exp(steepness * (x - threshold)))


def double_sigmoid(
    x: float | np.ndarray,
    lower_threshold: float = 0,
    upper_threshold: float = 1,
    steepness: float = 4,
) -> float | np.ndarray:
    """
    Double sigmoid function for middle/range optimization.

    Values between lower_threshold and upper_threshold get high scores
    (approaching 1), values outside this range get lower scores.

    Parameters
    ----------
    x : float | np.ndarray
        Input value(s) to score.
    lower_threshold : float, optional
        The lower bound of the optimal range (default: 0).
    upper_threshold : float, optional
        The upper bound of the optimal range (default: 1).
    steepness : float, optional
        Controls the sharpness of both transitions (default: 4).
        Higher values create steeper transitions at boundaries.

    Returns
    -------
    float | np.ndarray
        Score(s) in range [0, 1].

    Examples
    --------
    >>> double_sigmoid(2.0, lower_threshold=1.0, upper_threshold=3.0)
    0.96...  # In optimal range
    >>> double_sigmoid(0.0, lower_threshold=1.0, upper_threshold=3.0)
    0.01...  # Below range
    """
    left_sigmoid = sigmoid(x, lower_threshold, steepness)
    right_sigmoid = reverse_sigmoid(x, upper_threshold, steepness)
    return left_sigmoid * right_sigmoid


def create_scoring_columns(
    df: pd.DataFrame,
    preferences: dict[str, str],
    thresholds: dict[str, float | tuple[float, float]],
    steepness: float = 2.0,
    label: str = "",
) -> pd.DataFrame:
    """
    Create MPO scoring columns for a DataFrame.

    Transforms raw parameter values into desirability scores (0-1)
    based on the specified preferences and thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with parameter columns to score.
    preferences : dict[str, str]
        Dictionary mapping column names to optimization direction:
        - "maximize": Higher values are better
        - "minimize": Lower values are better
        - "middle": Values in a range are optimal
    thresholds : dict[str, float | tuple[float, float]]
        Dictionary mapping column names to threshold values:
        - Single float for "maximize" and "minimize"
        - Tuple (lower, upper) for "middle"
    steepness : float, optional
        Steepness parameter for sigmoid functions (default: 2.0).
    label : str, optional
        Optional label to append to column names (default: "").

    Returns
    -------
    pd.DataFrame
        DataFrame with scoring columns only (same index as input).

    Raises
    ------
    ValueError
        If preference/threshold combination is invalid.

    Examples
    --------
    >>> df = pd.DataFrame({'logP': [1.0, 2.0, 3.0], 'MW': [200, 300, 400]})
    >>> preferences = {'logP': 'middle', 'MW': 'minimize'}
    >>> thresholds = {'logP': (1.5, 2.5), 'MW': 350}
    >>> scores = create_scoring_columns(df, preferences, thresholds)
    """
    scored_columns = {}

    for param_name in df.columns:
        if param_name not in preferences:
            continue

        preference = preferences[param_name]

        if param_name not in thresholds:
            continue

        threshold_value = thresholds[param_name]

        # Create column name with optional label
        if label:
            column_name = f"{param_name}_score_{label}"
        else:
            column_name = f"{param_name}_score"

        if preference == Preference.MAXIMIZE.value:
            if isinstance(threshold_value, tuple):
                raise ValueError(
                    f"'maximize' preference expects single threshold for "
                    f"'{param_name}', got tuple"
                )
            scored_columns[column_name] = df[param_name].apply(
                lambda x, t=threshold_value, s=steepness: sigmoid(x, threshold=t, steepness=s)
            )

        elif preference == Preference.MINIMIZE.value:
            if isinstance(threshold_value, tuple):
                raise ValueError(
                    f"'minimize' preference expects single threshold for "
                    f"'{param_name}', got tuple"
                )
            scored_columns[column_name] = df[param_name].apply(
                lambda x, t=threshold_value, s=steepness: reverse_sigmoid(
                    x, threshold=t, steepness=s
                )
            )

        elif preference == Preference.MIDDLE.value:
            if not isinstance(threshold_value, tuple) or len(threshold_value) != 2:
                raise ValueError(
                    f"'middle' preference expects tuple (lower, upper) for "
                    f"'{param_name}', got {type(threshold_value)}"
                )
            lower_thresh, upper_thresh = threshold_value
            scored_columns[column_name] = df[param_name].apply(
                lambda x, lt=lower_thresh, ut=upper_thresh, s=steepness: double_sigmoid(
                    x,
                    lower_threshold=lt,
                    upper_threshold=ut,
                    steepness=2 * s,
                )
            )

        else:
            raise ValueError(f"Unknown preference: {preference}")

    return pd.DataFrame(scored_columns, index=df.index)


def compute_mpo_scores(
    df: pd.DataFrame,
    config: ScoringConfig,
    columns: list[str] | None = None,
    use_predicted: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute MPO scores for a DataFrame using the provided configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the parameter columns.
    config : ScoringConfig
        Configuration object with parameter definitions.
    columns : list[str] | None, optional
        Specific columns to use for scoring. If None, uses columns
        from config based on use_predicted flag.
    use_predicted : bool, optional
        If True, use predicted_column; else use experimental_column
        from each ParameterConfig (default: True).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - DataFrame with individual parameter scores
        - Series with combined weighted MPO scores
    """
    # Determine which columns to use
    if columns is None:
        columns = []
        for param in config.parameters.values():
            col = param.predicted_column if use_predicted else param.experimental_column
            if col and col in df.columns:
                columns.append(col)

    # Build preferences and thresholds for the selected columns
    preferences = {}
    thresholds = {}
    weights_list = []

    for col in columns:
        # Find matching parameter config
        for param in config.parameters.values():
            param_col = param.predicted_column if use_predicted else param.experimental_column
            if param_col == col:
                preferences[col] = param.preference.value
                thresholds[col] = param.threshold
                weights_list.append(param.weight)
                break

    # Create scoring columns
    scores_df = create_scoring_columns(
        df[columns],
        preferences,
        thresholds,
        steepness=config.steepness,
    )

    # Normalize weights and compute weighted sum
    weights = np.array(weights_list)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights_list)) / len(weights_list)

    mpo_scores = scores_df.values @ weights

    return scores_df, pd.Series(mpo_scores, index=df.index, name="mpo_score")

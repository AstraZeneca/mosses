"""
Machine Learning-based weight estimators for MPO optimization.

This module provides ML-based approaches to estimate optimal
parameter weights, including:
    - Random Forest Regression
    - Random Forest Classification
    - Logistic Regression
    - Ridge Classification

These approaches treat weight estimation as a supervised learning
problem, using the reference MPO scores as targets.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


@dataclass
class MLEstimatorResult:
    """
    Result of ML-based weight estimation.

    Attributes
    ----------
    weights : dict[str, float]
        Estimated feature importance/weights.
    predictions : np.ndarray
        Model predictions for all samples.
    model : Any
        The trained model/pipeline.
    metrics : dict[str, float]
        Performance metrics (R2, MSE, accuracy, etc.).
    method : str
        Name of the ML method used.
    """

    weights: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    model: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    method: str = ""

    def get_normalized_weights(self) -> dict[str, float]:
        """Get weights normalized to sum to 1."""
        abs_weights = {k: abs(v) for k, v in self.weights.items()}
        total = sum(abs_weights.values())
        if total == 0:
            n = len(abs_weights)
            return {k: 1.0 / n for k in abs_weights} if n > 0 else {}
        return {k: v / total for k, v in abs_weights.items()}


def _check_class_balance(y_train: np.ndarray) -> tuple[float | None, int, int]:
    """
    Check class balance in training data.

    Returns
    -------
    tuple
        (ratio, min_count, max_count)
    """
    counts = Counter(y_train)
    if len(counts) == 0:
        return None, 0, 0
    min_count = min(counts.values())
    max_count = max(counts.values())
    ratio = min_count / max_count if max_count > 0 else None
    return ratio, min_count, max_count


def _apply_smote_if_needed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ratio_threshold: float = 0.25,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE if class imbalance is detected.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    ratio_threshold : float
        Apply SMOTE if minority/majority ratio is below this.
    verbose : bool
        Print information about SMOTE application.

    Returns
    -------
    tuple
        Resampled (X_train, y_train).
    """

    ratio, min_count, max_count = _check_class_balance(y_train)

    if ratio is not None and ratio <= ratio_threshold and min_count > 1:
        if verbose:
            print("Applying SMOTE due to class imbalance.")
        k_neighbors = min(5, min_count - 1)
        smote = SMOTE(k_neighbors=k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train


def rf_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    verbose: bool = False,
) -> MLEstimatorResult:
    """
    Estimate weights using Random Forest Regression.

    Uses feature importances from Random Forest as weights.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    test_size : float, optional
        Fraction for test split (default: 0.2).
    random_state : int, optional
        Random seed (default: 42).
    n_estimators : int, optional
        Number of trees (default: 100).
    max_depth : int, optional
        Maximum tree depth (default: 10).
    min_samples_split : int, optional
        Minimum samples for split (default: 5).
    verbose : bool, optional
        Print results (default: False).

    Returns
    -------
    MLEstimatorResult
        Result containing weights, predictions, and model.
    """
    X = df[feature_cols].values
    y = df[reference_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all = model.predict(X)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    importance_dict = dict(zip(feature_cols, model.feature_importances_))

    if verbose:
        print("RF Features:", importance_dict)
        print("\nModel performance:")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")

    return MLEstimatorResult(
        weights=importance_dict,
        predictions=y_all,
        model=model,
        metrics={
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mse": train_mse,
            "test_mse": test_mse,
        },
        method="rf_regression",
    )


def rf_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    apply_smote: bool = True,
    verbose: bool = False,
) -> MLEstimatorResult:
    """
    Estimate weights using Random Forest Classification.

    Uses feature importances as weights for binary classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column (binary labels).
    test_size : float, optional
        Fraction for test split (default: 0.2).
    random_state : int, optional
        Random seed (default: 42).
    n_estimators : int, optional
        Number of trees (default: 100).
    max_depth : int, optional
        Maximum tree depth (default: 10).
    min_samples_split : int, optional
        Minimum samples for split (default: 5).
    apply_smote : bool, optional
        Apply SMOTE for imbalanced classes (default: True).
    verbose : bool, optional
        Print results (default: False).

    Returns
    -------
    MLEstimatorResult
        Result containing weights, predictions, and model.
    """
    X = df[feature_cols].values
    y = df[reference_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if apply_smote:
        X_train, y_train = _apply_smote_if_needed(X_train, y_train, verbose=verbose)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all = model.predict(X)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    importance_dict = dict(zip(feature_cols, model.feature_importances_))

    if verbose:
        print("Features:", importance_dict)
        print("\nModel performance:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print("\nClassification Report (Test set):")
        print(classification_report(y_test, y_test_pred))

    return MLEstimatorResult(
        weights=importance_dict,
        predictions=y_all,
        model=model,
        metrics={
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1": train_f1,
            "test_f1": test_f1,
        },
        method="rf_classifier",
    )


def logistic_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    apply_smote: bool = True,
    scale_features: bool = True,
    verbose: bool = False,
) -> MLEstimatorResult:
    """
    Estimate weights using Logistic Regression.

    Uses regression coefficients (normalized) as weights.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column (binary labels).
    test_size : float, optional
        Fraction for test split (default: 0.2).
    random_state : int, optional
        Random seed (default: 42).
    max_iter : int, optional
        Maximum iterations (default: 1000).
    apply_smote : bool, optional
        Apply SMOTE for imbalanced classes (default: True).
    scale_features : bool, optional
        Standardize features before fitting (default: True).
    verbose : bool, optional
        Print results (default: False).

    Returns
    -------
    MLEstimatorResult
        Result containing weights, predictions, and model.
    """
    X = df[feature_cols].values
    y = df[reference_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if apply_smote:
        X_train, y_train = _apply_smote_if_needed(X_train, y_train, verbose=verbose)

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X = scaler.transform(X)

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all = model.predict(X)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    try:
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        test_roc_auc = float("nan")

    coef_dict = dict(zip(feature_cols, model.coef_[0]))

    if verbose:
        print("Logistic Regression Coefficients:", coef_dict)
        print("\nModel performance:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Training F1: {train_f1:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
        print("Test Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))

    # Store scaler with model for later use
    model_bundle = {"model": model, "scaler": scaler}

    return MLEstimatorResult(
        weights=coef_dict,
        predictions=y_all,
        model=model_bundle,
        metrics={
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "test_roc_auc": test_roc_auc,
        },
        method="logistic_classifier",
    )


def ridge_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    scale_features: bool = True,
    verbose: bool = False,
) -> MLEstimatorResult:
    """
    Estimate weights using Ridge Classification.

    Uses regression coefficients (normalized) as weights.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column (binary labels).
    alpha : float, optional
        Regularization strength (default: 1.0).
    test_size : float, optional
        Fraction for test split (default: 0.2).
    random_state : int, optional
        Random seed (default: 42).
    apply_smote : bool, optional
        Apply SMOTE for imbalanced classes (default: True).
    scale_features : bool, optional
        Standardize features before fitting (default: True).
    verbose : bool, optional
        Print results (default: False).

    Returns
    -------
    MLEstimatorResult
        Result containing weights, predictions, and model.
    """
    X = df[feature_cols].values
    y = df[reference_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if apply_smote:
        X_train, y_train = _apply_smote_if_needed(X_train, y_train, verbose=verbose)

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X = scaler.transform(X)

    model = RidgeClassifier(
        alpha=alpha,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_all = model.predict(X)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    coefs = model.coef_
    coef_dict = dict(zip(feature_cols, coefs[0] if len(coefs.shape) > 1 else coefs))

    if verbose:
        print("\nRidgeClassifier coefficients:", coef_dict)
        print("\nModel performance:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_test_pred))
        print("Confusion Matrix (Test):")
        print(confusion_matrix(y_test, y_test_pred))

    model_bundle = {"model": model, "scaler": scaler}

    return MLEstimatorResult(
        weights=coef_dict,
        predictions=y_all,
        model=model_bundle,
        metrics={
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        },
        method="ridge_classifier",
    )


# Registry of ML estimators
ML_ESTIMATORS = {
    "rf_regression": rf_regression,
    "rf_classifier": rf_classifier,
    "logistic_classifier": logistic_classifier,
    "ridge_classifier": ridge_classifier,
    # Aliases for backward compatibility
    "RF_reg": rf_regression,
    "RF_class": rf_classifier,
    "log_reg": logistic_classifier,
    "ridge": ridge_classifier,
}


def estimate_weights_ml(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    method: str = "rf_regression",
    verbose: bool = False,
    **kwargs,
) -> MLEstimatorResult:
    """
    Estimate weights using the specified ML method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and reference.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    method : str, optional
        ML method to use (default: "rf_regression").
        Options: "rf_regression", "rf_classifier",
        "logistic_classifier", "ridge_classifier".
    verbose : bool, optional
        Print results (default: False).
    **kwargs
        Additional arguments passed to the estimator.

    Returns
    -------
    MLEstimatorResult
        Result containing weights, predictions, and model.

    Examples
    --------
    >>> result = estimate_weights_ml(
    ...     df, ['score_logP', 'score_MW'], 'ref_class',
    ...     method='rf_classifier'
    ... )
    >>> print(result.get_normalized_weights())
    """
    if method not in ML_ESTIMATORS:
        raise ValueError(
            f"Unknown ML method: {method}. "
            f"Available methods: {list(ML_ESTIMATORS.keys())}"
        )

    estimator = ML_ESTIMATORS[method]
    return estimator(df, feature_cols, reference_col, verbose=verbose, **kwargs)

"""
Weight optimization algorithms for Multi-Parameter Optimization (MPO).

This module provides various optimization algorithms to find optimal
parameter weights that best match a reference MPO score.

Algorithms included:
    - Least squares with penalty
    - Scipy minimize with constraints
    - Dual annealing (global optimization)
    - Differential evolution (genetic algorithm)
    - Powell method
    - PyGAD genetic algorithm (optional)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
import pygad
from scipy.optimize import differential_evolution, dual_annealing, least_squares, minimize


@dataclass
class OptimizationResult:
    """
    Result of weight optimization.

    Attributes
    ----------
    weights : dict[str, float]
        Optimized weights for each feature.
    loss : float
        Final loss value.
    method : str
        Optimization method used.
    success : bool
        Whether optimization converged successfully.
    message : str
        Additional information about the optimization.
    """

    weights: dict[str, float]
    loss: float
    method: str
    success: bool = True
    message: str = ""

    def to_array(self, columns: list[str]) -> np.ndarray:
        """Convert weights to array in column order."""
        return np.array([self.weights.get(col, 0.0) for col in columns])


def objective_sum_squares(
    weights: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
) -> float:
    """
    Compute sum of squared residuals between weighted prediction and reference.

    Parameters
    ----------
    weights : np.ndarray
        Array of weights for each feature.
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.

    Returns
    -------
    float
        Sum of squared residuals (scalar).
    """
    prediction = df[feature_cols].values @ weights
    reference = df[reference_col].values
    residuals = prediction - reference
    return np.sum(residuals**2)


def objective_residuals_penalty(
    weights: np.ndarray,
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    penalty: float = 1000,
) -> np.ndarray:
    """
    Compute residuals with penalty for weights not summing to 1.

    Used with least_squares optimizer which expects residual vector.

    Parameters
    ----------
    weights : np.ndarray
        Array of weights for each feature.
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    penalty : float, optional
        Penalty multiplier for sum constraint (default: 1000).

    Returns
    -------
    np.ndarray
        Array of residuals including the constraint penalty.
    """
    prediction = df[feature_cols].values @ weights
    reference = df[reference_col].values
    residuals = prediction - reference
    sum_constraint_residual = penalty * (np.sum(weights) - 1)
    return np.append(residuals, sum_constraint_residual)


def optimize_weights_least_squares(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    initial_weights: np.ndarray | None = None,
    penalty: float = 1000,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Optimize weights using least squares with penalty constraint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    initial_weights : np.ndarray | None, optional
        Initial weight guess. If None, uses equal weights.
    penalty : float, optional
        Penalty for sum constraint violation (default: 1000).
    verbose : bool, optional
        If True, print optimization results (default: False).

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.
    """
    n_features = len(feature_cols)
    if initial_weights is None:
        initial_weights = np.ones(n_features) / n_features

    bounds = (np.zeros(n_features), np.ones(n_features))

    result = least_squares(
        objective_residuals_penalty,
        initial_weights,
        bounds=bounds,
        args=(df, feature_cols, reference_col, penalty),
    )

    # Normalize weights to sum to 1
    optimal_weights = result.x / result.x.sum() if result.x.sum() > 0 else result.x
    weights_dict = {col: round(w, 3) for col, w in zip(feature_cols, optimal_weights)}

    if verbose:
        print("Optimized Weights:", weights_dict)
        print("Final Loss (with penalty):", round(result.cost, 3))

    return OptimizationResult(
        weights=weights_dict,
        loss=result.cost,
        method="least_squares",
        success=result.success,
        message=result.message,
    )


def optimize_weights_minimize(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    initial_weights: np.ndarray | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Optimize weights using scipy.optimize.minimize with constraints.

    Weights are constrained to [0, 1] and must sum to 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    initial_weights : np.ndarray | None, optional
        Initial weight guess. If None, uses equal weights.
    verbose : bool, optional
        If True, print optimization results (default: False).

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.
    """
    n_features = len(feature_cols)
    if initial_weights is None:
        initial_weights = np.ones(n_features) / n_features

    bounds = [(0, 1)] * n_features
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        objective_sum_squares,
        initial_weights,
        args=(df, feature_cols, reference_col),
        bounds=bounds,
        constraints=constraints,
    )

    weights_dict = {col: round(w, 3) for col, w in zip(feature_cols, result.x)}

    if verbose:
        mse = result.fun / len(df)
        print("Optimized Weights:", weights_dict)
        print("Final Loss (sum of squares):", round(result.fun, 3))
        print("Mean Squared Error:", mse)

    return OptimizationResult(
        weights=weights_dict,
        loss=result.fun,
        method="minimize",
        success=result.success,
        message=result.message if hasattr(result, "message") else "",
    )


def optimize_weights_dual_annealing(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Optimize weights using dual annealing (global optimization).

    Dual annealing is a stochastic global optimization algorithm
    that can escape local minima.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    verbose : bool, optional
        If True, print optimization results (default: False).

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.
    """
    n_features = len(feature_cols)
    bounds = [(0, 1)] * n_features

    result = dual_annealing(
        objective_sum_squares,
        bounds=bounds,
        args=(df, feature_cols, reference_col),
    )

    # Normalize weights to sum to 1
    optimal_weights = result.x / result.x.sum() if result.x.sum() > 0 else result.x
    weights_dict = {col: round(w, 3) for col, w in zip(feature_cols, optimal_weights)}

    if verbose:
        print("Optimized Weights:", weights_dict)
        print("Final Loss (sum of squares):", round(result.fun, 3))

    return OptimizationResult(
        weights=weights_dict,
        loss=result.fun,
        method="dual_annealing",
        success=result.success if hasattr(result, "success") else True,
        message=result.message if hasattr(result, "message") else "",
    )


def optimize_weights_differential_evolution(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Optimize weights using differential evolution (genetic algorithm).

    Differential evolution is a population-based optimization
    algorithm that can find global optima.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    verbose : bool, optional
        If True, print optimization results (default: False).

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.
    """
    n_features = len(feature_cols)
    bounds = [(0, 1)] * n_features

    result = differential_evolution(
        objective_sum_squares,
        bounds=bounds,
        args=(df, feature_cols, reference_col),
    )

    # Normalize weights to sum to 1
    optimal_weights = result.x / result.x.sum() if result.x.sum() > 0 else result.x
    weights_dict = {col: round(w, 3) for col, w in zip(feature_cols, optimal_weights)}

    if verbose:
        print("Optimized Weights:", weights_dict)
        print("Final Loss (sum of squares):", round(result.fun, 3))

    return OptimizationResult(
        weights=weights_dict,
        loss=result.fun,
        method="differential_evolution",
        success=result.success,
        message=result.message,
    )


def optimize_weights_powell(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    initial_weights: np.ndarray | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Optimize weights using Powell's method.

    Powell's method is a derivative-free optimization algorithm
    that uses conjugate directions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    initial_weights : np.ndarray | None, optional
        Initial weight guess. If None, uses equal weights.
    verbose : bool, optional
        If True, print optimization results (default: False).

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.
    """
    n_features = len(feature_cols)
    if initial_weights is None:
        initial_weights = np.ones(n_features) / n_features

    bounds = [(0, 1)] * n_features

    result = minimize(
        objective_sum_squares,
        initial_weights,
        args=(df, feature_cols, reference_col),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 500, "disp": verbose},
    )

    # Normalize weights
    weights_sum = np.sum(result.x)
    if weights_sum == 0:
        optimal_weights = np.ones(n_features) / n_features
    else:
        optimal_weights = result.x / weights_sum

    weights_dict = {col: round(w, 3) for col, w in zip(feature_cols, optimal_weights)}

    if verbose:
        print("Optimized Weights (normalized):", weights_dict)
        print("Final Loss (sum of squares):", round(result.fun, 3))

    return OptimizationResult(
        weights=weights_dict,
        loss=result.fun,
        method="powell",
        success=result.success,
        message=result.message if hasattr(result, "message") else "",
    )


def optimize_weights_pygad(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    num_generations: int = 100,
    population_size: int = 50,
    verbose: bool = False,
) -> OptimizationResult:
    """
    Optimize weights using PyGAD genetic algorithm.

    This is an advanced genetic algorithm implementation that
    can be more effective for complex optimization landscapes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    num_generations : int, optional
        Number of generations to run (default: 100).
    population_size : int, optional
        Size of the population (default: 50).
    verbose : bool, optional
        If True, print optimization progress (default: False).

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.
    """
    n_features = len(feature_cols)
    best_losses = []

    def fitness_func(ga_instance, solution, solution_idx):
        try:
            weights = np.abs(solution)
            if np.sum(weights) == 0:
                weights = np.ones(len(weights))
            weights = weights / np.sum(weights)

            loss = objective_sum_squares(weights, df, feature_cols, reference_col)
            best_losses.append(loss)

            return -loss  # GA maximizes, so negate

        except Exception:
            return -1e6

    def on_generation(ga_instance):
        if verbose and ga_instance.generations_completed % 10 == 0:
            solution, fitness, _ = ga_instance.best_solution()
            weights = solution / np.sum(solution)
            actual_loss = objective_sum_squares(weights, df, feature_cols, reference_col)
            print(f"Generation {ga_instance.generations_completed}: Loss = {actual_loss:.6f}")

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=max(2, population_size // 5),
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=n_features,
        gene_type=float,
        gene_space={"low": 0.01, "high": 1.0},
        mutation_probability=0.15,
        mutation_num_genes=max(1, n_features // 3),
        crossover_probability=0.8,
        parent_selection_type="tournament",
        K_tournament=3,
        on_generation=on_generation if verbose else None,
        stop_criteria=["saturate_20"],
    )

    if verbose:
        print("Starting GA optimization...")

    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    optimal_weights = best_solution / np.sum(best_solution)
    actual_loss = objective_sum_squares(optimal_weights, df, feature_cols, reference_col)

    weights_dict = {col: round(w, 3) for col, w in zip(feature_cols, optimal_weights)}

    if verbose:
        print("\n" + "=" * 50)
        print("GA Optimization Results:")
        print("=" * 50)
        print("Optimized Weights:", weights_dict)
        print(f"Actual Loss: {actual_loss:.6f}")
        print(f"Generations completed: {ga_instance.generations_completed}")

    return OptimizationResult(
        weights=weights_dict,
        loss=actual_loss,
        method="pygad",
        success=True,
        message=f"Completed {ga_instance.generations_completed} generations",
    )


OPTIMIZERS: dict[str, Callable] = {
    "least_squares": optimize_weights_least_squares,
    "minimize": optimize_weights_minimize,
    "dual_annealing": optimize_weights_dual_annealing,
    "differential_evolution": optimize_weights_differential_evolution,
    "powell": optimize_weights_powell,
    "pygad": optimize_weights_pygad,
}


class OptimizerMethod(str, Enum):
    """Available optimization methods for weight optimization."""
    LEAST_SQUARES = "least_squares"
    MINIMIZE = "minimize"
    DUAL_ANNEALING = "dual_annealing"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    POWELL = "powell"
    PYGAD = "pygad"


def optimize_weights(
    df: pd.DataFrame,
    feature_cols: list[str],
    reference_col: str,
    method: OptimizerMethod | str = OptimizerMethod.MINIMIZE,
    verbose: bool = False,
    **kwargs,
) -> OptimizationResult:
    """
    Optimize weights using the specified method.

    This is the main entry point for weight optimization,
    providing a unified interface to all optimization algorithms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature and reference columns.
    feature_cols : list[str]
        List of feature column names.
    reference_col : str
        Name of the reference column.
    method : OptimizerMethod | str, optional
        Optimization method to use (default: OptimizerMethod.MINIMIZE).
        Options: OptimizerMethod.LEAST_SQUARES, OptimizerMethod.MINIMIZE,
        OptimizerMethod.DUAL_ANNEALING, OptimizerMethod.DIFFERENTIAL_EVOLUTION,
        OptimizerMethod.POWELL, OptimizerMethod.PYGAD.
    verbose : bool, optional
        If True, print optimization results (default: False).
    **kwargs
        Additional arguments passed to the specific optimizer.

    Returns
    -------
    OptimizationResult
        Optimization result with weights and loss.

    Raises
    ------
    ValueError
        If the specified method is not recognized.

    Examples
    --------
    >>> result = optimize_weights(
    ...     df, ['score_logP', 'score_MW'], 'ref_mpo',
    ...     method=OptimizerMethod.DUAL_ANNEALING
    ... )
    >>> print(result.weights)
    """
    method_str = method.value if isinstance(method, OptimizerMethod) else method
    
    if method_str not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimization method: {method_str}. "
            f"Available methods: {list(OPTIMIZERS.keys())}"
        )

    optimizer = OPTIMIZERS[method_str]
    return optimizer(df, feature_cols, reference_col, verbose=verbose, **kwargs)

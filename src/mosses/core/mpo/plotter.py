"""
Visualization functions for Multi-Parameter Optimization (MPO).

This module provides plotting functions for MPO analysis, including:
    - Scoring function curves with data overlay
    - MPO score distributions
    - Comparison plots (histograms, Venn diagrams)
    - Mutual information bar charts
    - Correlation heatmaps
    - Best fit scatter plots
    - ROC curves
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, r2_score, roc_curve

from mosses.core.mpo.scoring import double_sigmoid, reverse_sigmoid, sigmoid
from mosses.core.mpo.statistics import find_top_n_percent_ids


def plot_scoring_curves(
    df: pd.DataFrame,
    experimental_columns: list[str],
    score_columns: list[str],
    preferences: dict[str, str],
    thresholds: dict[str, float | tuple[float, float]],
    steepness: float = 2.0,
    figsize: tuple[int, int] = (15, 5),
    show: bool = True,
) -> plt.Figure:
    """
    Plot experimental values vs scores with sigmoid curves overlaid.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental and score columns.
    experimental_columns : list[str]
        List of experimental/input column names.
    score_columns : list[str]
        List of corresponding score column names.
    preferences : dict[str, str]
        Dictionary with parameter names and optimization direction
        ("maximize", "minimize", "middle").
    thresholds : dict[str, float | tuple[float, float]]
        Dictionary with threshold values.
    steepness : float, optional
        Steepness parameter for sigmoid functions (default: 2.0).
    figsize : tuple[int, int], optional
        Base figure size (default: (15, 5)).
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    n_params = len(experimental_columns)

    if n_params == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No parameters to plot", ha="center", va="center")
        if show:
            plt.show()
        return fig

    # Calculate subplot arrangement
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    adjusted_figsize = (figsize[0], figsize[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=adjusted_figsize)

    # Handle different subplot configurations
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1 and n_params > 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for i, (exp_col, score_col) in enumerate(zip(experimental_columns, score_columns)):
        if exp_col not in df.columns or score_col not in df.columns:
            continue

        # Get parameter name
        param_name = (
            exp_col.replace("_experimental", "")
            .replace("_exp", "")
            .replace(" experiment", "")
            .replace(" prediction", "")
        )

        exp_values = df[exp_col]
        scores = df[score_col]

        # Create range for curve
        x_min, x_max = exp_values.min(), exp_values.max()
        x_range = x_max - x_min if x_max != x_min else 1.0
        x_curve = np.linspace(x_min - 0.3 * x_range, x_max + 0.3 * x_range, 200)

        # Calculate sigmoid curve
        preference = preferences.get(param_name, preferences.get(exp_col, "maximize"))
        threshold_value = thresholds.get(param_name, thresholds.get(exp_col))

        if threshold_value is not None:
            if preference == "maximize":
                threshold = (
                    threshold_value[0]
                    if isinstance(threshold_value, tuple)
                    else threshold_value
                )
                y_curve = sigmoid(x_curve, threshold=threshold, steepness=steepness)
                axes[i].axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Threshold: {threshold:.2f}",
                )

            elif preference == "minimize":
                threshold = (
                    threshold_value[0]
                    if isinstance(threshold_value, tuple)
                    else threshold_value
                )
                y_curve = reverse_sigmoid(
                    x_curve, threshold=threshold, steepness=steepness
                )
                axes[i].axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Threshold: {threshold:.2f}",
                )

            elif preference == "middle":
                if isinstance(threshold_value, tuple) and len(threshold_value) == 2:
                    lower_thresh, upper_thresh = threshold_value
                    y_curve = double_sigmoid(
                        x_curve,
                        lower_threshold=lower_thresh,
                        upper_threshold=upper_thresh,
                        steepness=2 * steepness,
                    )
                    axes[i].axvline(
                        lower_thresh,
                        color="red",
                        linestyle="--",
                        alpha=0.8,
                        linewidth=2,
                        label=f"Lower: {lower_thresh:.2f}",
                    )
                    axes[i].axvline(
                        upper_thresh,
                        color="red",
                        linestyle="--",
                        alpha=0.8,
                        linewidth=2,
                        label=f"Upper: {upper_thresh:.2f}",
                    )
                else:
                    continue
            else:
                continue

            # Plot curve
            axes[i].plot(
                x_curve,
                y_curve,
                "r-",
                linewidth=3,
                alpha=0.9,
                label="Scoring Function",
                zorder=2,
            )

        # Plot data points
        scatter = axes[i].scatter(
            exp_values,
            scores,
            alpha=0.8,
            s=80,
            c=scores,
            cmap="viridis",
            edgecolors="black",
            linewidth=1,
            zorder=5,
            label="Data Points",
        )

        # Formatting
        axes[i].set_xlabel(exp_col, fontweight="bold")
        axes[i].set_ylabel("MPO Score", fontweight="bold")
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"{param_name}\n({preference})", fontweight="bold")
        axes[i].legend(loc="best", fontsize=9)

        # Add colorbar
        plt.colorbar(scatter, ax=axes[i], label="Score")

    # Hide empty subplots
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_mpo_histogram(
    mpo_scores: pd.Series | np.ndarray | list,
    title: str = "Distribution of MPO Scores",
    bins: int = 20,
    color: str = "blue",
    show: bool = True,
) -> plt.Figure:
    """
    Plot histogram of MPO scores.

    Parameters
    ----------
    mpo_scores : pd.Series | np.ndarray | list
        MPO scores to plot.
    title : str, optional
        Plot title (default: "Distribution of MPO Scores").
    bins : int, optional
        Number of histogram bins (default: 20).
    color : str, optional
        Bar color (default: "blue").
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(mpo_scores, bins=bins, kde=True, color=color, edgecolor="black", ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("MPO Score", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_comparison(
    df: pd.DataFrame,
    reference_column: str,
    method_column: str,
    percent_top: float = 10.0,
    id_column: str = "Compound Name",
    show: bool = True,
) -> plt.Figure:
    """
    Compare two MPO scores with histograms and Venn diagram.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both score columns.
    reference_column : str
        Name of the reference score column.
    method_column : str
        Name of the method score column.
    percent_top : float, optional
        Percentage for top compound selection (default: 10.0).
    id_column : str, optional
        Column name for compound identifiers (default: "Compound Name").
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Create histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[reference_column], color="skyblue", bins=20, ax=axes[0], kde=True)
    axes[0].set_title(f"Distribution: {reference_column}")
    axes[0].set_xlabel("Score")

    sns.histplot(df[method_column], color="lightgreen", bins=20, ax=axes[1], kde=True)
    axes[1].set_title(f"Distribution: {method_column}")
    axes[1].set_xlabel("Score")

    plt.tight_layout()

    if show:
        plt.show()

    # Create Venn diagram
    ref_top = set(find_top_n_percent_ids(percent_top, df, reference_column, id_column))
    method_top = set(
        find_top_n_percent_ids(percent_top, df, method_column, id_column)
    )

    if not ref_top.isdisjoint(method_top):
        fig_venn, ax_venn = plt.subplots(figsize=(8, 6))
        venn = venn2(
            [ref_top, method_top],
            (f"Top {reference_column}", f"Top {method_column}"),
            ax=ax_venn,
        )

        # Customize colors
        if venn.get_patch_by_id("10"):
            venn.get_patch_by_id("10").set_color("skyblue")
        if venn.get_patch_by_id("01"):
            venn.get_patch_by_id("01").set_color("lightgreen")
        if venn.get_patch_by_id("11"):
            venn.get_patch_by_id("11").set_color("yellow")

        ax_venn.set_title(f"Top {percent_top}% Compound Overlap")

        if show:
            plt.show()
    else:
        print(f"No overlap between top {percent_top}% of both methods.")
        print(f"  {reference_column}: {len(ref_top)} compounds")
        print(f"  {method_column}: {len(method_top)} compounds")

    return fig


def plot_mutual_info(
    mi_df: pd.DataFrame,
    title: str = "Mutual Information",
    color: str = "steelblue",
    show: bool = True,
) -> plt.Figure:
    """
    Plot mutual information scores as bar chart.

    Parameters
    ----------
    mi_df : pd.DataFrame
        DataFrame with 'Feature' and 'Mutual Information' columns.
    title : str, optional
        Plot title (default: "Mutual Information").
    color : str, optional
        Bar color (default: "steelblue").
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Clean labels
    labels = [str(f).split()[0] for f in mi_df["Feature"]]
    bars = ax.bar(
        range(len(mi_df)), mi_df["Mutual Information"], color=color, edgecolor="black"
    )

    # Annotate bars
    for bar, score in zip(bars, mi_df["Mutual Information"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mutual Information Score")
    ax.set_title(title)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    cmap: str = "Blues",
    title: str = "Correlation Matrix",
    show: bool = True,
) -> plt.Figure:
    """
    Plot correlation heatmap for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns.
    columns : list[str]
        Columns to include in correlation matrix.
    cmap : str, optional
        Colormap name (default: "Blues").
    title : str, optional
        Plot title (default: "Correlation Matrix").
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    valid_cols = [c for c in columns if c in df.columns]
    corr_matrix = df[valid_cols].corr()

    labels = [col.split()[0] for col in valid_cols]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title(title)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_best_fit_scatter(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    label: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot scatter with best fit line and R² annotation.

    Parameters
    ----------
    actual : pd.Series | np.ndarray
        Actual/reference values.
    predicted : pd.Series | np.ndarray
        Predicted values.
    label : str | None, optional
        Label for title (default: None).
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    actual_arr = actual.values if hasattr(actual, "values") else np.array(actual)
    pred_arr = predicted if isinstance(predicted, np.ndarray) else np.array(predicted)

    model = LinearRegression()
    actual_reshaped = actual_arr.reshape(-1, 1)
    model.fit(actual_reshaped, pred_arr)
    best_fit = model.predict(actual_reshaped)
    r_squared = r2_score(pred_arr, best_fit)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual_arr, pred_arr, alpha=0.7, color="blue", label=label)
    ax.plot(
        actual_arr,
        best_fit,
        color="red",
        linewidth=2,
        label=f"Best Fit (R² = {r_squared:.2f})",
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    if label:
        ax.set_title(f"Linear Fit: {label}")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_roc_curve(
    actual: np.ndarray | pd.Series,
    predicted_proba: np.ndarray | pd.Series,
    label: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    actual : np.ndarray | pd.Series
        True binary labels.
    predicted_proba : np.ndarray | pd.Series
        Predicted probabilities for positive class.
    label : str | None, optional
        Label for title (default: None).
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fpr, tpr, _ = roc_curve(actual, predicted_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    title = "ROC Curve"
    if label:
        title = f"ROC Curve: {label}"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_id_occurrences(
    df: pd.DataFrame,
    id_column: str,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """
    Plot occurrences of each ID in a list column.

    Useful for analyzing which compounds appear frequently
    in top selections across different methods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the ID column.
    id_column : str
        Name of the column containing ID lists.
    title : str | None, optional
        Plot title (default: auto-generated).
    ax : plt.Axes | None, optional
        Existing axes to plot on (default: None).
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure | None
        Figure object if ax was None.
    """
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")

    # Explode list of IDs
    exploded_ids = df[id_column].explode()
    id_counts = exploded_ids.value_counts()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    id_counts.plot(kind="bar", color="skyblue", ax=ax, edgecolor="black")

    if title is None:
        title = f"Occurrences of IDs in {id_column}"
    ax.set_title(title)
    ax.set_xlabel("Compound ID")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if show and created_fig:
        plt.show()

    return fig if created_fig else None

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
    - Enrichment-style likelihood plots
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib_venn import venn2
from scipy.signal import savgol_filter
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
    color: str = "#2196F3",
    noise_threshold: float | None = None,
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
        Bar color for significant features (default: "#2196F3").
        Features below *noise_threshold* are shown in grey.
    noise_threshold : float | None, optional
        When provided, draws a dashed red 95 %ile noise floor line and
        greys out bars that fall below it.
    show : bool, optional
        Whether to display the plot (default: True).

    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(16, 14))

    labels = [str(f) for f in mi_df["Feature"]]
    mi_values = mi_df["Mutual Information"].values

    # Colour bars: significant (above noise) vs insignificant
    if noise_threshold is not None:
        bar_colors = [
            color if m > noise_threshold else "#BDBDBD" for m in mi_values
        ]
    else:
        bar_colors = color

    bars = ax.bar(
        range(len(mi_df)),
        mi_values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Noise floor line
    if noise_threshold is not None:
        ax.axhline(
            noise_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.text(
            0.98,
            noise_threshold,
            f" 95% noise = {noise_threshold:.2f}",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="right",
            fontsize=35,
            color="red",
            fontstyle="italic",
        )

    # Annotate bars
    for bar, score in zip(bars, mi_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=30,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mutual Information", fontsize=35, labelpad=12)
    ax.set_title(title, fontsize=40, pad=12)
    ax.tick_params(labelsize=35)

    # Add headroom so bar labels and title don't overlap
    y_max = mi_values.max() if len(mi_values) > 0 else 1.0
    ax.set_ylim(top=y_max * 1.25)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def compute_vertical_threshold(
    in_vitro: np.ndarray | pd.Series,
    in_silico: np.ndarray | pd.Series,
    horizontal_threshold: float,
    ratio_top_compounds: float = 90.0,
) -> float | None:
    """
    Calculate the in-silico MPO threshold that captures a given
    percentage of compounds above the in-vitro MPO goal.

    Parameters
    ----------
    in_vitro : array-like
        In-vitro (Goal) MPO values.
    in_silico : array-like
        In-silico MPO values.
    horizontal_threshold : float
        The in-vitro MPO score threshold (green horizontal line).
    ratio_top_compounds : float
        Percentage of "good" compounds to capture (default 90%).
        The vertical threshold is at the
        ``(100 - ratio_top_compounds)``-th percentile of in-silico
        scores among compounds above the horizontal threshold.

    Returns
    -------
    float | None
        The in-silico MPO threshold, or ``None`` if no compounds
        exceed the horizontal threshold.
    """
    in_vitro = np.asarray(in_vitro)
    in_silico = np.asarray(in_silico)
    mask = in_vitro >= horizontal_threshold
    if mask.sum() == 0:
        return None
    x_above = in_silico[mask]
    percentile = 100.0 - ratio_top_compounds
    return float(np.percentile(x_above, percentile))


def plot_mpo_scatter_with_thresholds(
    in_vitro: np.ndarray | pd.Series,
    in_silico: np.ndarray | pd.Series,
    horizontal_threshold: float | None = None,
    vertical_threshold: float | None = None,
    title: str = "In silico MPO vs In vitro MPO",
    xlabel: str = "In silico MPO",
    ylabel: str = "In vitro MPO",
    figsize: tuple[int, int] = (9, 7),
    vertical_threshold_label: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Scatter plot of in-silico vs in-vitro MPO with threshold lines.

    Axes
    ----
    X-axis = in-silico MPO (``in_silico``)
    Y-axis = in-vitro / goal MPO (``in_vitro``)

    Parameters
    ----------
    in_vitro : array-like
        Y-axis values (in-vitro / goal MPO).
    in_silico : array-like
        X-axis values (in-silico MPO score).
    horizontal_threshold : float | None
        Green horizontal line at the in-vitro MPO goal score (y-axis).
    vertical_threshold : float | None
        Pink vertical line at the recommended in-silico MPO cutoff (x-axis).
    title, xlabel, ylabel : str
        Plot labels.
    figsize : tuple[int, int]
        Figure size (default ``(9, 8)``).
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    in_vitro = np.asarray(in_vitro, dtype=float)
    in_silico = np.asarray(in_silico, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # X = in_silico, Y = in_vitro
    # Colour points by quadrant
    if horizontal_threshold is not None and vertical_threshold is not None:
        above_h = in_vitro >= horizontal_threshold   # above goal (y-axis)
        above_v = in_silico >= vertical_threshold    # right of threshold (x-axis)
        colors = np.where(
            above_h & above_v, "#4CAF50",        # upper-right → green  (true positive)
            np.where(
                ~above_h & ~above_v, "#BDBDBD",  # lower-left  → grey   (true negative)
                np.where(
                    above_h & ~above_v, "#FF9800",  # upper-left  → orange (missed)
                    "#2196F3",                      # lower-right → blue   (false positive)
                ),
            ),
        )
        ax.scatter(in_silico, in_vitro, c=colors, alpha=0.7, s=40,
                   edgecolors="white", linewidth=0.3, zorder=5)
    else:
        ax.scatter(in_silico, in_vitro, alpha=0.7, s=40, color="#2196F3",
                   edgecolors="white", linewidth=0.3, zorder=5)

    # --- Linear regression trend line ---
    finite_mask = np.isfinite(in_silico) & np.isfinite(in_vitro)
    if finite_mask.sum() >= 2:
        slope, intercept = np.polyfit(in_silico[finite_mask], in_vitro[finite_mask], 1)
        x_range = np.linspace(np.nanmin(in_silico), np.nanmax(in_silico), 200)
        ax.plot(x_range, slope * x_range + intercept, color="red",
                linewidth=1.5, alpha=0.8, zorder=4, label="Linear regression")

    # --- Threshold lines ---
    if horizontal_threshold is not None:
        ax.axhline(horizontal_threshold, color="green", linestyle="--",
                   linewidth=2, alpha=0.8, label=f"Goal MPO = {horizontal_threshold:.1f}")
    if vertical_threshold is not None:
        ax.axvline(vertical_threshold, color="#E91E63", linestyle="--",
                   linewidth=2, alpha=0.8,
                   label=(
                       vertical_threshold_label
                       if vertical_threshold_label is not None
                       else f"In silico threshold = {vertical_threshold:.3f}"
                   ))

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def compute_ppv_for_curve(
    in_vitro: np.ndarray | pd.Series,
    in_silico: np.ndarray | pd.Series,
    horizontal_threshold: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PPV and FOR as functions of the in-silico MPO threshold.

    For each candidate threshold *t* along the in-silico score range:

    - **Positive prediction**: in_silico >= t
    - **True positive**: in_vitro >= horizontal_threshold

    PPV = TP / (TP + FP)   — precision of the positive prediction
    FOR = FN / (FN + TN)   — false omission rate among negatives

    Parameters
    ----------
    in_vitro : array-like
        In-vitro (goal) MPO values.
    in_silico : array-like
        In-silico MPO values.
    horizontal_threshold : float
        In-vitro MPO goal score.
    n_points : int
        Number of threshold points to evaluate.

    Returns
    -------
    thresholds : np.ndarray
        Array of in-silico threshold values.
    ppv : np.ndarray
        PPV at each threshold.
    for_rate : np.ndarray
        FOR at each threshold.
    """
    in_vitro = np.asarray(in_vitro, dtype=float)
    in_silico = np.asarray(in_silico, dtype=float)
    actual_positive = in_vitro >= horizontal_threshold

    lo = np.nanmin(in_silico)
    hi = np.nanmax(in_silico)
    if lo == hi:
        return np.array([lo]), np.array([np.nan]), np.array([np.nan])

    thresholds = np.linspace(lo, hi, n_points)
    ppv = np.full(n_points, np.nan)
    for_rate = np.full(n_points, np.nan)

    for i, t in enumerate(thresholds):
        pred_pos = in_silico >= t
        pred_neg = ~pred_pos
        tp = (pred_pos & actual_positive).sum()
        fp = (pred_pos & ~actual_positive).sum()
        fn = (pred_neg & actual_positive).sum()
        tn = (pred_neg & ~actual_positive).sum()

        if tp + fp > 0:
            ppv[i] = tp / (tp + fp)
        if fn + tn > 0:
            for_rate[i] = fn / (fn + tn)

    return thresholds, ppv, for_rate


def plot_ppv_for(
    thresholds: np.ndarray,
    ppv: np.ndarray,
    for_rate: np.ndarray,
    vertical_threshold: float | None = None,
    title: str = "PPV & FOR vs In silico MPO threshold",
    figsize: tuple[int, int] = (11, 9),
    show: bool = True,
) -> plt.Figure:
    """
    Plot PPV and FOR curves as a function of in-silico MPO threshold.

    Parameters
    ----------
    thresholds : np.ndarray
        In-silico MPO threshold values (x-axis).
    ppv : np.ndarray
        Positive Predictive Value at each threshold.
    for_rate : np.ndarray
        False Omission Rate at each threshold.
    vertical_threshold : float | None
        If provided, draws a vertical dashed line at this threshold.
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(thresholds, ppv * 100, color="blue", linewidth=2, label="PPV", marker="", zorder=3)
    ax.plot(thresholds, for_rate * 100, color="orange", linewidth=2, label="FOR", marker="", zorder=3)

    ax.fill_between(thresholds, ppv * 100, alpha=0.10, color="blue")
    ax.fill_between(thresholds, for_rate * 100, alpha=0.10, color="orange")

    if vertical_threshold is not None:
        ax.axvline(vertical_threshold, color="#E91E63", linestyle="--",
                   linewidth=2, alpha=0.8,
                   label=f"Threshold = {vertical_threshold:.3f}")

    ax.set_xlabel("User defined in silico MPO, threshold", fontsize=20)
    ax.set_ylabel("Likelihood (%)", fontsize=20)
    ax.set_title(title, fontsize=16, pad=10, wrap=True)
    ax.tick_params(labelsize=15)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ====================================================================
#  Enrichment-style likelihood plot  (PPV, FOR, CI, arrows,
#  % compounds tested on secondary axis)
# ====================================================================


@dataclass
class MpoLikelihoodMetrics:
    """Precomputed metrics for the MPO enrichment-style likelihood plot."""

    filtered_ppv: np.ndarray
    filtered_for: np.ndarray
    compounds_tested: np.ndarray
    ppv_ci_lower: np.ndarray
    ppv_ci_upper: np.ndarray
    for_ci_lower: np.ndarray
    for_ci_upper: np.ndarray
    arrow: tuple[float, float, float, float]
    desired_ppv: float | str
    desired_for: float | str


def compute_mpo_likelihood_metrics(
    in_vitro: np.ndarray | pd.Series,
    in_silico: np.ndarray | pd.Series,
    horizontal_threshold: float,
    vertical_threshold: float | None = None,
    n_points: int = 200,
) -> tuple[np.ndarray, MpoLikelihoodMetrics]:
    """Compute enrichment-style likelihood metrics for MPO analysis.

    For each candidate threshold *t* along the in-silico score range
    this function computes PPV, FOR, and the cumulative percentage of
    compounds that would be tested.  The raw curves are smoothed with
    a Savitzky-Golay filter, and 95 % confidence bands are estimated
    using the standard-error approach (same methodology as the model
    evaluation tab).

    Parameters
    ----------
    in_vitro : array-like
        In-vitro (Goal) MPO values.
    in_silico : array-like
        In-silico MPO values.
    horizontal_threshold : float
        In-vitro MPO goal score.
    vertical_threshold : float | None
        Selected in-silico threshold (for the pink arrow).
    n_points : int
        Number of threshold points to evaluate.

    Returns
    -------
    thresholds : np.ndarray
        Array of in-silico threshold values.
    metrics : MpoLikelihoodMetrics
        Precomputed curves, CI bounds, arrow data.
    """
    in_vitro = np.asarray(in_vitro, dtype=float)
    in_silico = np.asarray(in_silico, dtype=float)
    actual_positive = in_vitro >= horizontal_threshold
    n_total = len(in_silico)

    lo, hi = float(np.nanmin(in_silico)), float(np.nanmax(in_silico))
    if lo == hi:
        return (
            np.array([lo]),
            MpoLikelihoodMetrics(
                filtered_ppv=np.array([np.nan]),
                filtered_for=np.array([np.nan]),
                compounds_tested=np.array([100.0]),
                ppv_ci_lower=np.array([np.nan]),
                ppv_ci_upper=np.array([np.nan]),
                for_ci_lower=np.array([np.nan]),
                for_ci_upper=np.array([np.nan]),
                arrow=(-100.0, -100.0, -100.0, -100.0),
                desired_ppv="N/A",
                desired_for="N/A",
            ),
        )

    thresholds = np.linspace(lo, hi, n_points)
    ppv = np.full(n_points, np.nan)
    for_rate = np.full(n_points, np.nan)
    compounds_tested = np.full(n_points, np.nan)

    for i, t in enumerate(thresholds):
        pred_pos = in_silico >= t
        pred_neg = ~pred_pos
        tp = (pred_pos & actual_positive).sum()
        fp = (pred_pos & ~actual_positive).sum()
        fn = (pred_neg & actual_positive).sum()
        tn = (pred_neg & ~actual_positive).sum()

        if tp + fp > 0:
            ppv[i] = tp / (tp + fp) * 100.0
        if fn + tn > 0:
            for_rate[i] = fn / (fn + tn) * 100.0
        compounds_tested[i] = pred_pos.sum() / n_total * 100.0

    # Replace NaN / inf before smoothing
    ppv = np.nan_to_num(ppv, nan=0.0, posinf=100.0, neginf=0.0)
    for_rate = np.nan_to_num(for_rate, nan=0.0, posinf=100.0, neginf=0.0)
    compounds_tested = np.nan_to_num(
        compounds_tested, nan=0.0, posinf=100.0, neginf=0.0,
    )

    # Savitzky-Golay smoothing
    if len(thresholds) >= 3:
        ppv_s = savgol_filter(ppv, window_length=3, polyorder=2)
        for_s = savgol_filter(for_rate, window_length=3, polyorder=2)
        ct_s = savgol_filter(compounds_tested, window_length=3, polyorder=2)
    else:
        ppv_s, for_s, ct_s = ppv, for_rate, compounds_tested

    # 95 % CI (SE-based, same approach as model-eval tab)
    se_ppv = np.nanstd(ppv_s) / np.sqrt(len(ppv_s))
    se_for = np.nanstd(for_s) / np.sqrt(len(for_s))
    ci_ppv_upper = ppv_s + 1.96 * se_ppv
    ci_ppv_lower = ppv_s - 1.96 * se_ppv
    ci_for_upper = for_s + 1.96 * se_for
    ci_for_lower = for_s - 1.96 * se_for

    # Best threshold (max conservative gap: ci_ppv_lower − ci_for_upper)
    distances = ci_ppv_lower - ci_for_upper
    valid = np.isfinite(distances)
    if valid.any():
        idx = int(np.argmax(np.where(valid, distances, -np.inf)))
        arrow = (
            float(distances[idx]),
            float(thresholds[idx]),
            float(ppv_s[idx]),
            float(for_s[idx]),
        )
    else:
        arrow = (-100.0, -100.0, -100.0, -100.0)

    # PPV / FOR at the user-selected threshold
    desired_ppv: float | str = "N/A"
    desired_for: float | str = "N/A"
    if vertical_threshold is not None:
        idx_sel = int(np.argmin(np.abs(thresholds - vertical_threshold)))
        desired_ppv = float(ppv_s[idx_sel])
        desired_for = float(for_s[idx_sel])

    metrics = MpoLikelihoodMetrics(
        filtered_ppv=ppv_s,
        filtered_for=for_s,
        compounds_tested=ct_s,
        ppv_ci_lower=ci_ppv_lower,
        ppv_ci_upper=ci_ppv_upper,
        for_ci_lower=ci_for_lower,
        for_ci_upper=ci_for_upper,
        arrow=arrow,
        desired_ppv=desired_ppv,
        desired_for=desired_for,
    )
    return thresholds, metrics


def plot_mpo_likelihood(
    thresholds: np.ndarray,
    metrics: MpoLikelihoodMetrics,
    vertical_threshold: float | None = None,
    title: str = "Enrichment Plot",
    figsize: tuple[int, int] = (11, 9),
    selected_threshold_label: str | None = None,
    xlabel: str = "User defined in silico MPO, threshold",
    show: bool = True,
) -> plt.Figure:
    """Plot PPV & FOR enrichment-style likelihood curves for MPO.

    Replicates the visual style of the model-evaluation tab's
    enrichment plot: turquoise PPV with CI band, indigo FOR with CI
    band, grey ``% compounds tested`` on a secondary y-axis, and
    double-headed arrows indicating the best threshold (green) and
    the user-selected threshold (pink).

    Parameters
    ----------
    thresholds : np.ndarray
        In-silico MPO threshold values (x-axis).
    metrics : MpoLikelihoodMetrics
        Precomputed enrichment metrics.
    vertical_threshold : float | None
        Selected in-silico threshold (pink arrow position).
    title : str
        Plot title.
    figsize : tuple[int, int]
        Figure size.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.header_visible = False

    # --- PPV curve + CI band ---
    ax.plot(
        thresholds, metrics.filtered_ppv,
        color="turquoise", linewidth=2, zorder=3,
    )
    ax.fill_between(
        thresholds, metrics.ppv_ci_lower, metrics.ppv_ci_upper,
        color="turquoise", alpha=0.2,
    )

    # --- FOR curve + CI band ---
    ax.plot(
        thresholds, metrics.filtered_for,
        color="indigo", linewidth=2, zorder=3,
    )
    ax.fill_between(
        thresholds, metrics.for_ci_lower, metrics.for_ci_upper,
        color="indigo", alpha=0.2,
    )

    # --- Arrows ---
    if metrics.filtered_ppv.size and metrics.filtered_for.size:
        _, max_thresh, max_ppv, max_for = metrics.arrow
        if max_thresh != -100:
            ax.annotate(
                text="",
                xy=(max_thresh, max_for),
                xytext=(max_thresh, max_ppv),
                arrowprops=dict(arrowstyle="<->", color="green", lw=2),
            )
        if (
            vertical_threshold is not None
            and metrics.desired_ppv != "N/A"
            and metrics.desired_for != "N/A"
        ):
            ax.annotate(
                text="",
                xy=(vertical_threshold, metrics.desired_for),
                xytext=(vertical_threshold, metrics.desired_ppv),
                arrowprops=dict(arrowstyle="<->", color="plum", lw=2),
            )

    # --- Secondary y-axis: % compounds tested ---
    ax2 = ax.twinx()
    ax2.plot(
        thresholds, metrics.compounds_tested,
        color="grey", linewidth=2, zorder=2,
    )

    # --- Labels & formatting ---
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("PPV & FOR (%)", fontsize=14)
    ax2.set_ylabel("% of compounds tested", fontsize=14)
    ax.set_title(title, fontsize=16, pad=8)
    ax.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)

    # --- Legend (below plot) ---
    handles = [
        Line2D([], [], color="turquoise", linewidth=2),
        Line2D([], [], color="indigo", linewidth=2),
        Line2D([], [], color="grey", linewidth=2),
    ]
    labels = [
        "PPV – Likelihood to extract good compounds",
        "FOR – Likelihood to discard good compounds",
        "% of compounds tested (cumulative)",
    ]
    if metrics.arrow[1] != -100:
        handles.append(
            Line2D([], [], color="green", linewidth=2),
        )
        labels.append("Highest Predictive Balance (PPV-FOR)")
    if vertical_threshold is not None:
        handles.append(
            Line2D([], [], color="plum", linewidth=2),
        )
        labels.append(
            selected_threshold_label
            if selected_threshold_label is not None
            else f"Selected threshold = {vertical_threshold:.3f}"
        )

    ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        fontsize=11,
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    if show:
        plt.show()
    return fig


def plot_parameter_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    parameter_names: list[str] | None = None,
    cmap: str = "RdBu_r",
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (8, 7),
    show: bool = True,
) -> plt.Figure:
    """
    Plot a correlation heatmap for a set of parameter columns.

    This is the **generic** correlation-matrix plotter for MPO analysis.
    It accepts any list of columns (experimental *or* predicted) and
    renders an annotated Seaborn heatmap with optional clean parameter
    names as axis labels.

    Use this function directly when you have a custom set of columns,
    or call the thin convenience wrappers
    :func:`plot_experimental_correlation_matrix` /
    :func:`plot_predicted_correlation_matrix` which simply supply
    sensible default titles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data columns.
    columns : list[str]
        Column names to include in the correlation matrix.
        Columns not present in *df* are silently skipped.
    parameter_names : list[str] | None, optional
        Human-readable display labels for the parameters (same order
        as *columns*).  When ``None``, labels are derived by stripping
        common suffixes (``_exp``, ``_pred``, …) via
        ``_clean_column_labels``.
    cmap : str, optional
        Matplotlib / Seaborn colormap name (default: ``"RdBu_r"``).
    title : str, optional
        Plot title (default: ``"Correlation Matrix"``).
    figsize : tuple[int, int], optional
        Figure size in inches (default: ``(8, 7)``).
    show : bool, optional
        Whether to call ``plt.show()`` (default: ``True``).
        Set to ``False`` when embedding the figure in a notebook widget.

    Returns
    -------
    plt.Figure
        The matplotlib figure object.

    Examples
    --------
    Experimental parameters:

    >>> fig = plot_parameter_correlation_matrix(
    ...     df,
    ...     columns=["logD_exp", "sol_exp", "CL_exp"],
    ...     parameter_names=["LogD", "Solubility", "Clearance"],
    ...     title="Correlation Matrix of Experimental Parameters",
    ... )

    Predicted (in-silico) parameters:

    >>> fig = plot_parameter_correlation_matrix(
    ...     df,
    ...     columns=["logD_pred", "sol_pred", "CL_pred"],
    ...     parameter_names=["LogD", "Solubility", "Clearance"],
    ...     title="Correlation Matrix of Predicted Parameters",
    ... )

    Mixed / custom selection:

    >>> fig = plot_parameter_correlation_matrix(
    ...     df,
    ...     columns=["LogD", "Solubility", "Clearance", "Activity"],
    ...     title="All Parameters Correlation",
    ... )
    """
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5,
            "No valid columns found",
            ha="center", va="center", fontsize=12, color="#888",
        )
        ax.set_axis_off()
        if show:
            plt.show()
        return fig

    corr_matrix = df[valid_cols].corr()

    # Build labels
    if parameter_names is not None and len(parameter_names) == len(valid_cols):
        labels = list(parameter_names)
    else:
        labels = _clean_column_labels(valid_cols)

    corr_display = corr_matrix.copy()
    corr_display.index = labels
    corr_display.columns = labels

    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(
        corr_display,
        annot=True,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 19},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=18, pad=12)

    # Increase colorbar tick font size
    cbar = heatmap.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_experimental_correlation_matrix(
    df: pd.DataFrame,
    experimental_columns: list[str],
    parameter_names: list[str] | None = None,
    cmap: str = "RdBu_r",
    title: str = "Correlation Matrix of Experimental Parameters",
    figsize: tuple[int, int] = (8, 7),
    show: bool = True,
) -> plt.Figure:
    """
    Convenience wrapper for :func:`plot_parameter_correlation_matrix`
    with a default title suited for **experimental** (observed) data.

    See :func:`plot_parameter_correlation_matrix` for full parameter
    documentation.
    """
    return plot_parameter_correlation_matrix(
        df,
        columns=experimental_columns,
        parameter_names=parameter_names,
        cmap=cmap,
        title=title,
        figsize=figsize,
        show=show,
    )


def plot_predicted_correlation_matrix(
    df: pd.DataFrame,
    predicted_columns: list[str],
    parameter_names: list[str] | None = None,
    cmap: str = "RdBu_r",
    title: str = "Correlation Matrix of Predicted Parameters",
    figsize: tuple[int, int] = (8, 7),
    show: bool = True,
) -> plt.Figure:
    """
    Convenience wrapper for :func:`plot_parameter_correlation_matrix`
    with a default title suited for **predicted** (in-silico) data.

    See :func:`plot_parameter_correlation_matrix` for full parameter
    documentation.
    """
    return plot_parameter_correlation_matrix(
        df,
        columns=predicted_columns,
        parameter_names=parameter_names,
        cmap=cmap,
        title=title,
        figsize=figsize,
        show=show,
    )


def _clean_column_labels(columns: list[str]) -> list[str]:
    """
    Clean column names into readable labels.

    Strips common suffixes like '_experimental', '_exp', '_predicted',
    '_pred', ' experiment', ' prediction' and takes the first word
    as a fallback.

    Parameters
    ----------
    columns : list[str]
        Raw column names.

    Returns
    -------
    list[str]
        Cleaned labels.
    """
    suffixes = [
        "_experimental", "_exp", "_predicted", "_pred",
        " experimental", " experiment", " prediction", " predicted",
    ]
    labels = []
    for col in columns:
        label = col
        for suffix in suffixes:
            if label.lower().endswith(suffix):
                label = label[: -len(suffix)]
                break
        labels.append(label.strip())
    return labels


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

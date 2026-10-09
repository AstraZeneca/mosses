import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from mosses.core.metrics import (
    LikelihoodMetrics,
    LinePlotMetrics,
    _resolve_ops,
    apply_operation,
    needs_log_axis,
)

logging.getLogger("matplotlib.category").setLevel(logging.WARNING)


class Plotter:
    def __init__(self, scale: str = "log", op_exp: str | None = None, op_pred: str | None = None, threshold_transformed: bool = False) -> None:
        """
        Initialize the Plotter.

        Parameters
        ----------
        scale : str, optional
            Scale to use for the plots ('log' or 'linear'), by default 'log'.
        op_exp : str, optional
            Operation on experimental column ("None", "Log", "Negative Log", "Negative").
        op_pred : str, optional
            Operation on predicted column.
        threshold_transformed : bool, optional
            If True, threshold is already in transformed space (assay evaluation).
        """
        self.op_exp, self.op_pred = _resolve_ops(scale, op_exp, op_pred)
        self.scale = "log" if (needs_log_axis(self.op_exp) or needs_log_axis(self.op_pred)) else "linear"
        self.threshold_transformed = threshold_transformed

    @staticmethod
    def reset_y_ticks(
        ax: matplotlib.axes.Axes,
    ) -> matplotlib.axes.Axes:
        """
        Reset y-axis ticks to integer labels and set a fixed range (0 to 100).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object on which to reset the y-axis ticks.

        Returns
        -------
        matplotlib.axes.Axes
            The modified Axes object with updated y-axis ticks and labels.
        """
        ax.set_ylim(-10, 110)
        y_ticks = ax.get_yticks()
        new_y_ticks = [int(y) for y in y_ticks]
        ax.set_yticks(new_y_ticks)
        ax.set_yticklabels(new_y_ticks, rotation=45)
        return ax

    @staticmethod
    def reset_x_ticks(
        threshold: np.ndarray,
        ax: matplotlib.axes.Axes,
    ) -> matplotlib.axes.Axes:
        """
        Reset x-axis ticks based on the log-transformed threshold values.

        This function computes new tick positions for
        the x-axis based on the logarithm (base 10)
        of the threshold values and adjusts the tick labels accordingly.

        Parameters
        ----------
        threshold : np.ndarray
            An array of threshold values.
        ax : matplotlib.axes.Axes
            The Axes object on which to reset the ticks.

        Returns
        -------
        matplotlib.axes.Axes
            The modified Axes object with updated x-axis ticks and labels.
        """
        xlims = np.arange(
            min(np.log10(threshold)) - 0.5,
            max(np.log10(threshold)) + 0.5,
            0.5,
        )
        ax.set_xlim(
            min(np.log10(threshold)) - 0.5,
            max(np.log10(threshold)) + 0.5,
        )
        ax.xaxis.set_ticks(xlims)
        x_ticks = ax.get_xticks()
        if (x_ticks < 0).any():
            new_x_ticks = [round((10**x), 2) for x in x_ticks]
        else:
            new_x_ticks = [int(10**x) for x in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(new_x_ticks, rotation=45)
        return ax

    def _plot_log_exp_values_dist(
        self,
        agg_df: pd.DataFrame,
        df: pd.DataFrame,
        desired_threshold: float,
        plot_title: str,
    ) -> None:
        """
        Plot experimental values distribution over time
        using a logarithmic scale.

        Parameters
        ----------
        agg_df : pd.DataFrame
            Aggregated DataFrame with columns 'month_year' and 'median_exp'.
        df : pd.DataFrame
            Original DataFrame used to compute overall y-axis
            limits from 'observed'.
        desired_threshold : float
            The experimental threshold to display
            (will be plotted as a horizontal line).
        plot_title : str
            Title for the plot.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.canvas.header_visible = False
        ax.plot(
            agg_df["month_year"],
            apply_operation(agg_df["median_exp"].values, self.op_exp),
            marker="o",
            color="dodgerblue",
        )

        observed_log = apply_operation(df["observed"].values, self.op_exp)
        y_min = observed_log.min() - 0.5
        y_max = observed_log.max() + 0.5
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylim(y_min, y_max)

        ax.set_xticklabels(agg_df["month_year"], rotation=90, fontsize=8)

        # Convert log-space ticks back to original scale for display
        # (skip for assay evaluation where axes show transformed space)
        if not self.threshold_transformed:
            y_ticks = ax.get_yticks()
            ax.set_yticklabels([int(10**y) for y in y_ticks], fontsize=8)
        _thresh_y = desired_threshold if self.threshold_transformed else apply_operation(np.array([desired_threshold]), self.op_exp)[0]
        ax.axhline(
            y=_thresh_y,
            color="orangered",
            linestyle="dotted",
        )

        ax.set_xlabel("Sample Registration Date", fontweight="bold")
        ax.set_ylabel("Experimental values", fontweight="bold")

        ax.set_title(f"{plot_title} - Experimental values per month")
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        handles = [
            Line2D([], [], color="dodgerblue"),
            Line2D([], [], color="orange", linestyle=":"),
        ]
        ax.legend(
            handles,
            [
                "Median experimental values during each time period",
                "Desired project threshold",
            ],
            bbox_to_anchor=(0.5, -0.3),
            loc="upper center",
            fontsize=7,
        )
        fig.tight_layout()
        plt.show()

    def _plot_linear_exp_values_dist(
        self,
        agg_df: pd.DataFrame,
        df: pd.DataFrame,
        desired_threshold: float,
        plot_title: str,
    ) -> None:
        """
        Plot experimental values distribution over time using a linear scale.

        Parameters
        ----------
        agg_df : pd.DataFrame
            Aggregated DataFrame with columns 'month_year' and 'median_exp'.
        df : pd.DataFrame
            Original DataFrame used to compute
            overall y-axis limits from 'observed'.
        desired_threshold : float
            The experimental threshold to display (will be plotted
            as a horizontal line).
        plot_title : str
            Title for the plot.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.canvas.header_visible = False

        # For assay evaluation, plot in the configured transformed space.
        if self.threshold_transformed:
            median_vals = apply_operation(agg_df["median_exp"].values, self.op_exp)
            obs_vals = apply_operation(df["observed"].values, self.op_exp)
        else:
            median_vals = agg_df["median_exp"].values
            obs_vals = df["observed"].values

        inc = (obs_vals.max() - obs_vals.min()) / 5
        y_min = obs_vals.min() - inc
        y_max = obs_vals.max() + inc
        ax.plot(
            agg_df["month_year"], median_vals, marker="o", color="dodgerblue"
        )
        ax.set_xlabel("Sample Registration Date", fontweight="bold")
        ax.set_ylabel("Experimental values - Median", fontweight="bold")

        ax.set_xticklabels(agg_df["month_year"], rotation=90)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=desired_threshold, color="orangered", linestyle="dotted")

        ax.set_title(f"{plot_title} - Experimental values over time")
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        handles = [
            Line2D([], [], color="dodgerblue"),
            Line2D([], [], color="orange", linestyle=":"),
        ]
        ax.legend(
            handles,
            [
                "Median experimental values during each time period",
                "Desired project threshold",
            ],
            bbox_to_anchor=(0.5, -0.3),
            loc="upper center",
            fontsize=7,
        )
        fig.tight_layout()
        plt.show()

    def draw_exp_values_dist(
        self,
        agg_df: pd.DataFrame,
        df: pd.DataFrame,
        desired_threshold: float,
        plot_title: str,
    ) -> None:
        """
        Plot the distribution of experimental values
        over time based on aggregated data.

        Parameters
        ----------
        agg_df : pd.DataFrame
            Aggregated data with columns:
              - 'month_year'
              - 'median_exp'
        df : pd.DataFrame
            The original DataFrame
            (used to compute overall y-axis limits from 'observed').
        desired_threshold : float
            The experimental threshold to display.
        plot_title : str
            Title for the plot.
        """
        if len(agg_df) <= 1:
            print(
                f"\n{Fore.RED}No sufficient data to track "
                f"experimental values for {plot_title} over time!"
                f"{Fore.RESET}"
            )
            return

        if needs_log_axis(self.op_exp):
            self._plot_log_exp_values_dist(
                agg_df,
                df,
                desired_threshold,
                plot_title,
            )
        else:
            self._plot_linear_exp_values_dist(
                agg_df,
                df,
                desired_threshold,
                plot_title,
            )

    def _scatter_plot_log(
        self,
        df: pd.DataFrame,
        desired_threshold: float,
        plot_title: str,
    ) -> None:
        """
        Create a scatter plot of predicted vs.
        observed values on a logarithmic scale.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'predicted' and 'observed' columns.
        desired_threshold : float
            Experimental threshold to be indicated as reference lines.
        plot_title : str
            Title for the plot.
        scatter_metrics : ScatterMetrics | None
            Metrics (e.g., R2 and RMSE) to be printed.
            If None, a warning is shown.
        """
        # Only exclude zeros for axes that use a log-based transform (log(0)
        # is undefined). For other transforms (None, Negative) zeros are valid.
        _mask = pd.Series(True, index=df.index)
        if needs_log_axis(self.op_exp):
            _mask &= df["observed"] != 0
        if needs_log_axis(self.op_pred):
            _mask &= df["predicted"] != 0
        df = df[_mask]
        df["log_predicted"] = apply_operation(df["predicted"].values, self.op_pred)
        df["log_observed"] = apply_operation(df["observed"].values, self.op_exp)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False
        sns.regplot(
            data=df,
            x="log_predicted",
            y="log_observed",
            color="grey",
            ci=None,
            ax=ax,
            fit_reg=True,
        )

        _thresh_exp = desired_threshold if self.threshold_transformed else apply_operation(np.array([desired_threshold]), self.op_exp)[0]
        _thresh_pred = desired_threshold if self.threshold_transformed else apply_operation(np.array([desired_threshold]), self.op_pred)[0]
        ax.axhline(
            y=_thresh_exp,
            color="orangered",
            linestyle="dotted",
        )
        ax.axvline(
            x=_thresh_pred,
            color="orangered",
            linestyle="dotted",
        )

        if self.threshold_transformed:
            # Assay evaluation: transformed values may be negative (e.g.
            # Negative Log of values > 1), so use data-driven axis bounds.
            x_min = min(df["log_predicted"]) - 0.5
            x_max = max(df["log_predicted"]) + 0.5
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            ax.set_xlim(0, max(df["log_predicted"]))
            ax.set_ylim(0, max(df["log_observed"]))
            ax = Plotter.reset_x_ticks(df["predicted"], ax)

        y_min = min(df["log_observed"]) - 0.5
        y_max = max(df["log_observed"]) + 0.5
        ax.set_ylim(y_min, y_max)
        if needs_log_axis(self.op_exp):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            if not self.threshold_transformed:
                y_ticks = ax.get_yticks()
                if (y_ticks < 0).any():
                    ax.set_yticklabels([round(10**y, 2) for y in y_ticks])
                else:
                    ax.set_yticklabels([int(10**y) for y in y_ticks])
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Experimental", fontweight="bold")
        ax.set_title(plot_title)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.tight_layout()
        plt.show()

    def _scatter_plot_linear(
        self,
        df: pd.DataFrame,
        desired_threshold: float,
        plot_title: str,
    ) -> None:
        """
        Create a scatter plot of predicted vs.
        observed values on a linear scale.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'predicted' and 'observed' columns.
        desired_threshold : float
            Experimental threshold to be indicated as reference lines.
        plot_title : str
            Title for the plot.
        scatter_metrics : ScatterMetrics | None
            Metrics (e.g., R2 and RMSE) to be printed.
            If None, a warning is shown.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False

        # For assay evaluation, apply configured transforms so the plot is in
        # the intended comparison space (e.g. "Negative" -> negated values).
        if self.threshold_transformed:
            df = df.copy()
            df["predicted"] = apply_operation(df["predicted"].values, self.op_pred)
            df["observed"] = apply_operation(df["observed"].values, self.op_exp)

        sns.regplot(
            data=df,
            x="predicted",
            y="observed",
            color="grey",
            ci=None,
            ax=ax,
            fit_reg=True,
        )
        ax.axhline(y=desired_threshold, color="orangered", linestyle="dotted")
        ax.axvline(x=desired_threshold, color="orangered", linestyle="dotted")

        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Experimental", fontweight="bold")
        ax.set_title(plot_title)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.tight_layout()
        plt.show()

    def line_plot_threshold_metrics(
        self,
        threshold: np.ndarray,
        obs: np.ndarray,
        metrics: LinePlotMetrics,
        test_count: int,
        plot_title: str,
    ) -> None:
        """
        Plot a line chart of threshold metrics using precomputed metrics.

        Parameters
        ----------
        threshold : np.ndarray
            Array of threshold values.
        obs : np.ndarray
            Array of observed percentages of compounds tested.
        metrics : LinePlotMetrics
            Precomputed metrics
            (from MetricsCalculator.compute_lineplot_metrics).
        test_count : int
            Number of tests/compounds.
        plot_title : str
            Title for the plot.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False

        # The threshold axis (x) lives on the predicted side, so its log/linear
        # treatment must be driven solely by ``op_pred`` -- not by ``self.scale``
        # which collapses both per-axis operations and would incorrectly log the
        # x-axis when only the experimental (y) side needs a log transform
        # (e.g. assay-evaluation models with op_exp=Log, op_pred=None).
        # For assay evaluation (threshold_transformed), thresholds are already
        # in transformed space — always use linear branch.
        _log_x = needs_log_axis(self.op_pred) and not self.threshold_transformed
        if _log_x:
            ax.plot(
                np.log10(threshold),
                metrics.filtered_metric1,
                color="blue",
                marker=".",
            )
            ax.fill_between(
                np.log10(threshold),
                metrics.ppv_ci_lower,
                metrics.ppv_ci_upper,
                color="blue",
                alpha=0.2,
            )
            ax.plot(
                np.log10(threshold),
                metrics.filtered_metric2,
                color="orange",
                marker=".",
            )
            ax.fill_between(
                np.log10(threshold),
                metrics.for_ci_lower,
                metrics.for_ci_upper,
                color="orange",
                alpha=0.2,
            )
            ax = Plotter.reset_x_ticks(threshold, ax)

            if (
                len(metrics.filtered_metric1) != 0
                and len(metrics.filtered_metric2) != 0
            ):
                _, max_thresh, max_ppv, max_for = metrics.arrow
                plt.annotate(
                    text="",
                    xy=(max_thresh, max_for),
                    xytext=(max_thresh, max_ppv),
                    arrowprops=dict(arrowstyle="<->", color="#CC79A7", linestyle="dashed", lw=2),
                )

            ax2 = ax.twinx()
            ax2.plot(
                np.log10(threshold),
                obs,
                color="grey",
                marker=".",
            )
            ax2.set_xlim(
                min(np.log10(threshold)),
                max(np.log10(threshold)) + 0.5,
            )
        else:
            ax.plot(
                threshold,
                metrics.filtered_metric1,
                color="blue",
                marker=".",
            )
            ax.fill_between(
                threshold,
                metrics.ppv_ci_lower,
                metrics.ppv_ci_upper,
                color="blue",
                alpha=0.2,
            )
            ax.plot(
                threshold,
                metrics.filtered_metric2,
                color="orange",
                marker=".",
            )
            ax.fill_between(
                threshold,
                metrics.for_ci_lower,
                metrics.for_ci_upper,
                color="orange",
                alpha=0.2,
            )

            if (
                len(metrics.filtered_metric1) != 0
                and len(metrics.filtered_metric2) != 0
            ):
                _, max_thresh, max_ppv, max_for = metrics.arrow
                plt.annotate(
                    text="",
                    xy=(max_thresh, max_for),
                    xytext=(max_thresh, max_ppv),
                    arrowprops=dict(arrowstyle="<->", color="#CC79A7", linestyle="dashed", lw=2),
                )
            ax2 = ax.twinx()
            ax2.plot(threshold, obs, color="grey", marker="o")

        ax.set_xlabel("Prediction threshold", fontweight="bold")
        ax.set_ylabel("PPV & FOR (unbiased) - Likelihood% ", fontweight="bold")
        ax2.set_ylabel("% of compounds tested", fontweight="bold")
        ax = Plotter.reset_y_ticks(ax)
        ax2 = Plotter.reset_y_ticks(ax2)
        ax.set_title(plot_title)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        if test_count < 20:
            ax.set_facecolor("lemonchiffon")

        my_handle = [
            Line2D([], [], color="blue", linestyle="solid"),
            Line2D([], [], color="orange", linestyle="solid"),
            Line2D([], [], color="grey", linestyle="solid"),
        ]
        ax.legend(
            handles=my_handle,
            labels=[
                "PPV - Likelihood to extract good compounds at each " "prediction threshold",
                "FOR - Likelihood to discard good compounds at each " "prediction threshold",
                "% of compounds tested (cumulative)",
            ],
            bbox_to_anchor=(0.5, -0.23),
            loc="upper center",
            fontsize=7,
        )
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    def plot_likelihood(
        self,
        threshold: np.ndarray,
        metrics: LikelihoodMetrics,
        desired_threshold: float,
        test_count: int,
        pos_class: str,
        plot_title: str,
    ) -> None:
        """
        Plot the likelihood curves using precomputed metrics.

        Parameters
        ----------
        threshold : np.ndarray
            Array of threshold values.
        metrics : LikelihoodMetrics
            Precomputed likelihood metrics
            (via MetricsCalculator.compute_likelihood_metrics).
        desired_threshold : float
            The experimental threshold.
        test_count : int
            Number of compounds tested.
        pos_class : str
            Indicator for the positive class (e.g., '>' or '<')
            used in the legend.
        plot_title : str
            Title for the plot.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False

        threshold_label = (
            f"Selected Experimental Threshold: {pos_class} " f"{desired_threshold}"
        )

        # See ``line_plot_threshold_metrics``: the x-axis is the predicted
        # threshold, so its scale must be derived from ``op_pred`` only.
        # For assay evaluation (threshold_transformed), thresholds are already
        # in transformed space — always use linear branch.
        _log_x = needs_log_axis(self.op_pred) and not self.threshold_transformed
        if _log_x:
            ax.plot(
                np.log10(threshold),
                metrics.filtered_pred_pos,
                color="turquoise",
                marker=".",
            )
            ax.fill_between(
                np.log10(threshold),
                metrics.ppv_ci_lower,
                metrics.ppv_ci_upper,
                color="turquoise",
                alpha=0.2,
            )
            ax.plot(
                np.log10(threshold),
                metrics.filtered_pred_neg,
                color="indigo",
                marker=".",
            )
            ax.fill_between(
                np.log10(threshold),
                metrics.for_ci_lower,
                metrics.for_ci_upper,
                color="indigo",
                alpha=0.2,
            )
            ax = Plotter.reset_x_ticks(threshold, ax)

            if metrics.filtered_pred_pos.size and metrics.filtered_pred_neg.size:
                _, max_thresh, max_ppv, max_for = metrics.arrow
                plt.annotate(
                    text="",
                    xy=(max_thresh, max_for),
                    xytext=(max_thresh, max_ppv),
                    arrowprops=dict(arrowstyle="<->", color="green"),
                )
                if (
                    metrics.desired_pred_pos != "N/A"
                    and metrics.desired_pred_neg != "N/A"
                ):
                    _t = desired_threshold if self.threshold_transformed else apply_operation(np.array([desired_threshold]), self.op_pred)[0]
                    plt.annotate(
                        text="",
                        xy=(
                            _t,
                            metrics.desired_pred_neg,
                        ),
                        xytext=(
                            _t,
                            metrics.desired_pred_pos,
                        ),
                        arrowprops=dict(arrowstyle="<->", color="#CC79A7", linestyle="dashed", lw=2),
                    )

            ax2 = ax.twinx()
            ax2.plot(
                np.log10(threshold),
                metrics.obs,
                color="grey",
                marker=".",
            )
            ax2.set_xlim(
                min(np.log10(threshold)),
                max(np.log10(threshold)) + 0.5,
            )
        else:
            ax.plot(
                threshold,
                metrics.filtered_pred_pos,
                color="turquoise",
                marker=".",
            )
            ax.fill_between(
                threshold,
                metrics.ppv_ci_lower,
                metrics.ppv_ci_upper,
                color="turquoise",
                alpha=0.2,
            )
            ax.plot(
                threshold,
                metrics.filtered_pred_neg,
                color="indigo",
                marker=".",
            )
            ax.fill_between(
                threshold,
                metrics.for_ci_lower,
                metrics.for_ci_upper,
                color="indigo",
                alpha=0.2,
            )
            if metrics.filtered_pred_pos.size and metrics.filtered_pred_neg.size:
                _, max_thresh, max_ppv, max_for = metrics.arrow
                plt.annotate(
                    text="",
                    xy=(max_thresh, max_for),
                    xytext=(max_thresh, max_ppv),
                    arrowprops=dict(arrowstyle="<->", color="green"),
                )
                if (
                    metrics.desired_pred_pos != "N/A"
                    and metrics.desired_pred_neg != "N/A"
                ):
                    plt.annotate(
                        text="",
                        xy=(desired_threshold, metrics.desired_pred_neg),
                        xytext=(desired_threshold, metrics.desired_pred_pos),
                        arrowprops=dict(arrowstyle="<->", color="#CC79A7", linestyle="dashed", lw=2),
                    )

            ax2 = ax.twinx()
            ax2.plot(
                threshold,
                metrics.obs,
                color="grey",
                marker=".",
            )

        ax.set_xlabel("Prediction threshold", fontweight="bold")
        ax.set_ylabel(
            "PPV & FOR (using SET) - Likelihood% ",
            fontweight="bold",
        )
        ax2.set_ylabel("% of compounds tested", fontweight="bold")
        ax = Plotter.reset_y_ticks(ax)
        ax2 = Plotter.reset_y_ticks(ax2)

        ax.set_title(plot_title)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)

        if test_count < 20:
            ax.set_facecolor("lemonchiffon")

        myHandle = [
            Line2D([], [], color="white"),
            Line2D([], [], color="turquoise", linestyle="solid"),
            Line2D([], [], color="indigo", linestyle="solid"),
            Line2D([], [], color="grey", linestyle="solid"),
        ]
        ax.legend(
            handles=myHandle,
            labels=[
                threshold_label,
                "PPV - Likelihood to extract good compounds according "
                "to pre-selected experimental threshold",
                "FOR - Likelihood to discard good compounds according "
                "to pre-selected experimental threshold",
                "% of compounds tested (cumulative)",
            ],
            bbox_to_anchor=(0.5, -0.23),
            loc="upper center",
            fontsize=7,
        )
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    @staticmethod
    def _set_month_ticks(
        ax: matplotlib.axes.Axes,
        labels: list[str],
        max_labels: int = 30,
    ) -> np.ndarray:
        """
        Put one tick per point and label them with `labels`, returning the x
        positions to plot against.

        The points are plotted against their position rather than the labels
        themselves, which matplotlib would read as categories -- collapsing
        repeated labels onto one position and putting the tick labels out of
        step with the data. Past `max_labels` points only every n-th tick is
        labelled, counted back from the last one so the most recent month is
        always named.
        """
        x_pos = np.arange(len(labels))
        step = -(-len(labels) // max_labels) if len(labels) else 1
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [
                label if (len(labels) - 1 - i) % step == 0 else ""
                for i, label in enumerate(labels)
            ],
            rotation=90,
        )
        return x_pos

    def plot_time_weighted_scores(
        self,
        t_labels: list[str],
        scores: np.ndarray,
        w_scores: np.ndarray | None,
        plot_title: str,
        struct_scores: np.ndarray | None = None,
        r2_scores: np.ndarray | None = None,
        rmse_scores: np.ndarray | None = None,
        max_labels: int = 30,
    ) -> matplotlib.figure.Figure | None:
        """
        Plot the raw and time-weighted similarity
        and correlation scores over time.

        Parameters
        ----------
        t_labels : list[str]
            List of time labels (e.g., ['Feb 2020', 'Mar 2020', ...]).
        scores : np.ndarray
            Array containing similarity and correlation scores.
        w_scores : np.ndarray or None
            Array containing time-weighted similarity and correlation scores.
            None leaves the time-weighted curves out.
        plot_title : str
            Title for the plot.
        struct_scores : np.ndarray, optional
            Structural (Tanimoto) similarity of the prospective compounds to
            the earlier ones. Omitted from the plot when the data carries no
            structures.
        r2_scores : np.ndarray, optional
            R-squared of predicted vs. observed for each timepoint's
            prospective compounds. Plotted on the same [0, 1] axis as the
            similarity metrics so an accuracy drop can be read directly
            against a structural-novelty dip at the same x position; gaps
            mark timepoints with too few prospective compounds.
        rmse_scores : np.ndarray, optional
            RMSE of each timepoint's prospective compounds (see
            :func:`mosses.core.metrics.compute_rmse_over_time`), drawn on the
            same axis; the axis is extended when the RMSE exceeds it.
        max_labels : int, optional
            Maximum number of labelled x ticks; past it only every n-th
            point is labelled, counted back from the last one.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure, e.g. to save it; None when there is nothing to plot.
        """
        if len(t_labels) <= 1 or scores.size == 0:
            print(
                f"{Fore.RED}No sufficient datapoints to " f"generate plots!{Fore.RESET}"
            )
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False

        x_pos = self._set_month_ticks(ax, t_labels, max_labels=max_labels)

        ax.plot(
            x_pos,
            scores[:, 0],
            color="blue",
            label="Similarity of data",
        )
        ax.plot(
            x_pos,
            scores[:, 1],
            color="red",
            label="Similarity of correlations",
        )
        if w_scores is not None:
            ax.plot(
                x_pos,
                w_scores[:, 0],
                color="cyan",
                label="Similarity of data (Time-weighted)",
            )
            ax.plot(
                x_pos,
                w_scores[:, 1],
                color="orange",
                label="Similarity of correlations (Time-weighted)",
            )
        has_rmse = rmse_scores is not None and np.any(np.isfinite(rmse_scores))
        if has_rmse:
            ax.plot(x_pos, rmse_scores, color="green", label="RMSE")
        if struct_scores is not None and np.any(np.isfinite(struct_scores)):
            ax.plot(
                x_pos,
                struct_scores,
                color="black",
                linestyle="--",
                label="Similarity of structures",
            )
        if r2_scores is not None and np.any(np.isfinite(r2_scores)):
            ax.plot(
                x_pos,
                r2_scores,
                color="#7b2d6f",
                marker="o",
                markersize=3,
                label="R\u00b2 per month (prospective)",
            )

        ax.set_xlabel("Model version", fontweight="bold")
        ax.set_ylabel("Scores", fontweight="bold")
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)

        # Below the axes: inside, it covers the low end of the scale where
        # the R2 line often runs. It is a figure legend at the bottom, with
        # the axes laid out above the space it takes, so that it cannot run
        # into the rotated tick labels and the axis label.
        legend = fig.legend(loc="lower center", ncol=2, fontsize=7)
        plt.title(plot_title)
        ax.set_ylim(0, max(1.1, 1.05 * np.nanmax(rmse_scores)) if has_rmse else 1.1)
        legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height
        plt.tight_layout(rect=(0, legend_height / fig.bbox.height, 1, 1))
        plt.show()
        return fig

    def plot_model_stability(
        self,
        agg_df: pd.DataFrame,
        plot_title: str,
        cumulative_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Plot model stability over time using aggregated data.

        Assumes that `agg_df` has been aggregated via
        MetricsCalculator.aggregate_model_stability_data, and
        `cumulative_df` via MetricsCalculator.aggregate_cumulative_r2.

        RMSE is shown per month. R2 is shown cumulatively -- over all
        prospective compounds up to each month -- when `cumulative_df` is
        given, because R2 of a single small monthly batch is too unstable to
        read; together with the cumulative compound count it shows how many
        compounds it takes for the model to reach a stable R2. Without
        `cumulative_df` the per-month R2 of `agg_df` is shown instead.

        Parameters
        ----------
        agg_df : pd.DataFrame
            Aggregated data with columns 'rmse', 'r2' and 'no_of_cpds'.
        plot_title : str
            Title for the plot.
        cumulative_df : pd.DataFrame, optional
            Cumulative data with columns 'r2_cumulative' and
            'no_of_cpds_cumulative'.
        """
        count_legend = "No. of compounds considered for prediction each month"
        panels = [
            (agg_df, "rmse", "no_of_cpds", "RMSE", "RMSE over time", count_legend),
        ]
        if cumulative_df is not None and len(cumulative_df) > 1:
            panels.append(
                (cumulative_df, "r2_cumulative", "no_of_cpds_cumulative",
                 "Cumulative R\u00b2",
                 "Cumulative R\u00b2 (all prospective compounds up to this month)",
                 "No. of compounds (cumulative)")
            )
        else:
            panels.append((agg_df, "r2", "no_of_cpds", "R2", "R2 over time", count_legend))

        for data, metric, count_col, y_label, metric_legend, count_legend in panels:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.canvas.header_visible = False
            ax2 = ax.twinx()
            x_pos = self._set_month_ticks(ax, list(data["model_version"]))

            ax.plot(
                x_pos,
                data[metric],
                color="#0072B2",
                marker="o",
                linestyle="-",
                label=y_label,
            )
            ax2.plot(
                x_pos,
                data[count_col],
                color="grey",
                marker="^",
                linestyle="--",
                label="No. of compounds",
            )

            ax.set_xlabel("Model Version", fontweight="bold")
            ax.set_ylabel(y_label, fontweight="bold")
            ax2.set_ylabel("No. of compounds", fontweight="bold")

            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.set_title(plot_title + " - Model performance over time")
            handles = [
                Line2D([], [], color="#0072B2", marker="o", linestyle="-", label=y_label),
                Line2D([], [], color="grey", marker="^", linestyle="--", label="No. of compounds"),
            ]
            ax.legend(
                handles=handles,
                labels=[metric_legend, count_legend],
                bbox_to_anchor=(0.5, -0.3),
                loc="upper center",
                fontsize=7,
            )

            fig.tight_layout()
            plt.show()

    def scatter_plot(
        self,
        df: pd.DataFrame,
        desired_threshold: float,
        plot_title: str,
    ) -> None:
        """
        Generate a scatter plot comparing predicted
        and observed values based on the selected scale.

        Depending on the scale ('log' or 'linear'),
        this method calls the respective private scatter
        plot method.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'predicted' and 'observed' columns.
        desired_threshold : float
            The experimental threshold to display.
        plot_title : str
            Title for the plot.
        scatter_metrics : ScatterMetrics | None
            Metrics (e.g., R2 and RMSE) to be printed.
            If None, a warning is shown.
        """
        if self.scale == "log":
            self._scatter_plot_log(
                df,
                desired_threshold,
                plot_title,
            )
        else:
            self._scatter_plot_linear(
                df,
                desired_threshold,
                plot_title,
            )

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mosses.core.metrics import LikelihoodMetrics
from mosses.core.metrics import LinePlotMetrics


class Plotter:
    def __init__(self, scale: str = "log") -> None:
        """
        Initialize the Plotter.

        Parameters
        ----------
        scale : str, optional
            Scale to use for the plots ('log' or 'linear'), by default 'log'.
        """
        self.scale = scale

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
        ax.set_ylim(0, 100)
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
            np.log10(agg_df["median_exp"]),
            marker="o",
            color="dodgerblue",
        )

        observed_log = np.log10(df["observed"])
        y_min = observed_log.min() - 0.5
        y_max = observed_log.max() + 0.5
        ylims = np.arange(y_min, y_max, 0.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_ticks(ylims)

        ax.set_xticklabels(agg_df["month_year"], rotation=90, fontsize=8)

        y_ticks = ax.get_yticks()
        new_y_ticks = [int(10**y) for y in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(new_y_ticks, rotation=45, fontsize=8)
        ax.axhline(
            y=np.log10(desired_threshold),
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
        inc = (df["observed"].max() - df["observed"].min()) / 5
        y_min = df["observed"].min() - inc
        y_max = df["observed"].max() + inc
        ylims = np.arange(y_min, y_max, inc)
        ax.plot(
            agg_df["month_year"], agg_df["median_exp"], marker="o", color="dodgerblue"
        )
        ax.set_xlabel("Sample Registration Date", fontweight="bold")
        ax.set_ylabel("Experimental values - Median", fontweight="bold")

        ax.set_xticklabels(agg_df["month_year"], rotation=90)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_ticks(ylims)

        y_ticks = ax.get_yticks()
        new_y_ticks = [round(y, 1) for y in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(new_y_ticks, rotation=45)
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

        if self.scale == "log":
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
        df = df[(df['observed'] != 0) and (df['predicted'] != 0)]
        df["log_predicted"] = np.log10(df["predicted"])
        df["log_observed"] = np.log10(df["observed"])

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

        ax.axhline(
            y=np.log10(desired_threshold),
            color="orangered",
            linestyle="dotted",
        )
        ax.axvline(
            x=np.log10(desired_threshold),
            color="orangered",
            linestyle="dotted",
        )
        ax.set_xlim(0, max(np.log10(df["predicted"])))
        ax.set_ylim(0, max(np.log10(df["observed"])))

        ax = Plotter.reset_x_ticks(df["predicted"], ax)

        ylims = np.arange(
            min(np.log10(df["observed"])) - 0.5,
            max(np.log10(df["observed"])) + 0.5,
            0.5,
        )
        ax.set_ylim(
            min(np.log10(df["observed"])) - 0.5,
            max(np.log10(df["observed"])) + 0.5,
        )
        ax.yaxis.set_ticks(ylims)

        y_ticks = ax.get_yticks()
        if (y_ticks < 0).any():
            new_y_ticks = [round(10**y, 2) for y in y_ticks]
        else:
            new_y_ticks = [int(10**y) for y in y_ticks]

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(new_y_ticks, rotation=45)

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

        if self.scale == "log":
            ax.plot(
                np.log10(threshold),
                metrics.filtered_metric1,
                color="blue",
                marker="o",
            )
            ax.plot(
                np.log10(threshold),
                metrics.filtered_metric2,
                color="orange",
                marker="o",
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
                    arrowprops=dict(arrowstyle="<->", color="plum"),
                )

            ax2 = ax.twinx()
            ax2.plot(
                np.log10(threshold),
                obs,
                color="grey",
                marker="o",
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
                marker="o",
            )
            ax.plot(
                threshold,
                metrics.filtered_metric2,
                color="orange",
                marker="o",
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
                    arrowprops=dict(arrowstyle="<->", color="plum"),
                )
            ax2 = ax.twinx()
            ax2.plot(threshold, obs, color="grey", marker="o")

        ax.set_xlabel("Predicted threshold", fontweight="bold")
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
                "Likelihood to extract good compounds at each " "predicted threshold",
                "Likelihood to discard good compounds at each " "predicted threshold",
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

        if self.scale == "log":
            ax.plot(
                np.log10(threshold),
                metrics.filtered_pred_pos,
                color="turquoise",
                marker="o",
            )
            ax.plot(
                np.log10(threshold),
                metrics.filtered_pred_neg,
                color="indigo",
                marker="o",
            )
            ax = Plotter.reset_x_ticks(threshold, ax)

            if metrics.filtered_pred_pos.size and metrics.filtered_pred_neg.size:
                _, max_thresh, max_ppv, max_for = metrics.arrow
                plt.annotate(
                    text="",
                    xy=(max_thresh, max_for),
                    xytext=(max_thresh, max_ppv),
                    arrowprops=dict(arrowstyle="<->", color="plum"),
                )
                if (
                    metrics.desired_pred_pos != "N/A"
                    and metrics.desired_pred_neg != "N/A"
                ):
                    plt.annotate(
                        text="",
                        xy=(
                            np.log10(desired_threshold),
                            metrics.desired_pred_neg,
                        ),
                        xytext=(
                            np.log10(desired_threshold),
                            metrics.desired_pred_pos,
                        ),
                        arrowprops=dict(arrowstyle="<->", color="green"),
                    )

            ax2 = ax.twinx()
            ax2.plot(
                np.log10(threshold),
                metrics.obs,
                color="grey",
                marker="o",
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
                marker="o",
            )
            ax.plot(
                threshold,
                metrics.filtered_pred_neg,
                color="indigo",
                marker="o",
            )
            if metrics.filtered_pred_pos.size and metrics.filtered_pred_neg.size:
                _, max_thresh, max_ppv, max_for = metrics.arrow
                plt.annotate(
                    text="",
                    xy=(max_thresh, max_for),
                    xytext=(max_thresh, max_ppv),
                    arrowprops=dict(arrowstyle="<->", color="plum"),
                )
                if (
                    metrics.desired_pred_pos != "N/A"
                    and metrics.desired_pred_neg != "N/A"
                ):
                    plt.annotate(
                        text="",
                        xy=(desired_threshold, metrics.desired_pred_neg),
                        xytext=(desired_threshold, metrics.desired_pred_pos),
                        arrowprops=dict(arrowstyle="<->", color="green"),
                    )

            ax2 = ax.twinx()
            ax2.plot(
                threshold,
                metrics.obs,
                color="grey",
                marker="o",
            )

        ax.set_xlabel("Predicted threshold", fontweight="bold")
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
                "Likelihood to extract good compounds according "
                "to pre-selected experimental threshold",
                "Likelihood to discard good compounds according "
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

    def plot_time_weighted_scores(
        self,
        t_labels: list[str],
        scores: np.ndarray,
        w_scores: np.ndarray,
        plot_title: str,
    ) -> None:
        """
        Plot the raw and time-weighted similarity
        and correlation scores over time.

        Parameters
        ----------
        t_labels : list[str]
            List of time labels (e.g., ['Feb 2020', 'Mar 2020', ...]).
        scores : np.ndarray
            Array containing similarity and correlation scores.
        w_scores : np.ndarray
            Array containing time-weighted similarity and correlation scores.
        plot_title : str
            Title for the plot.
        """
        if len(t_labels) <= 1 or scores.size == 0:
            print(
                f"{Fore.RED}No sufficient datapoints to " f"generate plots!{Fore.RESET}"
            )
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False

        ax.plot(
            t_labels,
            scores[:, 0],
            color="blue",
            label="Similarity of data",
        )
        ax.plot(
            t_labels,
            scores[:, 1],
            color="orange",
            label="Similarity of correlation",
        )
        ax.plot(
            t_labels,
            w_scores[:, 0],
            color="green",
            label="Similarity of data (Time-weighted)",
        )
        ax.plot(
            t_labels,
            w_scores[:, 1],
            color="red",
            label="Similarity of correlation (Time-weighted)",
        )

        ax.set_xlabel("Model version", fontweight="bold")
        ax.set_ylabel("Scores", fontweight="bold")
        ax.set_xticklabels(t_labels, rotation=90)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)

        ax.legend(fontsize=7)
        plt.title(plot_title)
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.show()

    def plot_model_stability(
        self,
        agg_df: pd.DataFrame,
        plot_title: str,
    ) -> None:
        """
        Plot model stability over time using aggregated data.

        Assumes that `agg_df` has been aggregated via
        MetricsCalculator.aggregate_model_stability_data.

        Parameters
        ----------
        agg_df : pd.DataFrame
            Aggregated data with columns 'rmse' and 'no_of_cpds'.
        plot_title : str
            Title for the plot.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.canvas.header_visible = False
        ax2 = ax.twinx()

        ax.plot(
            agg_df["model_version"],
            agg_df["rmse"],
            color="deeppink",
            marker="o",
            label="RMSE",
        )
        ax2.plot(
            agg_df["model_version"],
            agg_df["no_of_cpds"],
            color="grey",
            marker="o",
            label="No. of compounds",
        )

        ax.set_ylim(0, 2.5)
        ax.set_xlabel("Model Version", fontweight="bold")
        ax.set_ylabel("RMSE (log scale)", fontweight="bold")
        ax2.set_ylabel("No. of compounds", fontweight="bold")

        ax.set_xticklabels(agg_df["model_version"], rotation=90)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(plot_title + " - Model performance over time")
        handles = [
            Line2D([], [], color="deeppink", marker="o", label="RMSE"),
            Line2D([], [], color="grey", marker="o", label="No. of compounds"),
        ]
        ax.legend(
            handles=handles,
            labels=[
                "RMSEs over time",
                "No. of compounds considered "
                "for prediction each month",
            ],
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

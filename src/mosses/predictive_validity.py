import warnings

import mosses.core.metrics as metrics_calculator
import pandas as pd
from colorama import Fore
from mosses.core.evaluator import EvaluatedData
from mosses.core.evaluator import PredictiveValidityEvaluator
from mosses.core.helpers import print_cpds_info_table
from mosses.core.helpers import print_metrics_table
from mosses.core.helpers import print_note
from mosses.core.helpers import print_ppv_for_table
from mosses.core.helpers import print_unbiased_ppv_for_table
from mosses.core.plotter import Plotter

warnings.filterwarnings("ignore")


def calculate_and_plot(
    all_df: pd.DataFrame,
    evaluated_data: EvaluatedData,
    current_threshold: float,
    plotter: Plotter,
    sample_registration_date: str,
    model_version: str,
    pos_class: str,
    plot_title: str,
    plot_scale: str,
    series: str | None = None,
):
    if (evaluated_data.test_count > 0 and series is None) or (
        len(evaluated_data.all_df) != 0 and series is not None
    ):
        total_compound_num = evaluated_data.test_count + evaluated_data.train_count
        series_title_postfix = f"for Series: {series}" if series else ""
        plot_title = f"{plot_title} (Series: {series})" if series else plot_title
        print_note(f"\n ### Overview {series_title_postfix}\n ---")
        print_cpds_info_table(
            total=total_compound_num,
            test_count=evaluated_data.test_count,
            below_count=evaluated_data.below_count,
            above_count=evaluated_data.above_count,
            good_cpds_percent=evaluated_data.good_cpds_percent,
        )

        print_note(
            f"\n --- \n ### Experimental values over time {series_title_postfix}"
        )
        exp_values_dist = metrics_calculator.aggregate_exp_values_dist_data(
            df=all_df,
            sample_reg_date_col=sample_registration_date,
        )
        plotter.draw_exp_values_dist(
            agg_df=exp_values_dist,
            df=all_df,
            desired_threshold=current_threshold,
            plot_title=plot_title,
        )

        print_note(f"\n --- \n ### Model evaluation {series_title_postfix}")
        if evaluated_data.test_count != 0 and evaluated_data.test_count < 20:
            print(
                f"{Fore.RED}Less than 20 compounds in the "
                f"validation set! Treat the statistics with caution.{Fore.RESET}"
            )

        print_note(
            f"\n#### Predicted vs Experimental Values (prospective) {series_title_postfix}"
        )
        if len(all_df["observed"]) > 0 and len(all_df["predicted"]) > 0:
            scatter_metrics_plot_title = f"{plot_title} - Prospective Validation Set"
            scatter_metrics = metrics_calculator.compute_scatter_metrics(
                df=evaluated_data.test_df,
                scale=plot_scale,
            )
            if evaluated_data.test_count >= 10:
                print_metrics_table(
                    r2=scatter_metrics.r2,
                    rmse=scatter_metrics.rmse,
                )
            else:
                print(
                    f"{Fore.RED}Less than 10 compounds to "
                    f"compute R2 and RMSEs!{Fore.RESET}"
                )

            plotter.scatter_plot(
                df=evaluated_data.test_df,
                desired_threshold=current_threshold,
                plot_title=scatter_metrics_plot_title,
            )
        else:
            print(
                f"{Fore.RED}No sufficient datapoints to generate "
                f"plots {series_title_postfix}!{Fore.RESET}"
            )

    # ============== 2.2 training metrics ===================
    print_note(
        f"\n #### Predicted vs Experimental Values (training set) {series_title_postfix}"
    )
    if evaluated_data.train_count > 0:
        scatter_metrics_plot_title = f"{plot_title} - Training Set"
        scatter_metrics = metrics_calculator.compute_scatter_metrics(
            df=evaluated_data.train_df,
            scale=plot_scale,
        )
        if evaluated_data.train_count >= 10:
            print_metrics_table(
                r2=scatter_metrics.r2,
                rmse=scatter_metrics.rmse,
            )
        else:
            print(
                f"{Fore.RED}Less than 10 compounds to "
                f"compute R2 and RMSEs!{Fore.RESET}"
            )
        plotter.scatter_plot(
            df=evaluated_data.train_df,
            desired_threshold=current_threshold,
            plot_title=scatter_metrics_plot_title,
        )
    else:
        print(
            f"{Fore.RED}Training set is empty - "
            f"Not possible to generate scatter plots "
            f"or compute any metrics {series_title_postfix}!{Fore.RESET}"
        )

    if evaluated_data.test_count >= 10:
        print_note(f"\n#### Model performance over time {series_title_postfix}")
        print_note(f"\n##### RMSE {series_title_postfix}")
        model_stability_data = metrics_calculator.aggregate_model_stability_data(
            df=evaluated_data.test_df,
            scale=plot_scale,
            model_version_col=model_version,
        )
        if len(model_stability_data) > 1:
            plotter.plot_model_stability(
                agg_df=model_stability_data,
                plot_title=plot_title,
            )
        else:
            print(
                f"{Fore.RED}No sufficient data to track model "
                f"performances for {plot_title} {series_title_postfix} over time "
                f"{series_title_postfix}!{Fore.RESET}"
            )

        print_note(
            f"\n##### Similarity of prospective data to training "
            f"set {series_title_postfix}"
        )

        # NOTE: Value set arbitrarily. Might have to be optimized based
        # on a few runs for a couple of pilot projects
        discount_factor = 0.9
        t_labels, scores, w_scores = metrics_calculator.compute_time_weighted_scores(
            df=all_df,
            model_version_col=model_version,
            discount_factor=discount_factor,
            scale=plot_scale,
        )
        plotter.plot_time_weighted_scores(
            t_labels=t_labels,
            scores=scores,
            w_scores=w_scores,
            plot_title=plot_title,
        )

    # ============ 3. threshold metrics and model usage advice ===============
    if evaluated_data.test_count >= 10:
        _, _, thresholds_selection = metrics_calculator.thresh_selection(
            preds=evaluated_data.test_df["predicted"],
            desired_threshold=current_threshold,
            scale=plot_scale,
        )
        threshold_metrics = metrics_calculator.compute_threshold_metrics(
            df=evaluated_data.test_df,
            thresholds=thresholds_selection,
            desired_threshold=current_threshold,
            pos_class=pos_class,
        )
        print_note(f"\n --- \n ### Model usage advice {series_title_postfix}")
        print_note(
            f"\n#### What predicted threshold gives best enrichment {series_title_postfix}"
        )
        desired_project_threshold = threshold_metrics[
            threshold_metrics["threshold"] == current_threshold
        ]
        likelihood_metrics = metrics_calculator.compute_likelihood_metrics(
            threshold=threshold_metrics["threshold"],
            pred_pos_likelihood=threshold_metrics["pred_pos_likelihood"],
            pred_neg_likelihood=threshold_metrics["pred_neg_likelihood"],
            desired_threshold_df=desired_project_threshold,
            scale=plot_scale,
            obs=threshold_metrics["compounds_tested"],
        )
        print_ppv_for_table(
            pre_threshold=current_threshold,
            ppv=likelihood_metrics.desired_pred_pos,
            for_val=likelihood_metrics.desired_pred_neg,
            rec_threshold=round(10 ** likelihood_metrics.arrow[1])
            if plot_scale == "log"
            else round(likelihood_metrics.arrow[1], 1),
            rec_ppv=(
                "N/A"
                if likelihood_metrics.arrow[2] == -1
                else int(likelihood_metrics.arrow[2])
            ),
            rec_for=(
                "N/A"
                if likelihood_metrics.arrow[3] == -1
                else int(likelihood_metrics.arrow[3])
            ),
        )
        plotter.plot_likelihood(
            threshold=threshold_metrics["threshold"],
            metrics=likelihood_metrics,
            desired_threshold=current_threshold,
            test_count=evaluated_data.test_count,
            pos_class=pos_class,
            plot_title=plot_title,
        )
        print_note(
            f"\n#### Explore other experimental thresholds {series_title_postfix}"
        )
        line_plot_metrics = metrics_calculator.compute_lineplot_metrics(
            threshold=threshold_metrics["threshold"],
            metric1=threshold_metrics["ppv"],
            metric2=threshold_metrics["compounds_discarded"],
            scale=plot_scale,
        )
        _, max_thresh, max_ppv, max_for = line_plot_metrics.arrow
        max_ppv = "N/A" if max_ppv == -1 else int(max_ppv)
        max_for = "N/A" if max_for == -1 else int(max_for)
        print_unbiased_ppv_for_table(
            threshold=int(10**max_thresh)
            if plot_scale == "log"
            else round(max_thresh, 1),
            ppv=max_ppv,
            for_val=max_for,
        )

        plotter.line_plot_threshold_metrics(
            threshold=threshold_metrics["threshold"],
            obs=likelihood_metrics.obs,
            test_count=evaluated_data.test_count,
            metrics=line_plot_metrics,
            plot_title=plot_title,
        )

    else:
        print_note("\n --- \n ### Predicted vs Experimental Values")
        plotter.scatter_plot(
            df=evaluated_data.test_df,
            desired_threshold=current_threshold,
            plot_title=scatter_metrics_plot_title,
        )
        print(
            f"{Fore.RED}Less than 10 compounds with measured values"
            f"in the prospective validation set!"
            f"Not possible to compute any metrics!{Fore.RESET}"
        )


def evaluate_pv(
    input_df,
    observed_column,
    predicted_column,
    training_set_column,
    pos_class,
    current_threshold,
    model_version,
    sample_registration_date,
    plot_scale,
    plot_title,
    series_column=None,
):
    """
    Evaluates the model performance for a given data set and desired criterion.

    Parameters:
        input_df (pd.DataFrame): Input dataframe containing observed and predicted data.
        observed_column (str): Name of the column with observed values.
        predicted_column (str): Name of the column with predicted values.
        training_set_column (str): Column indicating whether a sample
            was in the training or test set.
        pos_class (str): String (either ">" or "<=") indicating whether the
            predictions should be greater or lower than the threshold.
        current_threshold (float): Numerical cut-off used by the tool to
            determine PPV and FOR values.
        model_version (str): Version identifier for the model.
        sample_registration_date (str or datetime): Registration date for
            samples, used for temporal analysis.
        plot_scale (str): Scale of the plot (e.g., 'linear', 'log').
        plot_title (str): Name of the model evaluated.
        series_column (str, optional): Optional column name to group
            compounds by series name.

    Returns:
        None: The function prints out all results.
    """
    # ================ 1. Evaluation ===============
    pv_evaluator = PredictiveValidityEvaluator(
        df=input_df,
        pos_class=pos_class,
        desired_threshold=current_threshold,
        training_set_col=training_set_column,
        scale=plot_scale,
        series_column=series_column,
    )
    pv_evaluator.prepare_data(
        observed_col=observed_column,
        predicted_col=predicted_column,
        training_set_col=training_set_column,
        model_version_col=model_version,
        sample_reg_date_col=sample_registration_date,
    )
    plotter = Plotter(scale=plot_scale)

    if series_column is not None:
        series_distribution = pv_evaluator.get_test_series_distribution()
        if len(series_distribution) == 0:
            print(
                f"{Fore.RED}No compounds with measured values for any "
                f"of series in the prospective validation set! Not possible "
                f"to compute any metrics!{Fore.RESET}"
            )

        for series in series_distribution.index:
            evaluated_data = pv_evaluator.evaluate(series=series)
            calculate_and_plot(
                all_df=evaluated_data.all_df,
                evaluated_data=evaluated_data,
                current_threshold=current_threshold,
                plotter=plotter,
                sample_registration_date=sample_registration_date,
                model_version=model_version,
                pos_class=pos_class,
                plot_title=plot_title,
                plot_scale=plot_scale,
                series=series,
            )
    else:
        evaluated_data = pv_evaluator.evaluate()
        calculate_and_plot(
            all_df=evaluated_data.all_df,
            evaluated_data=evaluated_data,
            current_threshold=current_threshold,
            plotter=plotter,
            sample_registration_date=sample_registration_date,
            model_version=model_version,
            pos_class=pos_class,
            plot_title=plot_title,
            plot_scale=plot_scale,
        )

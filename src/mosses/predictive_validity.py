import math
import warnings

import mosses.core.metrics as metrics_calculator
from mosses.core.metrics import (
    apply_operation,
    invert_operation_scalar,
    _resolve_ops,
    needs_log_axis,
    performance_class_set,
    performance_class_opt,
    performance_class_compare,
)
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
    op_exp: str | None = None,
    op_pred: str | None = None,
    threshold_label: str = "prediction threshold",
    threshold_transformed: bool = False,
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
        # For assay evaluation the plot_title is "X vs Y"; use only the
        # experimental (Y) part so the chart header is not cluttered.
        if threshold_transformed and " vs " in plot_title:
            _exp_plot_title = plot_title.split(" vs ", 1)[1]
        else:
            _exp_plot_title = plot_title
        plotter.draw_exp_values_dist(
            agg_df=exp_values_dist,
            df=all_df,
            desired_threshold=current_threshold,
            plot_title=_exp_plot_title,
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
        scatter_plot_generated = False
        if len(all_df["observed"]) > 0 and len(all_df["predicted"]) > 0:
            scatter_metrics_plot_title = f"{plot_title} - Prospective Validation Set"
            scatter_metrics = metrics_calculator.compute_scatter_metrics(
                df=evaluated_data.test_df,
                scale=plot_scale,
                op_exp=op_exp,
                op_pred=op_pred,
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
            scatter_plot_generated = True
        else:
            print(
                f"{Fore.RED}No sufficient datapoints to generate "
                f"plots {series_title_postfix}!{Fore.RESET}"
            )

    # ============== 2.2 training metrics ===================
    # Model performance over time is not meaningful for assay evaluation
    # (two different assay columns are being compared, not a trained model).
    if evaluated_data.test_count >= 10 and not threshold_transformed:
        print_note(f"\n#### Model performance over time {series_title_postfix}")
        print_note(f"\n##### RMSE {series_title_postfix}")
        model_stability_data = metrics_calculator.aggregate_model_stability_data(
            df=evaluated_data.test_df,
            scale=plot_scale,
            model_version_col=model_version,
            op_exp=op_exp,
            op_pred=op_pred,
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
            op_exp=op_exp,
            op_pred=op_pred,
        )
        plotter.plot_time_weighted_scores(
            t_labels=t_labels,
            scores=scores,
            w_scores=w_scores,
            plot_title=plot_title,
        )

    # ============ 3. threshold metrics and model usage advice ===============
    if evaluated_data.test_count >= 10:
        if threshold_transformed:
            # Assay evaluation: work entirely in transformed space.
            # Transform predictions and observations so that the threshold
            # sweep, PPV/FOR comparisons, and axis values all live in the
            # configured transformed space.
            _df_t = evaluated_data.test_df.copy()
            _df_t["predicted"] = apply_operation(_df_t["predicted"].values, op_pred)
            _df_t["observed"] = apply_operation(_df_t["observed"].values, op_exp)
            _, _, thresholds_selection = metrics_calculator.thresh_selection(
                preds=_df_t["predicted"],
                desired_threshold=current_threshold,
                scale="linear",
                op_pred=None,
            )
            # For assay eval the experimental threshold (current_threshold)
            # lives in observed-space and may not be meaningful as a
            # prediction-cutoff. Remove it from the sweep if it falls far
            # outside the prediction range to avoid distorting the x-axis.
            _pred_min = _df_t["predicted"].min()
            _pred_max = _df_t["predicted"].max()
            _pred_margin = (_pred_max - _pred_min) * 0.1
            if current_threshold < (_pred_min - _pred_margin) or current_threshold > (_pred_max + _pred_margin):
                thresholds_selection = thresholds_selection[thresholds_selection != current_threshold]
            threshold_metrics = metrics_calculator.compute_threshold_metrics(
                df=_df_t,
                thresholds=thresholds_selection,
                desired_threshold=current_threshold,
                pos_class=pos_class,
            )
        else:
            _, _, thresholds_selection = metrics_calculator.thresh_selection(
                preds=evaluated_data.test_df["predicted"],
                desired_threshold=current_threshold,
                scale=plot_scale,
                op_pred=op_pred,
            )
            threshold_metrics = metrics_calculator.compute_threshold_metrics(
                df=evaluated_data.test_df,
                thresholds=thresholds_selection,
                desired_threshold=current_threshold,
                pos_class=pos_class,
            )
        print_note(f"\n --- \n ### Model usage advice {series_title_postfix}")
        print_note(
            f"\n#### What {threshold_label} gives best enrichment {series_title_postfix}"
        )
        desired_project_threshold = threshold_metrics[
            threshold_metrics["threshold"] == current_threshold
        ]
        # For assay eval, the experimental threshold may not exist in the
        # prediction sweep (different spaces). Use a synthetic row with N/A
        # so downstream code doesn't crash.
        if desired_project_threshold.empty:
            desired_project_threshold = pd.DataFrame(
                [{
                    "threshold": current_threshold,
                    "pred_pos_likelihood": math.nan,
                    "pred_neg_likelihood": math.nan,
                    "compounds_tested": math.nan,
                }]
            )
        likelihood_metrics = metrics_calculator.compute_likelihood_metrics(
            threshold=threshold_metrics["threshold"],
            pred_pos_likelihood=threshold_metrics["pred_pos_likelihood"],
            pred_neg_likelihood=threshold_metrics["pred_neg_likelihood"],
            desired_threshold_df=desired_project_threshold,
            scale="linear" if threshold_transformed else plot_scale,
            obs=threshold_metrics["compounds_tested"],
            op_pred=None if threshold_transformed else op_pred,
        )
        _, _pe = _resolve_ops(
            "linear" if threshold_transformed else plot_scale,
            None,
            None if threshold_transformed else op_pred,
        )

        # ---------------------------------------------------------------
        # Policy clipping (Option B): align the Predictive Validation
        # "Recommended Threshold" with the Heatmap's "Opt Pred Threshold"
        # so both pages tell the same story for the same model + series.
        #
        # The heatmap applies `performance_class_compare` after computing
        # the raw longest-arrow recommendation: when the SET-side model
        # quality is strictly better than the OPT-side quality, the OPT
        # threshold + PPV/FOR are snapped back to the SET values. We
        # reproduce that here by building a one-row DataFrame in the same
        # shape `calculate_heatmap_metrics` builds and reusing the same
        # policy functions verbatim -- no logic duplication.
        #
        # Backlog (Jenny): instead of snapping back to SET, search for a
        # nearby threshold that improves predictive balance vs SET while
        # still keeping PPV / model-quality above the "Bad" cutoff. Not
        # implemented now.
        # ---------------------------------------------------------------
        raw_max_dist, raw_max_thresh, raw_max_ppv, raw_max_for = (
            likelihood_metrics.arrow
        )

        def _to_num(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if math.isnan(f) else f

        ppv_set_num = _to_num(likelihood_metrics.desired_pred_pos)
        for_set_num = _to_num(likelihood_metrics.desired_pred_neg)
        raw_ppv_num = None if raw_max_ppv == -100 else _to_num(raw_max_ppv)
        raw_for_num = None if raw_max_for == -100 else _to_num(raw_max_for)
        raw_dist_num = None if raw_max_dist == -100 else _to_num(raw_max_dist)
        raw_thresh_user = (
            None
            if raw_max_thresh == -100
            else (
                invert_operation_scalar(raw_max_thresh, _pe)
                if needs_log_axis(_pe)
                else raw_max_thresh
            )
        )

        # Defaults (used when the snap cannot be evaluated -> show raw)
        rec_threshold_display = (
            None
            if raw_thresh_user is None
            else round(raw_thresh_user, 1)
        )
        rec_ppv_display = "N/A" if raw_ppv_num is None else int(raw_ppv_num)
        rec_for_display = "N/A" if raw_for_num is None else int(raw_for_num)

        snap_inputs = (
            ppv_set_num,
            for_set_num,
            raw_ppv_num,
            raw_for_num,
            raw_dist_num,
            raw_thresh_user,
        )
        if all(v is not None for v in snap_inputs):
            policy_row = pd.DataFrame(
                [
                    {
                        # `performance_class_set` / `_opt` only read
                        # PPV/FOR/ArrowLength + compounds count, so this is
                        # the minimum row needed to drive the same decision
                        # the heatmap makes.
                        "Compounds with measured values": (
                            evaluated_data.test_count
                        ),
                        "PPV %": ppv_set_num,
                        "FOR %": for_set_num,
                        "ArrowLength": ppv_set_num - for_set_num,
                        "PPVopt %": raw_ppv_num,
                        "FORopt %": raw_for_num,
                        "Recommended_LongestArrow": raw_dist_num,
                        "Opt Pred Threshold": raw_thresh_user,
                        "SET": current_threshold,
                    }
                ]
            )
            policy_row["Model Quality"] = policy_row.apply(
                performance_class_set, axis=1
            )
            policy_row["Model Quality opt"] = policy_row.apply(
                performance_class_opt, axis=1
            )
            policy_row = policy_row.apply(performance_class_compare, axis=1)

            rec_threshold_display = round(
                float(policy_row["Opt Pred Threshold"].iloc[0]), 1
            )
            rec_ppv_display = int(round(float(policy_row["PPVopt %"].iloc[0])))
            rec_for_display = int(round(float(policy_row["FORopt %"].iloc[0])))

        print_ppv_for_table(
            pre_threshold=current_threshold,
            ppv=likelihood_metrics.desired_pred_pos,
            for_val=likelihood_metrics.desired_pred_neg,
            rec_threshold=rec_threshold_display,
            rec_ppv=rec_ppv_display,
            rec_for=rec_for_display,
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
            scale="linear" if threshold_transformed else plot_scale,
            op_pred=None if threshold_transformed else op_pred,
        )
        _, max_thresh, max_ppv, max_for = line_plot_metrics.arrow
        max_ppv = "N/A" if max_ppv == -1 else int(max_ppv)
        max_for = "N/A" if max_for == -1 else int(max_for)
        print_unbiased_ppv_for_table(
            threshold=int(invert_operation_scalar(max_thresh, _pe))
            if needs_log_axis(_pe)
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
        if (
            (evaluated_data.test_count > 0 and series is None)
            or (len(evaluated_data.all_df) != 0 and series is not None)
        ) and not scatter_plot_generated:
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
        elif not scatter_plot_generated:
            print(
                f"{Fore.RED}There is no data to display. Test data is empty"
                f"Not possible to compute any metrics!{Fore.RESET}"
            )


def _process_and_plot(
    evaluated_data: EvaluatedData | None,
    plotter: Plotter,
    current_threshold: float,
    sample_registration_date: str,
    model_version: str,
    pos_class: str,
    plot_title: str,
    plot_scale: str,
    series: str | None = None,
    op_exp: str | None = None,
    op_pred: str | None = None,
    threshold_label: str = "prediction threshold",
    threshold_transformed: bool = False,
):
    if not evaluated_data:
        series_msg = f" for {series} series" if series else ""
        print(
            f"{Fore.RED}There is no enough data to "
            f"compute any metrics{series_msg}!{Fore.RESET}"
        )
        return
    
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
        op_exp=op_exp,
        op_pred=op_pred,
        threshold_label=threshold_label,
        threshold_transformed=threshold_transformed,
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
    op_exp=None,
    op_pred=None,
    threshold_label: str = "prediction threshold",
    threshold_transformed: bool = False,
):
    """
    Evaluates the model performance for a given data set and desired criterion.

    Parameters:
        input_df (pd.DataFrame): Input dataframe containing observed and predicted data.
        observed_column (str): Name of the column with observed values.
        predicted_column (str): Name of the column with predicted values.
        training_set_column (str): Column indicating whether a sample
            was in the training or test set. If the column does not exist
            in the DataFrame (e.g. the sentinel value "All are prospective"),
            all compounds are treated as the test set.
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
        op_exp=op_exp,
        threshold_transformed=threshold_transformed,
    )
    pv_evaluator.prepare_data(
        observed_col=observed_column,
        predicted_col=predicted_column,
        training_set_col=training_set_column,
        model_version_col=model_version,
        sample_reg_date_col=sample_registration_date,
    )
    plotter = Plotter(scale=plot_scale, op_exp=op_exp, op_pred=op_pred, threshold_transformed=threshold_transformed)

    if not series_column:
        evaluated_data = pv_evaluator.evaluate()
        _process_and_plot(
            evaluated_data=evaluated_data,
            plotter=plotter,
            current_threshold=current_threshold,
            sample_registration_date=sample_registration_date,
            model_version=model_version,
            pos_class=pos_class,
            plot_title=plot_title,
            plot_scale=plot_scale,
            op_exp=op_exp,
            op_pred=op_pred,
            threshold_label=threshold_label,
            threshold_transformed=threshold_transformed,
        )
        return

    if series_column:
        series_distribution = pv_evaluator.get_test_series_distribution()
        if len(series_distribution) == 0:
            print(
                f"{Fore.RED}No compounds with measured values for any "
                f"of series in the prospective validation set! Not possible "
                f"to compute any metrics!{Fore.RESET}"
            )
            return

        for series in series_distribution.index:
            evaluated_data = pv_evaluator.evaluate(series=series)
            _process_and_plot(
                evaluated_data=evaluated_data,
                plotter=plotter,
                current_threshold=current_threshold,
                sample_registration_date=sample_registration_date,
                model_version=model_version,
                pos_class=pos_class,
                plot_title=plot_title,
                plot_scale=plot_scale,
                series=series,
                op_exp=op_exp,
                op_pred=op_pred,
                threshold_label=threshold_label,
                threshold_transformed=threshold_transformed,
            )

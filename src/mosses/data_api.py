"""Headless, data-only computation API for MOSSES.

This module decouples the *computation* performed by the interactive /
plotting entry points (``predictive_validity.evaluate_pv``,
``heatmap.project_heatmap_stats`` …) from their rendering side effects so
the same numbers can be served over a REST API or consumed by other
programmatic clients (e.g. MCP tools).

Every public function here is **pure data**: it takes DataFrames / scalars
and returns JSON-serialisable Python primitives (``dict`` / ``list`` /
``float`` / ``str`` / ``None``).  Nothing is printed and no matplotlib
figure is produced.

The heavy lifting is delegated to the already-pure helpers in
``mosses.core.metrics`` and ``mosses.core.evaluator`` – this module only
orchestrates them and normalises the output, so there is no duplication of
the statistical logic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import mosses.core.metrics as metrics_calculator
from mosses.core.evaluator import EvaluatedData, PredictiveValidityEvaluator
from mosses.core.metrics import (
    apply_operation,
    invert_operation_scalar,
    needs_log_axis,
    performance_class_compare,
    performance_class_opt,
    performance_class_set,
    _resolve_ops,
)

__all__ = [
    "predictive_validity_metrics",
    "jsonify",
]


# --------------------------------------------------------------------------- #
#  Serialisation helpers
# --------------------------------------------------------------------------- #
def _clean_scalar(value: Any) -> Any:
    """Coerce a single numpy / pandas scalar into a JSON-safe primitive.

    ``NaN`` / ``NaT`` / ``inf`` become ``None`` so the value round-trips
    through ``json.dumps`` without producing invalid JSON tokens.
    """
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def jsonify(obj: Any) -> Any:
    """Recursively convert numpy / pandas objects into JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonify(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [jsonify(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [jsonify(rec) for rec in obj.to_dict(orient="records")]
    return _clean_scalar(obj)


def _records(df: pd.DataFrame | None) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe records (or ``[]``)."""
    if df is None or len(df) == 0:
        return []
    return jsonify(df)


# --------------------------------------------------------------------------- #
#  Predictive validity (single model / assay correlation)
# --------------------------------------------------------------------------- #
def _series_metrics(
    evaluated: EvaluatedData | None,
    *,
    current_threshold: float,
    model_version: str,
    sample_registration_date: str,
    pos_class: str,
    plot_scale: str,
    op_exp: str | None,
    op_pred: str | None,
    threshold_transformed: bool,
    include_time_series: bool,
    structure_column: str | None = None,
) -> dict[str, Any]:
    """Compute the full metric bundle for one evaluated (sub-)set."""
    if evaluated is None:
        return {"has_data": False}

    out: dict[str, Any] = {
        "has_data": True,
        "counts": {
            "total": int(evaluated.test_count + evaluated.train_count),
            "train": int(evaluated.train_count),
            "test": int(evaluated.test_count),
            "below_set": int(evaluated.below_count),
            "above_set": int(evaluated.above_count),
            "good_compounds_percent": evaluated.good_cpds_percent,
        },
        "scatter": None,
        "at_set": None,
        "recommended": None,
        "enrichment_sweep": [],
        "stability_over_time": [],
        "cumulative_r2_over_time": [],
        "time_weighted_stability": None,
        "experimental_values_over_time": [],
    }

    all_df = evaluated.all_df

    # ---- Experimental values over time (available with any data) ----
    try:
        exp_dist = metrics_calculator.aggregate_exp_values_dist_data(
            df=all_df.copy(),
            sample_reg_date_col=sample_registration_date,
        )
        out["experimental_values_over_time"] = _records(exp_dist)
    except Exception:
        out["experimental_values_over_time"] = []

    # ---- Scatter metrics (R2 / RMSE) on the prospective/test set ----
    if evaluated.test_count > 0 and len(all_df.get("observed", [])) > 0:
        try:
            scatter = metrics_calculator.compute_scatter_metrics(
                df=evaluated.test_df,
                scale=plot_scale,
                op_exp=op_exp,
                op_pred=op_pred,
            )
            out["scatter"] = {
                "r2": _clean_scalar(scatter.r2),
                "rmse": _clean_scalar(scatter.rmse),
                "reliable": bool(evaluated.test_count >= 10),
            }
        except Exception:
            out["scatter"] = None

    # Everything below needs a minimum sample size to be meaningful.
    if evaluated.test_count < 10:
        return out

    # ---- Model performance over time (not meaningful for assay eval) ----
    if include_time_series and not threshold_transformed:
        try:
            stability = metrics_calculator.aggregate_model_stability_data(
                df=evaluated.test_df,
                scale=plot_scale,
                model_version_col=model_version,
                op_exp=op_exp,
                op_pred=op_pred,
            )
            out["stability_over_time"] = _records(stability)
        except Exception:
            out["stability_over_time"] = []

        try:
            out["cumulative_r2_over_time"] = _records(
                metrics_calculator.aggregate_cumulative_r2(
                    df=evaluated.test_df,
                    scale=plot_scale,
                    model_version_col=model_version,
                    op_exp=op_exp,
                    op_pred=op_pred,
                )
            )
        except Exception:
            out["cumulative_r2_over_time"] = []

        try:
            t_labels, scores, w_scores, struct_scores, r2_scores = (
                metrics_calculator.compute_time_weighted_scores(
                    df=all_df,
                    model_version_col=model_version,
                    discount_factor=0.9,
                    scale=plot_scale,
                    op_exp=op_exp,
                    op_pred=op_pred,
                    structure_col=structure_column,
                    merge_months=True,
                    prospective_index=evaluated.test_df.index,
                )
            )
            out["time_weighted_stability"] = {
                "labels": jsonify(list(t_labels)),
                "scores": jsonify(scores),
                "weighted_scores": jsonify(w_scores),
                # Empty when the project data carries no structures.
                "structural_scores": jsonify(struct_scores),
                # R2 of predicted vs. observed for each month's prospective
                # compounds (same definition as the headline R2) -- not a
                # similarity-to-reference metric, but shares these same
                # timepoints so it can be read against the other curves.
                "r2_scores": jsonify(r2_scores),
            }
        except Exception:
            out["time_weighted_stability"] = None

    # ---- Threshold sweep + PPV/FOR enrichment ----
    try:
        if threshold_transformed:
            _df_t = evaluated.test_df.copy()
            _df_t["predicted"] = apply_operation(_df_t["predicted"].values, op_pred)
            _df_t["observed"] = apply_operation(_df_t["observed"].values, op_exp)
            _, _, thresholds_selection = metrics_calculator.thresh_selection(
                preds=_df_t["predicted"],
                desired_threshold=current_threshold,
                scale="linear",
                op_pred=None,
            )
            _pred_min = _df_t["predicted"].min()
            _pred_max = _df_t["predicted"].max()
            _pred_margin = (_pred_max - _pred_min) * 0.1
            if current_threshold < (_pred_min - _pred_margin) or current_threshold > (
                _pred_max + _pred_margin
            ):
                thresholds_selection = thresholds_selection[
                    thresholds_selection != current_threshold
                ]
            threshold_metrics = metrics_calculator.compute_threshold_metrics(
                df=_df_t,
                thresholds=thresholds_selection,
                desired_threshold=current_threshold,
                pos_class=pos_class,
            )
        else:
            _, _, thresholds_selection = metrics_calculator.thresh_selection(
                preds=evaluated.test_df["predicted"],
                desired_threshold=current_threshold,
                scale=plot_scale,
                op_pred=op_pred,
            )
            threshold_metrics = metrics_calculator.compute_threshold_metrics(
                df=evaluated.test_df,
                thresholds=thresholds_selection,
                desired_threshold=current_threshold,
                pos_class=pos_class,
            )
    except Exception:
        return out

    out["enrichment_sweep"] = _records(
        threshold_metrics[
            [c for c in threshold_metrics.columns if c != "calculation_date"]
        ]
        if "calculation_date" in threshold_metrics.columns
        else threshold_metrics
    )

    desired_project_threshold = threshold_metrics[
        threshold_metrics["threshold"] == current_threshold
    ]
    if desired_project_threshold.empty:
        desired_project_threshold = pd.DataFrame(
            [
                {
                    "threshold": current_threshold,
                    "pred_pos_likelihood": math.nan,
                    "pred_neg_likelihood": math.nan,
                    "compounds_tested": math.nan,
                }
            ]
        )

    try:
        likelihood_metrics = metrics_calculator.compute_likelihood_metrics(
            threshold=threshold_metrics["threshold"],
            pred_pos_likelihood=threshold_metrics["pred_pos_likelihood"],
            pred_neg_likelihood=threshold_metrics["pred_neg_likelihood"],
            desired_threshold_df=desired_project_threshold,
            scale="linear" if threshold_transformed else plot_scale,
            obs=threshold_metrics["compounds_tested"],
            op_pred=None if threshold_transformed else op_pred,
        )
    except Exception:
        return out

    out["at_set"] = {
        "threshold": _clean_scalar(current_threshold),
        "ppv": _clean_scalar(likelihood_metrics.desired_pred_pos),
        "for": _clean_scalar(likelihood_metrics.desired_pred_neg),
    }

    # Recommended (optimized) threshold – reproduce the heatmap's policy
    # so the API and the dashboard tell the same story.
    _, _pe = _resolve_ops(
        "linear" if threshold_transformed else plot_scale,
        None,
        None if threshold_transformed else op_pred,
    )
    raw_max_dist, raw_max_thresh, raw_max_ppv, raw_max_for = likelihood_metrics.arrow

    def _to_num(v: Any) -> float | None:
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

    rec_threshold = None if raw_thresh_user is None else round(raw_thresh_user, 1)
    rec_ppv = None if raw_ppv_num is None else int(raw_ppv_num)
    rec_for = None if raw_for_num is None else int(raw_for_num)

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
                    "Compounds with measured values": evaluated.test_count,
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
        policy_row["Model Quality"] = policy_row.apply(performance_class_set, axis=1)
        policy_row["Model Quality opt"] = policy_row.apply(
            performance_class_opt, axis=1
        )
        policy_row = policy_row.apply(performance_class_compare, axis=1)
        rec_threshold = round(float(policy_row["Opt Pred Threshold"].iloc[0]), 1)
        rec_ppv = int(round(float(policy_row["PPVopt %"].iloc[0])))
        rec_for = int(round(float(policy_row["FORopt %"].iloc[0])))

    out["recommended"] = {
        "threshold": _clean_scalar(rec_threshold),
        "ppv": rec_ppv,
        "for": rec_for,
    }
    return out


def predictive_validity_metrics(
    input_df: pd.DataFrame,
    observed_column: str,
    predicted_column: str,
    training_set_column: str,
    pos_class: str,
    current_threshold: float,
    model_version: str,
    sample_registration_date: str,
    plot_scale: str,
    series_column: str | None = None,
    op_exp: str | None = None,
    op_pred: str | None = None,
    threshold_transformed: bool = False,
    structure_column: str | None = None,
) -> dict[str, Any]:
    """Compute predictive-validity metrics **without any rendering**.

    This is the data-only counterpart of
    :func:`mosses.predictive_validity.evaluate_pv`.  It returns the same
    numbers ``evaluate_pv`` would plot / print, structured as JSON-safe
    primitives.

    Parameters mirror :func:`evaluate_pv`.  When ``series_column`` is
    provided the result contains a per-series breakdown; otherwise a single
    ``"overall"`` entry is returned.

    Returns
    -------
    dict
        ``{"overall": {...}}`` or ``{"series": {name: {...}, ...}}`` where
        each value is the metric bundle produced by :func:`_series_metrics`.
    """
    evaluator = PredictiveValidityEvaluator(
        df=input_df.copy(),
        pos_class=pos_class,
        desired_threshold=current_threshold,
        training_set_col=training_set_column,
        scale=plot_scale,
        series_column=series_column,
        op_exp=op_exp,
        threshold_transformed=threshold_transformed,
        structure_column=structure_column,
    )
    evaluator.prepare_data(
        observed_col=observed_column,
        predicted_col=predicted_column,
        training_set_col=training_set_column,
        model_version_col=model_version,
        sample_reg_date_col=sample_registration_date,
    )

    common = dict(
        current_threshold=current_threshold,
        model_version=model_version,
        sample_registration_date=sample_registration_date,
        pos_class=pos_class,
        plot_scale=plot_scale,
        op_exp=op_exp,
        op_pred=op_pred,
        threshold_transformed=threshold_transformed,
        include_time_series=True,
        structure_column=evaluator.structure_column,
    )

    if not series_column:
        evaluated = evaluator.evaluate()
        return {"overall": _series_metrics(evaluated, **common)}

    distribution = evaluator.get_test_series_distribution()
    series_out: dict[str, Any] = {}
    for series in distribution.index:
        evaluated = evaluator.evaluate(series=series)
        series_out[str(series)] = _series_metrics(evaluated, **common)
    return {"series": series_out}

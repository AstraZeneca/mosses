from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import mosses.core.metrics as metrics_calculator
from mosses.core.helpers import highlight_cells, highlight_pct_cells


def project_heatmap_stats(
    df: pd.DataFrame,
    models_metadata: list[dict[str, Any]],
    series_column: str,
    return_models_with_missing_columns: bool = False,
    return_raw: bool = False,
    threshold_transformed: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    result_df = pd.DataFrame()
    endpoint_category_mapping = {}
    models_order = {}
    units_mapping = {}

    for idx, entry in enumerate(models_metadata):
        endpoint_category_mapping[entry['name']] = entry['attributes']['category']
        models_order[entry['name']] = entry['attributes'].get('heatmap_order', idx + 1)
        units_mapping[entry['name']] = entry['attributes']['units']

    columns = df.columns.to_list()

    models_with_missing_columns = None

    if return_models_with_missing_columns:
        models_with_missing_columns = []

    for model in models_metadata:
        endpoint = model['name']
        observed_column = model['attributes']['observed_column']
        predicted_column =  model['attributes']['predicted_column']
        training_set_column = model['attributes']['training_set_column']
        model_version = model['attributes']['model_version']
        scale = model['attributes'].get('plot_scale', 'linear')
        pos_class = model['attributes']['pos_class']
        raw_threshold = model['attributes']['threshold']
        try:
            selected_threshold = float(raw_threshold)
        except (ValueError, TypeError):
            selected_threshold = 0.0
        sample_reg_date = model['attributes'].get(
            'sample_registration_date', model['attributes']['model_version']
        )
        exp_error = model['attributes']['exp_error']
        op_exp = model['attributes'].get('operation_on_experimental_column')
        op_pred = model['attributes'].get('operation_on_predicted_column')

        if all((
            observed_column in columns,
            predicted_column in columns,
            training_set_column in columns,
        )):
            all_metrics = metrics_calculator.generate_heatmap_table(
                df,
                endpoint,
                observed_column,
                predicted_column,
                training_set_column,
                pos_class,
                selected_threshold,
                series_column,
                model_version,
                sample_reg_date,
                scale,
                exp_error,
                op_exp=op_exp,
                op_pred=op_pred,
                threshold_transformed=threshold_transformed,
            )
            result_df = pd.concat(
                [
                    result_df,
                    all_metrics,
                ],
                axis=0
            )
        elif isinstance(models_with_missing_columns, list):
            models_with_missing_columns.append(endpoint)

        # Always emit a stub "Overall" row for this model, even when the
        # required columns are absent from the project's CSV.  This prevents
        # parameters from disappearing entirely from heatmaps just because a
        # project has no data for them.  The stub row uses 0 compounds and
        # all-NaN metrics so it flows through the FD path (Compounds < 10)
        # and is rendered consistently with the existing few-datapoints style.
        if not all((
            observed_column in columns,
            predicted_column in columns,
            training_set_column in columns,
        )):
            class_annotation = "<" if pos_class in ("<", "<=") else ">"
            # 16 columns to match calculate_heatmap_metrics' FD row layout:
            # [endpoint, series, n_cpds, exp_error, aim, SET, pct_obeying,
            #  NaN, NaN,  <-- PPV/FOR
            #  NaN x 7]  <-- R2, RMSE, LongestArrow, OptThresh, PPVopt, FORopt, TDS
            stub = pd.DataFrame([
                [endpoint, "Overall", 0, exp_error, class_annotation,
                 selected_threshold, 0]
                + [np.nan] * 9
            ])
            result_df = pd.concat([result_df, stub], axis=0)

    if result_df.empty:
        empty = pd.DataFrame()
        if return_raw and return_models_with_missing_columns:
            return empty, models_with_missing_columns
        if return_raw:
            return empty
        if return_models_with_missing_columns:
            return empty.style, models_with_missing_columns
        return empty.style

    result_df.columns = [
        'Model',
        'Series',
        'Compounds with measured values',
        'Exp_Error (log)',
        'Aim',
        'SET',
        'Compounds Obeying SET %',
        'PPV %',
        'FOR %',
        'R2',
        'RMSE (log)',
        'Recommended_LongestArrow',
        'Opt Pred Threshold',
        'PPVopt %',
        'FORopt %',
        'TimeDependant_Stability',
    ]
    
    # Calculate arrow length at the selected experimental threshold
    result_df['ArrowLength'] = result_df['PPV %'] - result_df['FOR %']

    # Calculate predictive balance at optimized threshold
    result_df['Predictive balance (PPV-FOR)'] = result_df['PPVopt %'] - result_df['FORopt %']

    # Assign model quality based on a set of predefined criteria
    result_df['Model Quality'] = result_df.apply(
        metrics_calculator.performance_class_set,
        axis=1,
    )
    result_df['Model Quality opt'] = result_df.apply(
        metrics_calculator.performance_class_opt,
        axis=1,
    )

    result_df['TimeDependant_Stability'] = round(
        result_df['TimeDependant_Stability'],
        1,
    )

    result_df.loc[
        result_df['TimeDependant_Stability'] >= 0.8,
        'Time Dependant Stability Class'
    ] = 'Stable'

    result_df.loc[
        result_df['TimeDependant_Stability'] <= 0.4,
        'Time Dependant Stability Class'
    ] = 'Unstable'

    result_df.loc[
        (
            (result_df['TimeDependant_Stability'] > 0.4) & (result_df['TimeDependant_Stability'] < 0.8)
        ),
        'Time Dependant Stability Class'
    ] = 'Neutral'

    result_df.loc[
        result_df['TimeDependant_Stability'].isna(),
        'Time Dependant Stability Class'
    ] = 'NA'

    # Assign 0, when R2 values are negative
    result_df.loc[result_df['R2'] < 0.0, 'R2'] = 0
    
    # Don't recommend thresholds, if the suggested threshold make the model quality look bad
    result_df = result_df.apply(
        metrics_calculator.performance_class_compare,
        axis=1
    )

    # Recompute predictive balance after performance_class_compare may have
    # rewritten PPVopt %/FORopt % back to the SET-based values; otherwise the
    # displayed balance becomes stale and no longer equals PPVopt - FORopt.
    result_df['Predictive balance (PPV-FOR)'] = result_df['PPVopt %'] - result_df['FORopt %']

    # Add end point category & sorting order to the table
    category_df = pd.DataFrame(endpoint_category_mapping.items())
    category_df.columns = ['Model', 'Category']
    sort_order_df = pd.DataFrame(models_order.items())
    sort_order_df.columns = ['Model','Sort_Order']
    units_df = pd.DataFrame(units_mapping.items())
    units_df.columns = ['Model','Units']

    result_df = result_df.merge(
        category_df,
        on='Model',
    ).merge(
        sort_order_df,
        on='Model',
    ).merge(
        units_df,
        on='Model',
    )

    result_df = result_df.loc[
        :,
        [
            'Series',
            'Category',
            'Model',
            'Units',
            'Aim',
            'SET',
            'Opt Pred Threshold',
            'PPVopt %',
            'FORopt %',
            'Predictive balance (PPV-FOR)',
            'Model Quality opt',
            'Time Dependant Stability Class',
            'Exp_Error (log)',
            'Compounds with measured values',
            'Compounds Obeying SET %',
            'RMSE (log)',
            'R2',
            'PPV %',
            'FOR %',
            'Model Quality',
            'Recommended_LongestArrow',
            'TimeDependant_Stability',
            'ArrowLength',
            'Sort_Order',
        ]
    ]
    result_df = result_df.sort_values(
        by=['Series', 'Sort_Order'],
        ascending=[True, True]
    ).reset_index(drop=True)
    
    result_df = result_df.astype(
        {
            'PPV %': 'Int64',
            'FOR %': 'Int64',
            'PPVopt %': 'Int64',
            'FORopt %': 'Int64',
            'Predictive balance (PPV-FOR)': 'Int64',
        }
    )
    
    result_df = result_df.drop(
        [
            'Recommended_LongestArrow',
            'TimeDependant_Stability',
            'ArrowLength',
            'Sort_Order',
        ],
        axis=1,
    )

    result_df['Opt Pred Threshold'] = result_df['Opt Pred Threshold'].apply(
        lambda x: float(f"{x:.2g}") if pd.notna(x) else x
    )

    display_name_map = {
        'SET': 'Selected experimental threshold (SET)',
        'Exp_Error (log)': 'Experimental Error (log)',
        'PPV %': 'Likelihood to extract good compounds (PPV %) at threshold = SET',
        'FOR %': 'Likelihood to lose good compounds (FOR %) at threshold = SET',
        'Model Quality': 'Model Quality at threshold = SET',
        'Opt Pred Threshold': 'Optimized Prediction Threshold for filtering',
        'PPVopt %': 'Likelihood to extract good compounds (PPV %) at optimized threshold',
        'FORopt %': 'Likelihood to lose good compounds (FOR %) at optimized threshold',
        'Model Quality opt': 'Model Quality at optimized threshold',
    }
    result_df = result_df.rename(columns=display_name_map)

    result_df_regrouped = pd.DataFrame()
    result_df_regrouped = pd.concat(
        [
            result_df_regrouped,
            result_df[result_df.Series != 'Overall']
        ],
        ignore_index=True
    )
    result_df_regrouped = pd.concat(
        [
            result_df_regrouped,
            result_df[result_df.Series == 'Overall']
        ],
        ignore_index=True
    )

    if return_raw:
        if return_models_with_missing_columns:
            return result_df_regrouped, models_with_missing_columns
        return result_df_regrouped

    highlight_subset = [
        'Model Quality at threshold = SET',
        'Model Quality at optimized threshold',
        'Time Dependant Stability Class',
    ]

    result_df_regrouped = result_df_regrouped.style.applymap(
        highlight_cells,
        subset=highlight_subset
    ).format(
        precision=1,
        na_rep=""
    ).hide(axis=0).set_table_styles(
        [
            dict(
                selector='thead th',
                props=[
                    ('text-align', 'left')
                ]
            ),
        ]
    ).set_properties(
        **{'text-align': 'left'}
    )

    if return_models_with_missing_columns:
        return result_df_regrouped, models_with_missing_columns
    else:
        return result_df_regrouped


def global_heatmap_table(
    project_results: list[dict[str, Any]],
    default_model_names: list[str],
    my_project_pid: str | None = None,
    by_series: bool = True,
    cell_mode: str = "quality",
) -> pd.io.formats.style.Styler:
    """Build a cross-project global heatmap table.

    Parameters
    ----------
    project_results : list[dict]
        Each dict contains:
            pid, project_name, phase, modality, therapeutic_area,
            compound_count (int), heatmap_raw (pd.DataFrame from
            project_heatmap_stats with return_raw=True).
    default_model_names : list[str]
        Model display names in heatmap_order (column order).
    my_project_pid : str | None
        If set, this project's rows are pinned to the top.
    by_series : bool
        True  -> show per-series rows (exclude Overall).
        False -> show only the Overall aggregate per project.
    cell_mode : str
        ``"quality"``  -> Model Quality at optimized threshold (categorical).
        ``"pct_set"``  -> Compounds Obeying SET % (numeric 0-100).
    """
    _MODE_COL = {
        "quality": "Model Quality at optimized threshold",
        "pct_set": "Compounds Obeying SET %",
    }
    source_col = _MODE_COL.get(cell_mode, _MODE_COL["quality"])

    rows: list[dict] = []

    for pr in project_results:
        raw = pr["heatmap_raw"]
        if raw is None or raw.empty:
            continue

        if by_series:
            view = raw[raw["Series"] != "Overall"]
        else:
            view = raw[raw["Series"] == "Overall"]

        if view.empty:
            continue

        # Build one output row per (project, series)
        for series_name, grp in view.groupby("Series", sort=False):
            # Series-level compound count: max of "Compounds with measured
            # values" across all models for this series. Each model's value
            # is len(df_all) for that series, so the max gives the largest
            # fully-paired (obs+pred) subset, which is the best per-series
            # approximation without re-counting from the raw CSV.
            # The project-level total is kept as "_sort_compounds" so
            # sorting by project size is preserved.
            series_cpds_col = "Compounds with measured values"
            series_cpds = (
                int(grp[series_cpds_col].max())
                if series_cpds_col in grp.columns and not grp[series_cpds_col].isna().all()
                else 0
            )
            row: dict[str, Any] = {
                "_pid": pr["pid"],
                "_sort_compounds": pr["compound_count"],
                "Project": pr["project_name"],
                "Series": series_name,
                "Modality": pr.get("modality") or "Uncategorized",
                "Phase": pr.get("phase") or "Uncategorized",
                "Therapeutic Area": pr.get("therapeutic_area") or "Uncategorized",
                "Compounds": series_cpds,
            }
            # Fill model columns from this series' data
            for _, mrow in grp.iterrows():
                model_name = mrow.get("Model", "")
                cell_val = mrow.get(source_col, "")
                if model_name in default_model_names:
                    row[model_name] = cell_val if pd.notna(cell_val) else ""
            rows.append(row)

    # Build DataFrame with deterministic column order
    left_cols = ["Project", "Series", "Modality", "Phase", "Therapeutic Area", "Compounds"]
    all_cols = left_cols + default_model_names
    result = pd.DataFrame(rows, columns=all_cols)

    # Blank cells for models not evaluated for a given project/series
    result[default_model_names] = result[default_model_names].fillna("")

    # --- Sorting: my project first, rest by project compound count desc ---
    pids = [r.get("_pid", "") for r in rows]
    is_mine = [p == my_project_pid for p in pids] if my_project_pid else [False] * len(rows)
    sort_compounds = [r.get("_sort_compounds", 0) for r in rows]
    result["_is_mine"] = is_mine
    result["_sort_compounds"] = sort_compounds

    result = result.sort_values(
        by=["_is_mine", "_sort_compounds", "Project", "Series"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    drop_cols = ["_is_mine", "_sort_compounds"]
    if "_pid" in result.columns:
        drop_cols.append("_pid")
    result = result.drop(columns=drop_cols)

    # --- Styling ---
    model_cols_present = [c for c in default_model_names if c in result.columns]

    if cell_mode == "pct_set":
        styled = (
            result.style
            .applymap(highlight_pct_cells, subset=model_cols_present)
            .format(
                {c: lambda v: f"{int(v)}%" if v != "" and v is not None and not (isinstance(v, float) and pd.isna(v)) else ""
                 for c in model_cols_present},
                na_rep="",
            )
            .hide(axis=0)
            .set_table_styles([
                dict(selector="thead th", props=[("text-align", "left")]),
            ])
            .set_properties(**{"text-align": "left"})
        )
    else:
        styled = (
            result.style
            .applymap(highlight_cells, subset=model_cols_present)
            .format(precision=0, na_rep="")
            .hide(axis=0)
            .set_table_styles([
                dict(selector="thead th", props=[("text-align", "left")]),
            ])
            .set_properties(**{"text-align": "left"})
        )
    return styled

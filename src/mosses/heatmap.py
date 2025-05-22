import pandas as pd
from typing import Any
import mosses.core.metrics as metrics_calculator
from mosses.core.helpers import highlight_cells


def project_heatmap_stats(
    df: pd.DataFrame,
    models_metadata: list[dict[str, Any]],
    series_column: str,
    return_models_with_missing_columns: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    result_df = pd.DataFrame()
    endpoint_category_mapping = {}
    models_order = {}
    units_mapping = {}

    for entry in models_metadata:
        endpoint_category_mapping[entry['name']] = entry['attributes']['category']
        models_order[entry['name']] = entry['attributes']['heatmap_order']
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
        scale = model['attributes']['plot_scale']
        pos_class = model['attributes']['pos_class']
        selected_threshold = model['attributes']['threshold']
        sample_reg_date = model['attributes']['sample_registration_date']
        exp_error = model['attributes']['exp_error']

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
    
    result_df.columns = [
        'Model',
        'Series',
        'Compounds',
        'Exp_Error',
        'Aim',
        'SET',
        'Compounds Obeying SET %',
        'PPV %',
        'FOR %',
        'R2',
        'RMSE',
        'Recommended_LongestArrow',
        'Opt Pred Threshold',
        'PPVopt %',
        'FORopt %',
        'TimeDependant_Stability',
    ]
    
    # Calculate arrow length at the selected experimental threshold
    result_df['ArrowLength'] = abs(result_df['PPV %'] - result_df['FOR %'])

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

    # Assign 0, when R2 values are negative; Done to avoid confusions among the users
    result_df.loc[result_df['R2'] < 0.0, 'R2'] = 0
    
    # Don't recommend thresholds, if the suggested threshold make the model quality look bad
    result_df = result_df.apply(
        metrics_calculator.performance_class_compare,
        axis=1
    )

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

    # Columns reorderd to make the interpretation of outputs more intuitive
    result_df = result_df.loc[
        :,
        [
            'Model',
            'Category',
            'Sort_Order',
            'Series',
            'Aim',
            'SET',
            'Units',
            'Compounds',
            'Compounds Obeying SET %',
            'Exp_Error',
            'RMSE',
            'R2',
            'PPV %',
            'FOR %',
            'Model Quality',
            'Opt Pred Threshold',
            'PPVopt %',
            'FORopt %',
            'Model Quality opt',
            'Recommended_LongestArrow',
            'TimeDependant_Stability',
            'Time Dependant Stability Class',
            'ArrowLength',
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

    result_df_regrouped = result_df_regrouped.style.applymap(
        highlight_cells,
        subset=[
            'Model Quality',
            'Model Quality opt',
            'Time Dependant Stability Class'
        ]
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

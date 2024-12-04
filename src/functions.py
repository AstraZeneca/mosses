import re
import pandas as pd


def _convert_column_to_numeric(pd_column):
    """Casts a column to string, ignores any
    errors due to presence of NaNs, trims
    scientific notations, and finally
    attemps to convert the column to numeric."""
    NOTATION_REGEX = r'>|<|NV|;|\?|,'
    return pd.to_numeric(
            pd_column.astype(str).str.replace(
                NOTATION_REGEX, '', regex=True),
                errors='coerce')


def _round_column(pd_column, rounding_decimals=3):
    return pd_column.round(rounding_decimals)


def _process_numeric_columns(df, numeric_columns):
    for c in numeric_columns:
        df[c] = _convert_column_to_numeric(df[c])
        df[c] = _round_column(df[c])
    return df


def preprocess_data_set(df,
                        cpd_name_column,
                        exp_column,
                        pred_column,
                        traintest_column,
                        version_column,
                        sample_regdate_column):
    column_mapping = {
        "Compound Name": cpd_name_column,
        "Observed": exp_column,
        "Predicted": pred_column,
        "CompoundsInTrainingSet": traintest_column,
        "ModelVersion": version_column,
        "SampleRegDate": sample_regdate_column
    }
    numeric_columns = ["Observed"]

    columns_to_select = [v for v in column_mapping.values()]
    column_names = [k for k in column_mapping.keys()]

    df = df[columns_to_select]
    df.columns = column_names
    df = _process_numeric_columns(df, numeric_columns)
    df.dropna(inplace=True)
    return df

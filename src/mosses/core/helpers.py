import sys
import textwrap
import pandas as pd

from IPython.core.display import display_markdown


def is_in_notebook() -> bool:
    """
    Check if the code is running inside a Jupyter notebook.

    Returns
    -------
    bool
        True if running inside a Jupyter notebook, False otherwise.
    """
    return "ipykernel" in sys.modules


def print_note(
    notebook_message: str,
    plain_message: str | None = None,
) -> None:
    """
    Print a message in markdown format if in a notebook; otherwise, print as plain text.

    Parameters
    ----------
    notebook_message : str
        The message to display in a notebook (markdown formatted).
    plain_message : Optional[str], optional
        The plain text message to print when not in a notebook. If not provided,
        `notebook_message` is used after dedentation.
    """
    if is_in_notebook():
        display_markdown(textwrap.dedent(notebook_message), raw=True)
    else:
        print(
            textwrap.dedent(plain_message)
            if plain_message
            else textwrap.dedent(notebook_message)
        )


def print_metrics_table(
    r2: float,
    rmse: float,
) -> None:
    """
    Print a table of evaluation metrics (R² and RMSE).

    Parameters
    ----------
    r2 : float
        Coefficient of determination between experimental and predicted values.
    rmse : float
        Root Mean Squared Error (in log scale) between experimental and predicted values.
    """
    msg = f"""
    | Metric | Value |
    | ------ | ----- |
    | Experimental vs Predicted correlation (Coefficient of determination, R²) | {r2} |
    | Root Mean Squared Error (RMSE in log scale) | {rmse} |
    """
    alt_message = f"""
    Experimental vs Predicted correlation (Coefficient of determination, R²): {r2}
    Root Mean Squared Error (RMSE in log scale): {rmse}
    """
    print_note(msg, alt_message)


def print_cpds_info_table(
    total: int,
    test_count: int,
    below_count: int,
    above_count: int,
    good_cpds_percent: str,
) -> None:
    """
    Print a table summarizing compound information.

    Parameters
    ----------
    total : int
        Total number of compounds with measured values.
    test_count : int
        Number of compounds in the prospective validation (test) set.
    below_count : int
        Number of compounds with measured values below the selected experimental threshold.
    above_count : int
        Number of compounds with measured values above the selected experimental threshold.
    good_cpds_percent : str
        The percentage of compounds considered good based on the selected threshold.
    """
    msg = f"""
    |  | No. of Compounds |
    | ------ | ----- |
    | Compounds with measured values | {total} |
    | Below Selected Experimental Threshold | {below_count} |
    | Above Selected Experimental Threshold | {above_count} |
    | Ratio of good compounds made so far | {good_cpds_percent} |
    | Prospective Validation Set | {test_count} |
    """
    alt_message = f"""
    Compounds below the desired project threshold: {below_count}
    Compounds above the desired project threshold: {above_count}
    Ratio of good compounds made so far: {good_cpds_percent}
    """
    print_note(msg, alt_message)


def print_ppv_for_table(
    pre_threshold: float,
    ppv: float,
    for_val: float,
    rec_threshold: float,
    rec_ppv: float,
    rec_for: float,
) -> None:
    """
    Print a table for PPV (Positive Predictive Value)
    and FOR (False Omission Rate) metrics at different thresholds.

    Parameters
    ----------
    pre_threshold : float
        The selected experimental threshold.
    ppv : float
        PPV at the selected experimental threshold.
    for_val : float
        FOR at the selected experimental threshold.
    rec_threshold : float
        The recommended threshold.
    rec_ppv : float
        PPV at the recommended threshold.
    rec_for : float
        FOR at the recommended threshold.
    """
    msg = f"""
    |  | Prediction Threshold | PPV % | FOR % |
    | ------ | ----- | ----- | ----- |
    | Selected Experimental Threshold | {pre_threshold} | {ppv} | {for_val} |
    | Recommended Threshold | {rec_threshold} | {rec_ppv} | {rec_for} |
    """
    alt_message = f"""
    Threshold Type: Selected Experimental Threshold
    Threshold: {pre_threshold}
    PPV at the selected threshold: {ppv}
    FOR at the selected threshold: {for_val}

    Threshold Type: Recommended Threshold
    Threshold: {rec_threshold}
    PPV at the selected threshold: {rec_ppv}
    FOR at the selected threshold: {rec_for}
    """
    print_note(msg, alt_message)


def print_unbiased_ppv_for_table(
    threshold: float,
    ppv: float,
    for_val: float,
) -> None:
    """
    Print a table for unbiased PPV and FOR values.

    Parameters
    ----------
    threshold : float
        The experimental or prediction threshold.
    ppv : float
        The unbiased Positive Predictive Value (PPV) at the threshold.
    for_val : float
        The unbiased False Omission Rate (FOR) at the threshold.
    """
    msg = f"""
    | Experimental = Prediction threshold | PPV % | FOR % |
    | ----- | ----- | ----- |
    | {threshold} | {ppv} | {for_val} |
    """
    alt_message = f"""
    Threshold Type: Recommended Threshold
    Threshold: {threshold}
    PPV at the selected threshold: {ppv}
    FOR at the selected threshold: {for_val}
    """
    print_note(msg, alt_message)

def highlight_cells(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Highlight cells with different colors in the dataframe based on model quality and stability.

    Uses a color-blind-friendly palette (Paul Tol muted scheme) consistent with
    highlight_pct_cells so that both heatmap modes share the same visual language:
      Good / Stable   → teal   (#44AA99)  — matches high end of pct_set scale
      Medium / Neutral → sand  (#EEDD88)  — matches mid range of pct_set scale
      Bad / Unstable  → vermillion (#CC3311) — matches low end of pct_set scale
      NA / FD         → light grey (#BBBBBB)
    """
    color_mapping = {
        'Good':             'background-color:#44AA99; color:#000',
        'Medium':           'background-color:#EEDD88; color:#000',
        'Bad':              'background-color:#CC3311; color:#fff',
        'FD':               'background-color:#BBBBBB; color:#000',
        'NA in area of SET':'background-color:#BBBBBB; color:#000',

        'Stable':           'background-color:#44AA99; color:#000',
        'Neutral':          'background-color:#EEDD88; color:#000',
        'Unstable':         'background-color:#CC3311; color:#fff',
        'NA':               'background-color:#BBBBBB; color:#000',
    }
    return color_mapping.get(df, '')


def highlight_pct_cells(val) -> str:
    """Color-blind-friendly background for a 0-100 percentage value.

    Uses a 5-step sequential scale (Paul Tol's muted scheme) that is
    readable on both light and dark text and distinguishable under the
    three most common forms of color-vision deficiency.

    The returned CSS always includes ``color`` so that text stays legible
    against the background.
    """
    if val == "" or val is None:
        return ""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if pd.isna(v):
        return ""
    # Stepped thresholds — low is concerning (warm), high is good (cool)
    if v < 20:
        return "background-color:#CC3311; color:#fff"     # vermillion
    if v < 40:
        return "background-color:#EE7733; color:#000"     # orange
    if v < 60:
        return "background-color:#EEDD88; color:#000"     # sand/yellow
    if v < 80:
        return "background-color:#88CCEE; color:#000"     # cyan
    return "background-color:#44AA99; color:#000"         # teal

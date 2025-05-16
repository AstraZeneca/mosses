import math
from dataclasses import dataclass
from datetime import date
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ScatterMetrics:
    r2: float
    rmse: float


@dataclass
class LinePlotMetrics:
    filtered_metric1: np.ndarray
    filtered_metric2: np.ndarray
    arrow: tuple[float, float, float, float]


@dataclass
class LikelihoodMetrics:
    filtered_pred_pos: np.ndarray
    filtered_pred_neg: np.ndarray
    obs: np.ndarray
    arrow: tuple[float, float, float, float]
    desired_pred_pos: int | str
    desired_pred_neg: int | str


def rmse_score(
    obs: np.ndarray,
    pred: np.ndarray,
    scale: str,
) -> float:
    """
    Compute the Root Mean Squared Error (RMSE)
    between observed and predicted values.

    Parameters
    ----------
    obs : np.ndarray
    pred : np.ndarray
    scale : str

    Returns
    -------
    float
        The RMSE score rounded to two decimal places.
    """

    if scale == "log":
        obs = obs[obs != 0]
        pred = pred[pred != 0]
        return round(math.sqrt(mean_squared_error(np.log10(obs), np.log10(pred))), 2)

    return round(math.sqrt(mean_squared_error(obs, pred)), 2)


def thresh_selection(
    preds: np.ndarray,
    desired_threshold: float,
    scale: str,
) -> tuple[float, float, np.ndarray]:
    """
    Generate a sequence of thresholds based on the given prediction values.

    Parameters
    ----------
    preds : np.ndarray
    desired_threshold : float
    scale : str

    Returns
    -------
    tuple[float, float, np.ndarray]
        - **min_thresh (float)** : The minimum threshold value.
        - **max_thresh (float)** : The maximum threshold value.
        - **thresholds (np.ndarray)** : An array of
        threshold values including `desired_threshold`.
    """
    if scale == "log":
        preds = preds[(preds != 0)]
        min_thresh = np.log10(min(preds))
        max_thresh = np.log10(max(preds))
        inc = (max_thresh - min_thresh) / 50
        thresholds = np.append(
            [10**x for x in np.arange(min_thresh, max_thresh, inc)],
            desired_threshold,
        )
    else:
        min_thresh = min(preds)
        max_thresh = max(preds)
        inc = (max_thresh - min_thresh) / 50
        thresholds = np.append(
            np.arange(min_thresh, max_thresh, inc),
            desired_threshold,
        )
    return min_thresh, max_thresh, thresholds


def longest_arrow(
    thresholds: np.ndarray,
    ppv: np.ndarray,
    for_vals: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Identify the threshold with the maximum
    absolute difference between PPV and FOR.

    Parameters
    ----------
    thresholds : np.ndarray
    ppv : np.ndarray
    for_vals : np.ndarray

    Returns
    -------
    tuple[float, float, float, float]
        - **Max Distance (float)**
        - **Best Threshold (float)**
        - **PPV at Best Threshold (float)**
        - **FOR at Best Threshold (float)**
    """
    df = pd.DataFrame({"thresh": thresholds, "ppv": ppv, "for": for_vals})
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        return -1, -1, -1, -1
    distances = np.abs(df["ppv"] - df["for"])
    idx = np.argmax(distances)
    return (
        distances[idx],
        df.loc[idx, "thresh"],
        df.loc[idx, "ppv"],
        df.loc[idx, "for"],
    )


def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing:
        - `observed_binaries`
        - `predicted_binaries`

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - `PPV` (float).
        - `CompoundsDiscarded` (float).
    """
    tn, fp, fn, tp = confusion_matrix(
        df["observed_binaries"],
        df["predicted_binaries"],
        labels=[0, 1],
    ).ravel()

    ppv = (
        precision_score(
            df["observed_binaries"],
            df["predicted_binaries"],
            zero_division=np.nan,
        )
        * 100
        if (tp + fp) > 10
        else np.nan
    )

    compounds_discarded = (fn / (tn + fn)) * 100 if (tn + fn) > 10 else np.nan

    return pd.DataFrame(
        [[ppv, compounds_discarded]],
    ).round(1)


def similarity_score(
    x: np.ndarray,
    y: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    d: int | float,
) -> float:
    """
    Compute a similarity score between training
    and test data predictions based on scaled pairwise distances.

    Parameters
    ----------
    x : np.ndarray
        Ground truth labels for training data.
    y : np.ndarray
        Predictions for training data.
    x2 : np.ndarray
        Ground truth labels for test/prospective data.
    y2 : np.ndarray
        Predictions for test/prospective data.
    d : int or float
        Smoothing factor (d > 0). Smaller values of `d`
        apply stronger penalties to larger distances.

    Returns
    -------
    float
        The mean similarity score across all test samples.
    """
    ref = np.vstack((x, y)).T
    test = np.vstack((x2, y2)).T
    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref)
    X_test = scaler.transform(test)
    dists = cdist(X_ref, X_test)
    return np.exp(-dists.min(axis=0) / d).mean()


def correlation_score(
    x: np.ndarray,
    y: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    d: int | float,
    return_nbr_idx: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Compute a correlation-based similarity score
    between training and test data predictions.

    Parameters
    ----------
    x : np.ndarray
        Ground truth labels for training data.
    y : np.ndarray
        Predictions for training data.
    x2 : np.ndarray
        Ground truth labels for test/prospective data.
    y2 : np.ndarray
        Predictions for test/prospective data.
    d : int or float
        Smoothing factor (d > 0). Smaller values of `d`
        apply stronger penalties to larger correlation differences.
    return_nbr_idx : bool, optional, default=False
        If True, returns the indices of the closest
        training neighbors for visualization.

    Returns
    -------
    float or Tuple[float, np.ndarray]
        - If `return_nbr_idx=False`, returns a single similarity
          score (float).
        - If `return_nbr_idx=True`, returns a tuple containing
          the similarity score and the neighbor indices.
    """
    ref = np.vstack((x, y)).T
    test = np.vstack((x2, y2)).T

    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref)
    X_test = scaler.transform(test)
    dists = cdist(X_ref, X_test)

    nbr_idx = np.argmin(dists, axis=0)
    nbr_r = (spearmanr(X_ref[nbr_idx, 0], X_ref[nbr_idx, 1]).statistic + 1) / 2
    test_r = (spearmanr(X_test[:, 0], X_test[:, 1]).statistic + 1) / 2

    corr_score = np.exp(-np.abs(nbr_r - test_r) / d)
    if return_nbr_idx:
        return corr_score, nbr_idx
    else:
        return corr_score


def _format_month_year(
    df: pd.DataFrame,
    sample_reg_date_col: str,
) -> pd.DataFrame:
    """
    Format a date column in the DataFrame to a 'month_year' string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the date column.
    sample_reg_date_col : str
        Column name that contains the
        registration date in the format '%d-%b-%Y'.

    Raises
    ------
    ValueError
        If the date parsing fails.
    """
    try:
        df["month_year"] = df[sample_reg_date_col].apply(
            lambda x: datetime.strptime(x, "%d-%b-%Y").strftime("%b %Y")
        )
    except Exception:
        raise ValueError("Error parsing 'SampleRegDate'. Expected format '%d-%b-%Y'.")


def _aggregate_exp_values(
    df: pd.DataFrame,
):
    """
    Aggregate experimental values by 'month_year'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the 'month_year'
        column and an 'observed' column.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame containing:
          - no_of_cpds: Count of compounds.
          - median_exp: Median experimental value.
          - min: Minimum observed value.
          - max: Maximum observed value.
          - regdate_month_year: Datetime representation of 'month_year'.
    """
    grouped = df.groupby("month_year")["observed"]

    agg_df = pd.DataFrame()
    agg_df["no_of_cpds"] = grouped.agg(total="count")["total"]
    agg_df["median_exp"] = grouped.agg(Median="median")["Median"]
    agg_df["min"] = grouped.agg(minimum="min")["minimum"]
    agg_df["max"] = grouped.agg(maximum="max")["maximum"]
    agg_df = agg_df.dropna(how="any")

    agg_df["regdate_month_year"] = pd.to_datetime(
        agg_df.index,
        format="%b %Y",
        errors="coerce",
    )
    agg_df = agg_df.sort_values(by="regdate_month_year").reset_index()
    agg_df = agg_df[agg_df["no_of_cpds"] >= 2]
    return agg_df


def aggregate_exp_values_dist_data(
    df: pd.DataFrame,
    sample_reg_date_col: str,
) -> pd.DataFrame:
    """
    Aggregate experimental values distribution data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental data.
    sample_reg_date_col : str
        Column name with sample registration dates.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with experimental values distribution.
    """
    _format_month_year(df, sample_reg_date_col)
    agg_df = _aggregate_exp_values(df)
    return agg_df


def aggregate_model_stability_data(
    df: pd.DataFrame,
    scale: str,
    model_version_col: str,
) -> pd.DataFrame:
    """
    Aggregates data to compute RMSE and compound
    counts for model stability tracking.

    Parameters
    ----------
    df : pd.DataFrame
    scale : str

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns:
            - 'rmse'
            - 'no_of_cpds'
            - 'model_version' (the group label)
            - The index is set to the datetime
              representation of 'model_version'.
    """
    df["model_version_date"] = df[model_version_col].apply(
        lambda x: x.split("-")[0],
    )
    df["model_version_date"] = pd.to_datetime(
        df["model_version_date"],
        errors="coerce",
    )
    df["model_month_year"] = df["model_version_date"].apply(
        lambda x: x.strftime("%b %Y") if pd.notnull(x) else None
    )

    grouped = df.groupby("model_month_year")
    agg_df = pd.DataFrame()
    agg_df["rmse"] = grouped.apply(
        lambda x: rmse_score(x["observed"], x["predicted"], scale),
    )
    agg_df["no_of_cpds"] = grouped["observed"].count()
    agg_df["model_version"] = agg_df.index

    agg_df = agg_df.reset_index(drop=True)

    agg_df["datetime"] = pd.to_datetime(
        agg_df["model_version"],
        format="%b %Y",
        errors="coerce",
    )
    agg_df = agg_df.dropna(subset=["datetime"])
    agg_df = agg_df.sort_values(by="datetime")

    agg_df = agg_df[agg_df["no_of_cpds"] >= 5]
    return agg_df


def compute_lineplot_metrics(
    threshold: np.ndarray, metric1: np.ndarray, metric2: np.ndarray, scale: str
) -> LinePlotMetrics:
    """
    Compute threshold metrics for line plotting.

    Parameters
    ----------
    threshold : np.ndarray
    metric1 : np.ndarray
    metric2 : np.ndarray
    desired_threshold_df : pd.DataFrame
    scale : str

    Returns
    -------
    LinePlotMetrics
        A dataclass instance containing the smoothed metric arrays,
        the longest arrow values,
        and the formatted desired metric values.
    """
    filt_metric1 = savgol_filter(metric1, window_length=3, polyorder=2)
    filt_metric2 = savgol_filter(metric2, window_length=3, polyorder=2)

    if scale == "log":
        logged_threshold = np.log10(threshold)
        arrow = longest_arrow(logged_threshold, filt_metric1, filt_metric2)
    else:
        arrow = longest_arrow(threshold, filt_metric1, filt_metric2)

    return LinePlotMetrics(
        filtered_metric1=filt_metric1,
        filtered_metric2=filt_metric2,
        arrow=arrow,
    )


def compute_likelihood_metrics(
    threshold: np.ndarray,
    obs: np.ndarray,
    pred_pos_likelihood: np.ndarray,
    pred_neg_likelihood: np.ndarray,
    desired_threshold_df: pd.DataFrame,
    scale: str,
) -> LikelihoodMetrics:
    """
    Compute metrics needed for likelihood plotting.

    Parameters
    ----------
    threshold : array-like
    pred_pos_likelihood : array-like
    pred_neg_likelihood : array-like
    desired_threshold_df : pd.DataFrame
    scale : str

    Returns
    -------
    dict
        object containing:
            - filtered_pred_pos: filtered positive likelihoods.
            - filtered_pred_neg: filtered negative likelihoods.
            - obs
            - arrow: tuple of (max_dist, max_tresh, max_ppv, max_for)
              computed via longest_arrow.
            - desired_pred_pos: formatted desired
              predicted positive likelihood.
            - desired_pred_neg: formatted desired
              predicted negative likelihood.
    """
    filt_pred_pos = savgol_filter(
        pred_pos_likelihood,
        window_length=3,
        polyorder=2,
    )
    filt_pred_neg = savgol_filter(
        pred_neg_likelihood,
        window_length=3,
        polyorder=2,
    )
    obs = savgol_filter(obs, window_length=3, polyorder=2)

    if scale == "log":
        logged_threshold = np.log10(threshold)
        max_dist, max_thresh, max_ppv, max_for = longest_arrow(
            logged_threshold,
            filt_pred_pos,
            filt_pred_neg,
        )
        likelihood_value_for_nan = "N/A"

        desired_threshold_df["pred_pos_likelihood"] = (
            likelihood_value_for_nan
            if math.isnan(desired_threshold_df["pred_pos_likelihood"])
            else int(desired_threshold_df["pred_pos_likelihood"])
        )

        desired_threshold_df["pred_neg_likelihood"] = (
            likelihood_value_for_nan
            if math.isnan(desired_threshold_df["pred_neg_likelihood"])
            else int(desired_threshold_df["pred_neg_likelihood"])
        )
    else:
        max_dist, max_thresh, max_ppv, max_for = longest_arrow(
            threshold,
            filt_pred_pos,
            filt_pred_neg,
        )
        likelihood_value_for_nan = 0

        desired_threshold_df["pred_pos_likelihood"] = (
            likelihood_value_for_nan
            if math.isnan(desired_threshold_df["pred_pos_likelihood"])
            else int(desired_threshold_df["pred_pos_likelihood"])
        )

        desired_threshold_df["pred_neg_likelihood"] = (
            likelihood_value_for_nan
            if math.isnan(desired_threshold_df["pred_neg_likelihood"])
            else int(desired_threshold_df["pred_neg_likelihood"])
        )

    return LikelihoodMetrics(
        filtered_pred_pos=filt_pred_pos,
        filtered_pred_neg=filt_pred_neg,
        obs=obs,
        arrow=(
            max_dist,
            max_thresh,
            -1 if max_ppv == -1 else int(max_ppv),
            -1 if max_for == -1 else int(max_for),
        ),
        desired_pred_pos=desired_threshold_df["pred_pos_likelihood"][0],
        desired_pred_neg=desired_threshold_df["pred_neg_likelihood"][0],
    )


def compute_threshold_metrics(
    df: pd.DataFrame,
    thresholds: np.ndarray,
    desired_threshold: float,
    pos_class: str,
) -> pd.DataFrame | None:
    """
    For each threshold (excluding the first one)
    from the provided thresholds array,
    compute various metrics.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing at least:
            - 'observed'
            - 'predicted'
    thresholds : np.ndarray
    desired_threshold : float
    pos_class : str

    Returns
    -------
    pd.DataFrame
        Aggregated metrics DataFrame with columns:
            - 'calculation_date'
            - 'threshold'
            - 'compounds_tested' (as a percentage)
            - 'pred_pos_likelihood'
            - 'pred_neg_likelihood'
            - 'ppv'
            - 'compounds_discarded'
    """
    all_metrics_df = pd.DataFrame()

    for thresh in thresholds[1:]:
        num_obs = len(df[df["predicted"] <= thresh])
        compounds_percent = (num_obs / len(df)) * 100

        if pos_class == ">":
            df["observed_binaries"] = df["observed"].map(lambda x: int(x > thresh))
            df["predicted_binaries"] = df["predicted"].map(lambda x: int(x > thresh))

            obs_pos_subset = df[df["predicted"] > thresh]
            if len(obs_pos_subset) > 10:
                pred_pos_likelihood = (
                    len(obs_pos_subset[obs_pos_subset["observed"] > desired_threshold])
                    / len(obs_pos_subset)
                ) * 100
            else:
                pred_pos_likelihood = math.nan

            obs_neg_subset = df[df["predicted"] <= thresh]
            if len(obs_neg_subset) > 10:
                pred_neg_likelihood = (
                    len(obs_neg_subset[obs_neg_subset["observed"] > desired_threshold])
                    / len(obs_neg_subset)
                ) * 100
            else:
                pred_neg_likelihood = math.nan
        else:
            df["observed_binaries"] = df["observed"].map(lambda x: int(x <= thresh))
            df["predicted_binaries"] = df["predicted"].map(lambda x: int(x <= thresh))
            obs_pos_subset = df[df["predicted"] <= thresh]
            if len(obs_pos_subset) > 10:
                pred_pos_likelihood = (
                    len(obs_pos_subset[obs_pos_subset["observed"] <= desired_threshold])
                    / len(obs_pos_subset)
                ) * 100
            else:
                pred_pos_likelihood = math.nan
            obs_neg_subset = df[df["predicted"] > thresh]
            if len(obs_neg_subset) > 10:
                pred_neg_likelihood = (
                    len(obs_neg_subset[obs_neg_subset["observed"] <= desired_threshold])
                    / len(obs_neg_subset)
                ) * 100
            else:
                pred_neg_likelihood = math.nan

        metrics_df = calculate_all_metrics(df)
        row_df = pd.DataFrame(
            [
                [
                    date.today(),
                    thresh,
                    compounds_percent,
                    pred_pos_likelihood,
                    pred_neg_likelihood,
                ]
            ],
            columns=[
                "calculation_date",
                "threshold",
                "compounds_tested",
                "pred_pos_likelihood",
                "pred_neg_likelihood",
            ],
        )
        combined_df = pd.concat([row_df, metrics_df], axis=1)
        all_metrics_df = pd.concat([all_metrics_df, combined_df], axis=0)

    all_metrics_df.columns = [
        "calculation_date",
        "threshold",
        "compounds_tested",
        "pred_pos_likelihood",
        "pred_neg_likelihood",
        "ppv",
        "compounds_discarded",
    ]
    all_metrics_df = all_metrics_df.drop_duplicates()

    all_metrics_df = all_metrics_df.sort_values(
        by="threshold",
        ascending=False,
    )

    return all_metrics_df


def compute_time_weighted_scores(
    df: pd.DataFrame,
    model_version_col: str,
    discount_factor: float,
    scale: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Compute time-weighted similarity and correlation scores over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'model_version', 'observed', and 'predicted'.
    model_version_col: str
        Version of the model with which the predictions were made
    discount_factor : float
        A discount factor (d > 0). Smaller values
        put more weight on recent scores.
        Setting discount_factor=1 is equivalent to uniform weighting.
    scale : str

    Returns
    -------
    t_labels : list[str]
        List of month-year labels (e.g. ['Feb 2020', 'Mar 2020', ...])
        corresponding to each timepoint (excluding the first).
    scores : np.ndarray
    w_scores : np.ndarray
    """
    df = df.copy()
    df["model_version_date"] = df[model_version_col].apply(lambda x: x.split("-")[0])
    df["model_version_date"] = pd.to_datetime(
        df["model_version_date"],
        errors="coerce",
    )
    if scale == 'log':
        df = df[((df['observed'] != 0) & (df['predicted'] != 0))]
        df[['observed', 'predicted']] == df[['observed', 'predicted']].apply(np.log)
    else:
        df[['observed', 'predicted']] == df[['observed', 'predicted']]

    df_sorted = df.sort_values(by="model_version_date")

    t_arr, _ = np.unique(df_sorted["model_version_date"], return_counts=True)
    t_all = []
    scores_list = []

    for t in t_arr[1:]:
        train_mask = df["model_version_date"] < t
        test_mask = df["model_version_date"] == t
        train_df = df[train_mask]
        prospective_df = df[test_mask]

        if (len(train_df) >= 10) and (len(prospective_df) >= 10):
            train_x = train_df["observed"].to_numpy()
            train_y = train_df["predicted"].to_numpy()
            test_x = prospective_df["observed"].to_numpy()
            test_y = prospective_df["predicted"].to_numpy()

            sim_score = similarity_score(
                train_x,
                train_y,
                test_x,
                test_y,
                0.8,
            )
            corr_score = correlation_score(
                train_x,
                train_y,
                test_x,
                test_y,
                0.2,
                return_nbr_idx=False,
            )
            scores_list.append([sim_score, corr_score])
            t_all.append(t)

    if len(scores_list) == 0:
        return [], np.array([]), np.array([])

    scores = np.vstack(scores_list)

    n = len(t_all)

    x = discount_factor ** np.arange(n - 1, -1, -1)
    W = np.tril(x)
    w_scores = W @ scores / W.sum(axis=-1).reshape(-1, 1)

    t_labels = []
    for _date in t_all:
        t_labels.append(
            _date.astype("datetime64[D]").astype(datetime).strftime("%b %Y")
        )

    return t_labels, scores, w_scores


def compute_scatter_metrics(
    df: pd.DataFrame,
    scale: str,
) -> dict:
    """
    Compute scatter plot metrics: Pearson r², R² score, and RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'observed' and 'predicted' columns.
    scale : str
        Either 'log' or 'linear'. If 'log',
        the metrics are computed on log10-transformed data.

    Returns
    -------
    ScatterMetrics
        A ScatterMetrics with keys:
            - 'r2': R² (float)
            - 'rmse': RMSE (float)
    """
    if scale == "log":
        df = df[((df['observed'] != 0) & (df['predicted'] != 0))]
        obs = np.log10(df["observed"])
        pred = np.log10(df["predicted"])
    else:
        obs = df["observed"]
        pred = df["predicted"]

    r2_val = r2_score(obs, pred)
    r2_val_mod = 0.0 if r2_val < 0.0 else r2_val
    rmse_val = math.sqrt(mean_squared_error(obs, pred))

    return ScatterMetrics(
        r2=round(r2_val_mod, 1),
        rmse=round(rmse_val, 2),
    )

def calculate_heatmap_metrics(
    endpoint: str,
    series: str,
    df_all: pd.DataFrame,
    df: pd.DataFrame,
    pos_class: str,
    selected_threshold: float,
    scale: str,
    exp_error: float,)->pd.DataFrame:

    """
    Compute heatmap metrics: R² score, RMSE, PPV, FOR, longest arrow & Time dependant stability

    Parameters
    ----------
    endpoint : str
        Name of the model. For eg., AZlogD, ePSA, Solubility
    series : str
        Column name which contains information about the compound series
    df_all: pd.DataFrame
        A dataframe that includes all available data for a specific project
    df: pd.DataFrame
        A dataframe that's a subset of df_all and includes only test data
    pos_class: str
        Either '>' or '<' symbol that is used to define class annotations
    selected_threshold: float
        Selected experimental threshold to be used for PPV & FOR calculations
    scale: str
        Either 'log' or 'linear'
    exp_error: float
        Experimental errors calculated based on replicates (n >= 3)

    Returns
    -------
    selected_rec_threshold_df_all: 
        A data frame with all the calulated metrics to be displayed on the heatmap
    """
    class_annotation = 'below' if pos_class == '<' else 'above'

    compounds_below_thresh = len(
        df_all[df_all.observed <= selected_threshold]
    )
    compounds_above_thresh = len(
        df_all[df_all.observed > selected_threshold]
    )
    
    if len(df_all) > 0:
        if pos_class == '<':
            ratio_good_cpds = compounds_below_thresh / (compounds_below_thresh + compounds_above_thresh)
        else:
            ratio_good_cpds = compounds_above_thresh / (compounds_below_thresh + compounds_above_thresh)
            
        ratio_good_cpds = round(ratio_good_cpds*100)
    else:
        ratio_good_cpds = 0

    selected_rec_threshold_df_all = pd.DataFrame()

    all_metrics_df = pd.DataFrame()
    
    if len(df) > 10:
        # Call the thresh function to get the threshold ranges for calculating
        # the various metrics
        _, max_thresh, thresholds_selection = thresh_selection(
            preds= df.predicted,
            desired_threshold= selected_threshold,
            scale= scale,)

        # Exclude the first threshold, as there wouldn't be many compounds
        # below the minimal threshold - Generated statistics can be misleading
        for thresh in thresholds_selection[1:]:
            if pos_class == '>':
                # Mapping to binaries based on different thresholds
                df['observed_binaries'] = df['observed'].map(lambda x: int(x > thresh))
                df['predicted_binaries'] = df['predicted'].map(lambda x: int(x > thresh))
                        
                # Identifying the predicted likelihood to extract good compounds
                # at a selected experimental threshold
                observations_pos_extract = df[df.predicted > thresh]
                if len(observations_pos_extract)>10:
                    pred_pos_likelihood = int((
                        len(
                            observations_pos_extract[observations_pos_extract['observed'] > selected_threshold]
                        ) / len(observations_pos_extract)
                    ) * 100)
                else:
                    pred_pos_likelihood = np.nan
                            
                # Identifying the likelihood to remove good compounds at a
                # selected experimental threshold
                observations_neg_extract = df[df.predicted <= thresh]
                if len(observations_neg_extract) > 10:
                    pred_neg_likelihood = int((
                        len(
                            observations_neg_extract[observations_neg_extract['observed'] > selected_threshold]
                        ) / len(observations_neg_extract)
                    ) * 100)
                else:
                    pred_neg_likelihood = np.nan
                        
                all_metrics = pd.DataFrame(
                    [[
                        endpoint,
                        series,
                        len(df_all),
                        exp_error,
                        class_annotation,
                        thresh,
                        int(ratio_good_cpds),
                        pred_pos_likelihood,
                        pred_neg_likelihood
                    ]]
                )
                all_metrics_df = pd.concat(
                    [
                        all_metrics_df,
                        all_metrics
                    ],
                    axis=0
                )
            else:
                df['observed_binaries'] = df['observed'].map(lambda x: int(x <= thresh))
                df['predicted_binaries'] = df['predicted'].map(lambda x: int(x <= thresh))
                        
                observations_pos_extract = df[df.predicted <= thresh]
                if len(observations_pos_extract) > 10:
                    pred_pos_likelihood = int((
                        len(
                            observations_pos_extract[observations_pos_extract['observed'] <= selected_threshold]
                        ) / len(observations_pos_extract)
                    ) * 100)
                else:
                    pred_pos_likelihood = np.nan
                            
                observations_neg_extract = df[df.predicted > thresh]
                if len(observations_neg_extract)>10:
                    pred_neg_likelihood = int((
                        len(
                            observations_neg_extract[observations_neg_extract['observed'] <= selected_threshold]
                        ) / len(observations_neg_extract)
                    ) * 100)
                else:
                    pred_neg_likelihood = np.nan

                all_metrics = pd.DataFrame(
                    [[
                        endpoint,
                        series,
                        len(df_all),
                        exp_error,
                        class_annotation,
                        thresh,
                        int(ratio_good_cpds),
                        pred_pos_likelihood,
                        pred_neg_likelihood,
                    ]]
                )
                all_metrics_df = pd.concat(
                    [
                        all_metrics_df,
                        all_metrics
                    ],
                    axis=0
                )

        all_metrics_df.columns=[
            'model',
            'series',
            'compounds_tested',
            'exp_error',
            'class_annotate',
            'threshold',
            'prop_compounds_tested',
            'pred_pos_likelihood',
            'pred_neg_likelihood',
        ]

        # Duplicates exist, if the threshold chosen by the user is one among
        # the 50 thresholds chosen for analysis; Remove them prior to 
        # calling the plot functions
        all_metrics_df = all_metrics_df.drop_duplicates()

        all_metrics_df_sorted = all_metrics_df.sort_values(
            by=['threshold'],
            ascending=False,
        )

        # Extracting statistics at desired project threshold
        selected_threshold_df = all_metrics_df_sorted[
            all_metrics_df_sorted.threshold == selected_threshold
        ]
        selected_threshold_df.columns = pd.RangeIndex(selected_threshold_df.columns.size)
        
        # PPV & FOR values fluctuate a lot;
        # It's important to smoothen the curves prior to recommended threshold calculations
        # Apply Savitzky-Golay filter with window size 5 and polynomial order 2
        all_metrics_df_sorted['pred_pos_likelihood'] = savgol_filter(
            all_metrics_df_sorted.pred_pos_likelihood,
            window_length=3,
            polyorder=2,
        )
        all_metrics_df_sorted['pred_neg_likelihood'] = savgol_filter(
            all_metrics_df_sorted.pred_neg_likelihood,
            window_length=3,
            polyorder=2,
        )

        # Compute R2 and RMSES / call the function to calculate the longest arrow
        # and extract the PPVs and FORs corresponding to recommended thresholds
        if scale == 'log':
            df = df[(df.observed != 0) & (df.predicted != 0)]
            r2 = round(
                r2_score(
                    np.log10(df.observed),
                    np.log10(df.predicted)
                ),
                1
            )
            rmse = round(
                math.sqrt(
                    mean_squared_error(
                        np.log10(df.observed),
                        np.log10(df.predicted)
                    )
                ),
                1
            )

            logged_threshold = np.log10(all_metrics_df_sorted.threshold)

            max_dist, max_thresh, max_ppv, max_for = longest_arrow(
                np.array(logged_threshold),
                np.array(all_metrics_df_sorted['pred_pos_likelihood']),
                np.array(all_metrics_df_sorted['pred_neg_likelihood']),
            )

            # Convert threshold back to originial scale for easy interpretation
            max_thresh = 10**max_thresh

        else:
            r2 = round(
                r2_score(
                    df.observed,
                    df.predicted
                ),
                1
            )
            rmse = round(
                math.sqrt(
                    mean_squared_error(
                        df.observed,
                        df.predicted,
                    )
                ),
                1
            )
            max_dist, max_thresh, max_ppv, max_for = longest_arrow(
                np.array(all_metrics_df_sorted.threshold),
                np.array(all_metrics_df_sorted['pred_pos_likelihood']),
                np.array(all_metrics_df_sorted['pred_neg_likelihood']),
            )
        
        
        all_metrics_df_sorted['threshold'] = round(all_metrics_df_sorted['threshold'], 1)
        
        max_dist = np.nan if max_dist == -1 else round(max_dist)
        max_thresh = np.nan if max_thresh == -1 else round(max_thresh,1)
        max_ppv = np.nan if max_ppv == -1 else int(max_ppv)
        max_for = np.nan if max_for == -1 else int(max_for)
        

        # Calculate weighted scores and pick the worst case scenario
        # based on the last weighted score
        discount_factor = 0.9
        _, _, w_scores = compute_time_weighted_scores(
            df=df_all,
            model_version_col='ModelVersion',
            discount_factor=discount_factor,
            scale=scale
        )
        
        if len(w_scores)<=1:
            worst_score = np.nan
        else:
            worst_score = round(w_scores[-1].min(),1)

        recommended_thresh_other_metrics_df = pd.DataFrame(
            [[
                r2,
                rmse,
                max_dist,
                max_thresh,
                max_ppv,
                max_for,
                worst_score,
            ]]
        )
        selected_rec_threshold_df = pd.concat(
            [
                selected_threshold_df,
                recommended_thresh_other_metrics_df,
            ],
            axis=1,
        )
    else:
        selected_threshold_df = pd.concat(
            [
                pd.DataFrame(
                    [[
                        endpoint,
                        series,
                        len(df_all),
                        exp_error,
                        class_annotation,
                        selected_threshold,
                        int(ratio_good_cpds),
                    ]]
                ),
                pd.DataFrame(
                    np.full(
                        shape=(1,2),
                        fill_value=np.nan
                    )
                )
            ],
            axis=1
        )
        selected_threshold_df.columns = pd.RangeIndex(selected_threshold_df.columns.size)
        recommended_thresh_other_metrics_df = pd.DataFrame(
            np.full(
                shape=(1,7),
                fill_value=np.nan
            )
        )
        selected_rec_threshold_df = pd.concat(
            [
                selected_threshold_df,
                recommended_thresh_other_metrics_df,
            ],
            axis=1,
        )

    selected_rec_threshold_df_all = pd.concat(
        [
            selected_rec_threshold_df_all,
            selected_rec_threshold_df,
        ],
        axis=0,
    )
    selected_rec_threshold_df_all.columns = pd.RangeIndex(
        selected_rec_threshold_df_all.columns.size
    )
    return selected_rec_threshold_df_all

def generate_heatmap_table(
    data: pd.DataFrame,
    endpoint: str,
    observed_column: str,
    predicted_column: str,
    training_set_column: str,
    pos_class: str,
    selected_threshold: float,
    series_column: str,
    model_version: str,
    sample_reg_date: str,
    scale:str,
    exp_error: float,
) -> pd.DataFrame:

    """
    Generate heatmap metrics table corresponding to different series for a project

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe that includes all available data for a specific project
    endpoint : str
        Name of the model. For eg., AZlogD, ePSA, Solubility
    observed_column: str
        Column that includes experimental data for a specific endpoint
    predicted_column: str
        Column that includes predicted data for a specific endpoint
    training_set_column: str
        Column that includes 'train/test' annotations for each compound
    pos_class: str
        Either '>' or '<' symbol that is used to define class annotations
    selected_threshold: float
        Selected experimental threshold to be used for PPV & FOR calculations
    series_column : str
        Column name which contains information about the compound series
    model_version: str
        Version of the model with which the predictions were made
    sample_reg_date: str
        Date that corresponds to the first time, a compound is registered in the system
    scale: str
        Either 'log' or 'linear'
    exp_error: float
        Experimental errors calculated from replicates (n >= 3)

    Returns
    -------
    metrics_all: 
        A data frame with the calculated metrics for all series and all end points
    """

    observed_parameter = pd.to_numeric(
        data[observed_column].astype(str).str.replace(
            '>|<|NV|;|\?|,',
            '',
            regex=True,
        ),
        errors='coerce',
    )
    predicted_parameter = data[predicted_column]
    observed_predicted_df = pd.concat(
        [
            data['Compound Name'],
            observed_parameter,
            predicted_parameter,
            data[training_set_column],
            data[series_column],
            data[model_version],
            data[sample_reg_date]
        ],
        axis=1,
        keys=[
            'Compound Name',
            'observed',
            'predicted',
            'CompoundsInTrainingSet',
            'Series',
            'ModelVersion',
            'SampleRegDate',
        ]
    ).dropna(
        subset=[
            'Compound Name',
            'observed',
            'predicted',
            'CompoundsInTrainingSet',
        ]
    )

    observed_predicted_test = observed_predicted_df[
        observed_predicted_df.CompoundsInTrainingSet.isin(['test',np.nan])
    ]

    metrics_all = pd.DataFrame()
    series = 'Overall'
    metrics_overall = calculate_heatmap_metrics(
        endpoint,
        series,
        observed_predicted_df,
        observed_predicted_test,
        pos_class,
        selected_threshold,
        scale,
        exp_error,
    )
    metrics_all = pd.concat([metrics_all,metrics_overall])

    test_series_count = observed_predicted_test.groupby(by='Series')['Compound Name'].count()
        
    for series in test_series_count.index:
        series_all_df = observed_predicted_df[observed_predicted_df['Series']==series]
        series_test = observed_predicted_test[observed_predicted_df['Series']==series]
        metrics_series = calculate_heatmap_metrics(
            endpoint,
            series,
            series_all_df,
            series_test,
            pos_class,
            selected_threshold,
            scale,
            exp_error,
        ) 
        metrics_all = pd.concat([metrics_all,metrics_series],axis=0)

    return metrics_all

def performance_class_set(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign model performance classes for the metrics calculated using the selected experimental threshold

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing R2, PPV% & Arrowlength.

    Returns
    -------
    model_perf: str
        A string that defines model performance
    """    
    if ((df['PPV %'] >= 75) and (df['ArrowLength'] >= 50)):
        model_perf = 'Good'
    elif ((df['PPV %'] in range(55,75)) or  (df['ArrowLength'] in range(20,50))):
        model_perf = 'Medium'
    elif ((df['PPV %'] < 55) or (df['ArrowLength'] < 20)):
        model_perf = 'Bad'
    else:
        model_perf = 'NA in area of SET'
    return model_perf

def performance_class_opt(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign model performance classes for the metrics calculated using the recommended experimental threshold

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing R2, PPVopt% & Recommended_LongestArrow.

    Returns
    -------
    model_perf: str
        A string that defines model performance
    """    
    if ((df['PPVopt %'] >= 75) and (df['Recommended_LongestArrow'] >= 50)):
        model_perf = 'Good'
    elif ((df['PPVopt %'] in range(55,75)) or  (df['Recommended_LongestArrow'] in range(20,50))):
        model_perf = 'Medium'
    elif (((df['PPVopt %'] < 55) or (df['Recommended_LongestArrow'] < 20))):
        model_perf = 'Bad'
    else:
        model_perf = 'NA in area of SET'
    return model_perf

def performance_class_compare(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign model performance classes for the metrics calculated using the recommended experimental threshold

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing R2, PPVopt% & Recommended_LongestArrow.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with modified thresholds and metrics
    """    
    if (
        (df['Model Quality'] == 'Good')
        and (
            (df['Model Quality opt'] == 'Medium')
            or (df['Model Quality opt'] == 'Bad')
        )
    ):
        (
            df['Opt Pred Threshold'],
            df['PPVopt %'],
            df['FORopt %'],
            df['Recommended_LongestArrow'],
            df['Model Quality opt'] 
        ) = (
            df['SET'],
            df['PPV %'],
            df['FOR %'],
            df['ArrowLength'],
            df['Model Quality']
        )
    elif (
        (df['Model Quality'] == 'Medium')
        and (df['Model Quality opt'] == 'Bad')
    ):
        (
            df['Opt Pred Threshold'],
            df['PPVopt %'],
            df['FORopt %'],
            df['Recommended_LongestArrow'],
            df['Model Quality opt']
        ) = (
            df['SET'],
            df['PPV %'],
            df['FOR %'],
            df['ArrowLength'],
            df['Model Quality']
        )
    return df
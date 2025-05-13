from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluatedData:
    all_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_count: int
    test_count: int
    below_count: int
    above_count: int
    good_cpds_percent: str


class PredictiveValidityEvaluator:
    def __init__(
        self,
        df: pd.DataFrame,
        pos_class: str,
        desired_threshold: float,
        training_set_col: str,
        scale: str = "log",
        series_column: str | None = None,
    ) -> None:
        """
        Initialize the PredictiveValidityEvaluator.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing predictive validity data.
        pos_class : str
            The positive class indicator (e.g., '<' or '>') to
            determine good compounds.
        desired_threshold : float
            The experimental threshold used for evaluation.
        training_set_col : str
            Column name in `df` indicating whether a
            compound is in the training set.
        scale : str, optional
            Scale to use for analysis ('log' or 'linear'), by default 'log'.
        series_column : Optional[str], optional
            Column name to filter data by series, by default None.
        """
        self.df = df
        self.pos_class = pos_class
        self.desired_threshold = desired_threshold
        self.training_set_col = training_set_col
        self.scale = scale
        self.series_column = series_column

    def prepare_data(
        self,
        observed_col: str,
        predicted_col: str,
        training_set_col: str,
        model_version_col: str,
        sample_reg_date_col: str,
    ) -> None:
        """
        Prepare the DataFrame by converting observed values
        to numeric and subsetting required columns.

        Parameters
        ----------
        observed_col : str
            Column name for observed values.
        predicted_col : str
            Column name for predicted values.
        training_set_col : str
            Column name indicating the training set membership.
        model_version_col : str
            Column name indicating the model version.
        sample_reg_date_col : str
            Column name indicating the sample registration date.
        """
        if self.df is None or self.df.empty:
            raise Exception("Dataframe was not provided or it is empty")
        self.df["observed"] = pd.to_numeric(
            self.df[observed_col]
            .astype(str)
            .str.replace(
                r">|<|NV|;|\?|,",
                "",
                regex=True,
            ),
            errors="coerce",
        )
        self.df["predicted"] = self.df[predicted_col]
        columns = [
            "Compound Name",
            "observed",
            "predicted",
            training_set_col,
            model_version_col,
            sample_reg_date_col,
        ]
        drop_na_columns = [
            "Compound Name",
            "observed",
            "predicted",
            training_set_col,
        ]
        if self.series_column is not None:
            columns.append(self.series_column)
            drop_na_columns.append(self.series_column)

        self.df = self.df[columns].dropna(subset=drop_na_columns)

    def split_train_test(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the provided DataFrame into training and test sets.

        The split is based on the value of the `training_set_col` where:
          - 'train' indicates a training compound.
          - 'test' or NaN indicates a test compound.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to split.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training DataFrame and the test DataFrame.
        """
        train_df = df[df[self.training_set_col] == "train"]
        test_df = df[df[self.training_set_col].isin(["test", np.nan])]
        return train_df, test_df

    def get_ratio_good_cpds(self, below_count: int, above_count: int) -> float:
        """
        Calculate the ratio of good compounds based
        on the positive class indicator.

        For pos_class '<', the ratio is calculated as:
            below_count / (below_count + above_count)
        Otherwise, it is:
            above_count / (below_count + above_count)

        Parameters
        ----------
        below_count : int
            Count of compounds with observed values <= desired_threshold.
        above_count : int
            Count of compounds with observed values > desired_threshold.

        Returns
        -------
        float
            The ratio of good compounds.
        """
        if self.pos_class == "<":
            return below_count / (below_count + above_count)
        return above_count / (below_count + above_count)

    def get_percent(self, ratio_num: float) -> str:
        """
        Convert a numerical ratio to a percentage string.

        Parameters
        ----------
        ratio_num : float
            The ratio to be converted.

        Returns
        -------
        str
            The ratio expressed as a percentage string (e.g., '75%').
        """
        return f"{int(ratio_num*100)}%"

    def get_test_series_distribution(self) -> pd.DataFrame:
        """
        Compute the distribution of test compounds by series.

        Groups the test set by the series column
        and counts the number of compounds in each group.
        This method assumes that `self.series_column` is not None.

        Returns
        -------
        pd.DataFrame
            A DataFrame with counts of compounds for each series.
        """
        _, test_df = self.split_train_test(self.df)
        return test_df.groupby(by=self.series_column)["Compound Name"].count()

    def evaluate(self, series: str | None = None) -> EvaluatedData:
        """
        Evaluate predictive validity by splitting data
        into training and test sets, and computing summary counts.

        If a series is provided and `self.series_column` is set,
        the data is filtered to that series.
        Otherwise, all data is used.

        Parameters
        ----------
        series : Optional[str], optional
            Specific series to filter the data by, by default None.

        Returns
        -------
        EvaluatedData
            A dataclass instance containing the evaluated data
            and computed metrics.
        """
        df = self.df.copy()
        if series:
            df = df[df[self.series_column] == series]

        train_df, test_df = self.split_train_test(df)
        train_count = len(train_df)
        test_count = len(test_df)

        below_count = len(df[df["observed"] <= self.desired_threshold])
        above_count = len(df[df["observed"] > self.desired_threshold])

        good_cpds_percent = self.get_percent(
            ratio_num=self.get_ratio_good_cpds(below_count, above_count)
        )

        return EvaluatedData(
            all_df=df,
            train_df=train_df,
            test_df=test_df,
            train_count=train_count,
            test_count=test_count,
            below_count=below_count,
            above_count=above_count,
            good_cpds_percent=good_cpds_percent,
        )

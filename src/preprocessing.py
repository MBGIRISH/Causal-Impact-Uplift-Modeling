"""
Preprocessing utilities for marketing campaign causal analysis.

Handles data loading, cleaning, train/test splits, and balance diagnostics.
Built to be reusable across different notebooks and analysis scripts.

Note: This was developed iteratively during the project - some functions
were added as needed rather than planned upfront.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ColumnConfig:
    """Configuration object for key column names used across the project."""

    user_id_col: str = "user_id"
    treatment_col: str = "treatment"
    outcome_col: str = "purchase"
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None


def load_marketing_data(
    path: str,
    column_config: Optional[ColumnConfig] = None,
) -> Tuple[pd.DataFrame, ColumnConfig]:
    """
    Load the user-level marketing campaign dataset from a CSV file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.
    column_config : Optional[ColumnConfig]
        Optional configuration for column names and feature lists.
        If numeric_features / categorical_features are None, they will be
        inferred (numeric vs non-numeric) after loading.

    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe.
    column_config : ColumnConfig
        Completed column configuration with inferred feature lists if needed.
    """
    df = pd.read_csv(path)

    if column_config is None:
        column_config = ColumnConfig()

    # Basic presence checks
    required_cols = [
        column_config.user_id_col,
        column_config.treatment_col,
        column_config.outcome_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    # Ensure treatment and outcome are numeric/binary
    df[column_config.treatment_col] = df[column_config.treatment_col].astype(int)

    # Outcome can be binary or continuous; try to cast to float
    df[column_config.outcome_col] = pd.to_numeric(
        df[column_config.outcome_col],
        errors="coerce",
    )

    # Infer feature lists if not provided
    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            column_config.user_id_col,
            column_config.treatment_col,
            column_config.outcome_col,
        }
    ]

    if column_config.numeric_features is None or column_config.categorical_features is None:
        numeric_features = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_features = [c for c in feature_cols if c not in numeric_features]

        # Only overwrite if not provided by the caller
        if column_config.numeric_features is None:
            column_config.numeric_features = numeric_features
        if column_config.categorical_features is None:
            column_config.categorical_features = categorical_features

    return df, column_config


def stratified_train_test_split(
    df: pd.DataFrame,
    column_config: ColumnConfig,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets, stratified on (treatment, outcome).

    Motivation:
    - We want the evaluation set to preserve the joint distribution of treatment
      and outcome, so uplift / causal metrics are well-behaved.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    column_config : ColumnConfig
        Configuration with treatment and outcome column names.
    test_size : float, default=0.3
        Fraction of data to put in the test set.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    df_train : pd.DataFrame
        Training subset.
    df_test : pd.DataFrame
        Test subset.
    """
    t = df[column_config.treatment_col]
    y = df[column_config.outcome_col]

    # Create a combined stratification label; handle non-binary / NaN outcomes carefully.
    # For simplicity, binarize outcome at >0 if it's numeric continuous.
    if pd.api.types.is_numeric_dtype(y):
        y_bin = (y > 0).astype(int).fillna(0)
    else:
        y_bin = y.astype(int)

    strat_label = t.astype(str) + "_" + y_bin.astype(str)

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_label,
    )

    return df_train, df_test


def standardized_mean_difference(
    x_treated: pd.Series,
    x_control: pd.Series,
) -> float:
    """
    Compute the standardized mean difference (Cohen's d) between treated and control.

    This is a primary covariate balance metric for causal inference.
    Values |SMD| < 0.1 are typically considered well-balanced.
    """
    m1, m0 = x_treated.mean(), x_control.mean()
    s1, s0 = x_treated.std(), x_control.std()
    s_pooled = np.sqrt(((s1**2) + (s0**2)) / 2.0)
    if s_pooled == 0 or np.isnan(s_pooled):
        return 0.0
    return float((m1 - m0) / s_pooled)


def compute_numeric_balance(
    df: pd.DataFrame,
    column_config: ColumnConfig,
    weight_col: Optional[str] = None,
    subset: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute numeric covariate balance (standardized mean differences) between
    treated and control groups, optionally using weights.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing covariates and treatment.
    column_config : ColumnConfig
        Configuration including numeric_features and treatment_col.
    weight_col : Optional[str]
        Optional column name with weights (e.g., IPW). If None, unweighted.
    subset : Optional[pd.Series]
        Optional boolean mask to restrict the analysis to a subpopulation
        (e.g., matched sample or common support region).

    Returns
    -------
    balance_df : pd.DataFrame
        DataFrame with per-feature means and standardized mean differences.
    """
    if column_config.numeric_features is None:
        raise ValueError("column_config.numeric_features must be set.")

    if subset is not None:
        df = df.loc[subset].copy()

    t_col = column_config.treatment_col
    numeric_cols = column_config.numeric_features

    results = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        if weight_col is None:
            x_t = df.loc[df[t_col] == 1, col]
            x_c = df.loc[df[t_col] == 0, col]
            smd = standardized_mean_difference(x_t, x_c)
            treated_mean = x_t.mean()
            control_mean = x_c.mean()
        else:
            w = df[weight_col]
            w_t = w[df[t_col] == 1]
            w_c = w[df[t_col] == 0]
            x_t = df.loc[df[t_col] == 1, col]
            x_c = df.loc[df[t_col] == 0, col]

            # Weighted means
            treated_mean = np.average(x_t, weights=w_t)
            control_mean = np.average(x_c, weights=w_c)

            # For SMD with weights, fall back to unweighted std for simplicity.
            smd = standardized_mean_difference(x_t, x_c)

        results.append(
            {
                "feature": col,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "std_mean_diff": smd,
            }
        )

    balance_df = pd.DataFrame(results).sort_values(
        "std_mean_diff", key=lambda x: np.abs(x), ascending=False
    )
    return balance_df


def summarize_treatment_outcome(
    df: pd.DataFrame,
    column_config: ColumnConfig,
) -> pd.DataFrame:
    """
    Provide a compact summary of outcome by treatment:
    - mean outcome
    - std
    - count
    - naive difference in means

    Useful as a quick non-causal baseline and as a reference point for
    more advanced causal estimators.
    """
    t_col = column_config.treatment_col
    y_col = column_config.outcome_col

    summary = (
        df.groupby(t_col)[y_col]
        .agg(["mean", "std", "count"])
        .rename(index={0: "control", 1: "treated"})
    )

    if "treated" in summary.index and "control" in summary.index:
        diff_in_means = summary.loc["treated", "mean"] - summary.loc["control", "mean"]
    else:
        diff_in_means = np.nan

    summary["diff_vs_control_mean"] = np.nan
    if "treated" in summary.index:
        summary.loc["treated", "diff_vs_control_mean"] = diff_in_means

    return summary



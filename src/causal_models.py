"""
Causal inference methods for estimating treatment effects.

Implements propensity score matching (via stratification) and inverse
propensity weighting. These methods help adjust for confounding in
observational studies.

Note: The "PSM" here is actually stratification on propensity score bins,
not true 1:1 matching. This is faster and often works just as well for
large datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .preprocessing import ColumnConfig, compute_numeric_balance
except ImportError:
    from preprocessing import ColumnConfig, compute_numeric_balance


@dataclass
class PropensityModelResult:
    """Container for fitted propensity score model and resulting datasets."""

    model: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame
    ps_col: str = "ps"


def fit_propensity_score_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    column_config: ColumnConfig,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    max_iter: int = 1000,
) -> PropensityModelResult:
    """
    Fit a logistic regression propensity score model using all pre-treatment features.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
        Train and test sets.
    column_config : ColumnConfig
        Configuration for column names and feature lists.
    numeric_features, categorical_features : Optional[List[str]]
        Optionally override feature lists; defaults to those in column_config.
    max_iter : int
        Max iterations for logistic regression.

    Returns
    -------
    PropensityModelResult
        Container with model and train/test DataFrames including a 'ps' column.
    """
    if numeric_features is None:
        numeric_features = column_config.numeric_features or []
    if categorical_features is None:
        categorical_features = column_config.categorical_features or []

    feature_cols = numeric_features + categorical_features

    train = df_train.dropna(
        subset=feature_cols + [column_config.treatment_col, column_config.outcome_col]
    ).copy()
    test = df_test.dropna(
        subset=feature_cols + [column_config.treatment_col, column_config.outcome_col]
    ).copy()

    X_train = train[feature_cols]
    T_train = train[column_config.treatment_col]
    X_test = test[feature_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    ps_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("logreg", LogisticRegression(max_iter=max_iter)),
        ]
    )

    ps_model.fit(X_train, T_train)

    train["ps"] = ps_model.predict_proba(X_train)[:, 1]
    test["ps"] = ps_model.predict_proba(X_test)[:, 1]

    return PropensityModelResult(model=ps_model, train=train, test=test, ps_col="ps")


def ps_stratified_ate(
    train: pd.DataFrame,
    column_config: ColumnConfig,
    ps_col: str = "ps",
    n_bins: int = 10,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate ATE via propensity score stratification (approximate matching).

    Steps:
    - Bin the propensity scores into quantile-based strata.
    - Within each stratum, compute treated-control outcome differences.
    - Aggregate bin-level ATEs weighted by bin sample sizes.
    - Compute covariate balance before and after restriction to usable strata.

    Returns
    -------
    ate : float
        Stratified ATE estimate.
    matched_summary : pd.DataFrame
        Per-bin sizes and ATEs.
    balance_before : pd.DataFrame
        Numeric covariate balance on full train.
    balance_after : pd.DataFrame
        Numeric covariate balance restricted to bins with both treated & control.
    """
    work = train.copy()
    if ps_col not in work.columns:
        raise ValueError(f"Propensity score column '{ps_col}' not found in train DataFrame.")

    work["ps_bin"] = pd.qcut(work[ps_col], q=n_bins, duplicates="drop")

    rows = []
    t_col = column_config.treatment_col
    y_col = column_config.outcome_col

    for ps_bin, group in work.groupby("ps_bin", observed=True):
        g_treated = group[group[t_col] == 1]
        g_control = group[group[t_col] == 0]
        if len(g_treated) == 0 or len(g_control) == 0:
            continue
        bin_ate = g_treated[y_col].mean() - g_control[y_col].mean()
        rows.append(
            {
                "ps_bin": ps_bin,
                "n_treated": len(g_treated),
                "n_control": len(g_control),
                "bin_ate": bin_ate,
            }
        )

    matched_summary = pd.DataFrame(rows)
    if matched_summary.empty:
        ate = np.nan
    else:
        matched_summary["bin_n"] = matched_summary["n_treated"] + matched_summary["n_control"]
        ate = float(
            np.average(matched_summary["bin_ate"], weights=matched_summary["bin_n"])
        )

    balance_before = compute_numeric_balance(work, column_config)

    valid_bins = matched_summary["ps_bin"].tolist()
    subset_mask = work["ps_bin"].isin(valid_bins)
    balance_after = compute_numeric_balance(work, column_config, subset=subset_mask)

    return ate, matched_summary, balance_before, balance_after


def ipw_ate(
    train: pd.DataFrame,
    column_config: ColumnConfig,
    ps_col: str = "ps",
    stabilized: bool = True,
    truncate_quantile: float = 0.99,
) -> Tuple[float, pd.DataFrame]:
    """
    Estimate ATE using inverse propensity weighting (IPW).

    Parameters
    ----------
    train : pd.DataFrame
        Training set with a propensity score column.
    column_config : ColumnConfig
        Configuration including treatment and outcome column names.
    ps_col : str
        Name of the propensity score column.
    stabilized : bool
        If True, use stabilized weights to reduce variance.
    truncate_quantile : float
        Upper quantile at which to truncate extreme weights.

    Returns
    -------
    ate_ipw : float
        IPW ATE estimate.
    balance_ipw : pd.DataFrame
        Numeric covariate balance after IPW weighting.
    """
    if ps_col not in train.columns:
        raise ValueError(f"Propensity score column '{ps_col}' not found in train DataFrame.")

    # Work with a copy for calculations, but update original DataFrame
    work = train.copy()
    eps = 1e-3
    ps = np.clip(work[ps_col].values, eps, 1 - eps)
    t = work[column_config.treatment_col].values
    y = work[column_config.outcome_col].values

    p_treated = t.mean()

    if stabilized:
        w_treat = p_treated * t / ps
        w_control = (1 - p_treated) * (1 - t) / (1 - ps)
    else:
        w_treat = t / ps
        w_control = (1 - t) / (1 - ps)

    w = w_treat + w_control

    if truncate_quantile is not None:
        max_w = np.quantile(w, truncate_quantile)
        w = np.minimum(w, max_w)

    # ATE formula using IPW
    ate_ipw = (
        np.sum(w * t * y / ps) / np.sum(w * t / ps)
        - np.sum(w * (1 - t) * y / (1 - ps)) / np.sum(w * (1 - t) / (1 - ps))
    )

    # Add weights to original DataFrame
    train["ipw_weight"] = w
    work["ipw_weight"] = w
    balance_ipw = compute_numeric_balance(work, column_config, weight_col="ipw_weight")

    return float(ate_ipw), balance_ipw



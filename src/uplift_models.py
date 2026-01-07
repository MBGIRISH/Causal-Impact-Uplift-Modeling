"""
Uplift modeling using the two-model approach.

Trains separate models for treated and control groups, then computes
individual treatment effects as the difference in predictions.

The segmentation function helps categorize users into actionable groups
for business decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

try:
    from .preprocessing import ColumnConfig
except ImportError:
    from preprocessing import ColumnConfig


@dataclass
class TwoModelResult:
    """Container for two-model uplift results."""

    model_treated: ClassifierMixin
    model_control: ClassifierMixin
    uplift_col: str = "uplift"
    p_treated_col: str = "p_treated"
    p_control_col: str = "p_control"


def train_two_model_uplift(
    df_train: pd.DataFrame,
    column_config: ColumnConfig,
    base_model_cls: Type[ClassifierMixin],
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    model_kwargs: Optional[dict] = None,
) -> Tuple[TwoModelResult, List[str]]:
    """
    Train two separate models:
    - One on treated users: predict P(Y=1 | T=1, X)
    - One on control users: predict P(Y=1 | T=0, X)

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data.
    column_config : ColumnConfig
        Column configuration.
    base_model_cls : Type[ClassifierMixin]
        Any scikit-learn classifier with predict_proba (e.g., GradientBoostingClassifier).
    numeric_features, categorical_features : Optional[List[str]]
        Feature lists; if None, use those from column_config.
    model_kwargs : Optional[dict]
        Extra keyword args passed to the base model constructor.

    Returns
    -------
    TwoModelResult
        Trained models and column naming convention.
    feature_cols : List[str]
        List of features used for training.
    """
    if numeric_features is None:
        numeric_features = column_config.numeric_features or []
    if categorical_features is None:
        categorical_features = column_config.categorical_features or []
    if model_kwargs is None:
        model_kwargs = {}

    feature_cols = numeric_features + categorical_features

    df = df_train.dropna(
        subset=feature_cols
        + [column_config.treatment_col, column_config.outcome_col]
    ).copy()

    t_col = column_config.treatment_col
    y_col = column_config.outcome_col

    treated = df[df[t_col] == 1]
    control = df[df[t_col] == 0]

    # Preprocess features (one-hot encode categoricals)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_treated_raw = treated[feature_cols]
    y_treated = treated[y_col]
    X_control_raw = control[feature_cols]
    y_control = control[y_col]

    # Transform features
    X_treated = preprocess.fit_transform(X_treated_raw)
    X_control = preprocess.fit_transform(X_control_raw)

    # Create pipelines with preprocessing
    model_treated = Pipeline([
        ("preprocess", preprocess),
        ("model", base_model_cls(**model_kwargs))
    ])
    model_control = Pipeline([
        ("preprocess", preprocess),
        ("model", base_model_cls(**model_kwargs))
    ])

    model_treated.fit(X_treated_raw, y_treated)
    model_control.fit(X_control_raw, y_control)

    result = TwoModelResult(
        model_treated=model_treated,
        model_control=model_control,
    )

    return result, feature_cols


def predict_two_model_uplift(
    df: pd.DataFrame,
    two_model_result: TwoModelResult,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute uplift scores on a dataset using a trained two-model result.

    Adds three columns:
    - p_treated_col: predicted P(Y=1 | T=1, X)
    - p_control_col: predicted P(Y=1 | T=0, X)
    - uplift_col:   p_treated_col - p_control_col
    """
    work = df.copy()

    X = work[feature_cols]
    # Models are pipelines, so they handle preprocessing automatically
    p1 = two_model_result.model_treated.predict_proba(X)[:, 1]
    p0 = two_model_result.model_control.predict_proba(X)[:, 1]

    work[two_model_result.p_treated_col] = p1
    work[two_model_result.p_control_col] = p0
    work[two_model_result.uplift_col] = p1 - p0

    return work


def segment_uplift_quadrants(
    df: pd.DataFrame,
    two_model_result: TwoModelResult,
    base_prob_col: str = "base_prob",
    persuadable_threshold: float = 0.0,
    sure_thing_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Segment users into:
    - Persuadables:     high uplift, low baseline probability
    - Sure-things:      low uplift, high baseline probability
    - Lost causes:      low uplift, low baseline probability
    - Do-not-disturb:   negative uplift

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already has uplift predictions.
    two_model_result : TwoModelResult
        Container with column names.
    base_prob_col : str
        Column name to store baseline (untreated) probability; if missing, it
        will be computed as the average of treated and control probabilities.
    persuadable_threshold : float
        Minimum uplift required to be considered positively influenced.
    sure_thing_threshold : float
        Threshold on baseline probability to qualify as a sure-thing.

    Returns
    -------
    df_segmented : pd.DataFrame
        Copy of df with an added 'uplift_segment' column.
    """
    work = df.copy()
    u_col = two_model_result.uplift_col
    p_t_col = two_model_result.p_treated_col
    p_c_col = two_model_result.p_control_col

    if base_prob_col not in work.columns:
        work[base_prob_col] = 0.5 * (work[p_t_col] + work[p_c_col])

    uplift = work[u_col]
    base_p = work[base_prob_col]

    segment = np.empty(len(work), dtype=object)

    # Do-not-disturb: negative uplift (campaign hurts)
    dnd_mask = uplift < 0
    segment[dnd_mask] = "do_not_disturb"

    # Persuadables: positive uplift AND low baseline probability
    persuadable_mask = (uplift >= persuadable_threshold) & (base_p < sure_thing_threshold)
    segment[persuadable_mask] = "persuadable"

    # Sure-things: low uplift, high baseline probability
    sure_mask = (uplift < persuadable_threshold) & (base_p >= sure_thing_threshold)
    segment[sure_mask] = "sure_thing"

    # Lost causes: remaining users (low uplift, low baseline)
    # Use object array comparison for None values
    lost_mask = np.array([s is None or s == "" for s in segment])
    segment[lost_mask] = "lost_cause"

    work["uplift_segment"] = segment
    return work



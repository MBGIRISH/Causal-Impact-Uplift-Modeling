"""
Evaluation metrics for uplift models.

Computes uplift and Qini curves to assess model performance. The Qini
coefficient is similar to AUC but specifically for incremental response.

These metrics help compare different targeting strategies and validate
that the model is actually finding users who respond to treatment.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _cumulative_gain(
    y: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal helper to compute cumulative incremental outcomes over a ranked list.

    Parameters
    ----------
    y : array-like
        Binary or continuous outcome.
    t : array-like
        Treatment indicator (0/1).
    uplift_scores : array-like
        Predicted uplift scores; higher means more likely to benefit.

    Returns
    -------
    frac : np.ndarray
        Fraction of population targeted.
    inc_gain : np.ndarray
        Cumulative incremental gain (treated - control baseline).
    random_gain : np.ndarray
        Expected gain under random targeting (linear baseline).
    """
    df = pd.DataFrame({"y": y, "t": t, "score": uplift_scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Running totals
    df["cum_n"] = np.arange(1, len(df) + 1)
    df["cum_treated"] = df["t"].cumsum()
    df["cum_control"] = df["cum_n"] - df["cum_treated"]

    df["cum_y_treated"] = (df["y"] * df["t"]).cumsum()
    df["cum_y_control"] = (df["y"] * (1 - df["t"])).cumsum()

    # Avoid division by zero
    eps = 1e-9
    treated_rate = df["cum_y_treated"] / np.maximum(df["cum_treated"], eps)
    control_rate = df["cum_y_control"] / np.maximum(df["cum_control"], eps)

    inc_gain = (treated_rate - control_rate) * df["cum_n"]
    frac = df["cum_n"] / len(df)

    # Random baseline: straight line from (0,0) to (1, max_gain)
    random_gain = inc_gain.iloc[-1] * frac

    return frac.values, inc_gain.values, random_gain.values


def uplift_and_qini(
    y: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute uplift and Qini curve quantities from predicted uplift scores.

    Parameters
    ----------
    y : array-like
        Outcome (binary or continuous).
    t : array-like
        Treatment indicator (0/1).
    uplift_scores : array-like
        Predicted uplift.

    Returns
    -------
    dict
        {
          'frac': fraction of population targeted,
          'uplift_curve': incremental gain curve,
          'qini_curve': Qini curve (uplift minus random baseline),
          'qini_coef': area under Qini curve (numeric),
        }
    """
    frac, inc_gain, random_gain = _cumulative_gain(y, t, uplift_scores)
    qini_curve = inc_gain - random_gain

    # Qini coefficient approximated by trapezoidal rule
    # Manual trapezoidal integration: sum of (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2
    if len(qini_curve) > 1:
        qini_coef = np.sum(
            (frac[1:] - frac[:-1]) * (qini_curve[1:] + qini_curve[:-1]) / 2
        )
    else:
        qini_coef = 0.0

    return {
        "frac": frac,
        "uplift_curve": inc_gain,
        "qini_curve": qini_curve,
        "qini_coef": qini_coef,
    }



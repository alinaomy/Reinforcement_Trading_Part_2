from __future__ import annotations

import numpy as np
import pandas as pd

from features import add_stationary_features


def assert_feature_stability_when_future_appended(
    df: pd.DataFrame,
    feature_cols: list[str],
    cut_index: int,
    atr_period: int = 14,
    rsi_period: int = 14,
    atol: float = 1e-10,
) -> bool:
    """Checks that features before cut_index do not change when future rows are present.

    This is a mechanical guardrail against accidental look-ahead.
    """
    past = df.iloc[:cut_index].copy()
    full = df.copy()

    feat_past, _ = add_stationary_features(past, atr_period=atr_period, rsi_period=rsi_period)
    feat_full, _ = add_stationary_features(full, atr_period=atr_period, rsi_period=rsi_period)

    common_idx = feat_past.index.intersection(feat_full.index)
    a = feat_past.loc[common_idx, feature_cols]
    b = feat_full.loc[common_idx, feature_cols]

    diff = (a - b).abs().max().max()
    if not np.isfinite(diff):
        diff = 0.0
    if diff > atol:
        raise AssertionError(f"Potential leakage: max feature change after appending future rows = {diff}")
    return True

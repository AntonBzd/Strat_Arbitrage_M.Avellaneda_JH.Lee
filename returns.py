from __future__ import annotations

import numpy as np
import pandas as pd


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change()
    return returns.dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")


def standardize_returns_window(returns_window: pd.DataFrame) -> pd.DataFrame:
    """
    Standardisation colonne par colonne :
    Y_ik = (R_ik - mean_i) / sigma_i
    """
    means = returns_window.mean(axis=0)
    stds = returns_window.std(axis=0, ddof=1)
    stds = stds.replace(0.0, np.nan)

    standardized = (returns_window - means) / stds
    standardized = standardized.dropna(axis=1, how="any")
    return standardized


def get_rolling_window(df: pd.DataFrame, end_idx: int, lookback: int) -> pd.DataFrame:
    start_idx = end_idx - lookback + 1
    if start_idx < 0:
        raise IndexError("Not enough history for requested window")
    return df.iloc[start_idx:end_idx + 1].copy()


def get_valid_assets_for_window(window: pd.DataFrame) -> pd.Index:
    return window.columns[window.notna().all(axis=0)]


def filter_complete_assets(window: pd.DataFrame) -> pd.DataFrame:
    valid_assets = get_valid_assets_for_window(window)
    return window[valid_assets].copy()
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


def get_logger(name: str = "stat_arb") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def ensure_series_datetime_index(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def sanitize_ticker(name: str) -> str:
    return (
        name.replace(".xlsx", "")
        .replace(".xls", "")
        .replace(" ", "_")
        .replace("-", "_")
        .upper()
    )


def make_output_dirs(base_dir: str | Path = "outputs") -> None:
    base = Path(base_dir)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "tables").mkdir(parents=True, exist_ok=True)
    (base / "cache").mkdir(parents=True, exist_ok=True)


def align_on_index(*objects: Iterable[pd.DataFrame | pd.Series]):
    common_index = None
    for obj in objects:
        idx = obj.index
        common_index = idx if common_index is None else common_index.intersection(idx)

    aligned = []
    for obj in objects:
        aligned.append(obj.loc[common_index])
    return aligned


def drop_columns_with_any_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.notna().all(axis=0)]


def assert_wide_price_panel(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty:
        raise ValueError("prices is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")
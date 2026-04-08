from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from utils import ensure_datetime_index, sanitize_ticker


def load_single_price_series(
    file_path: str | Path,
    ticker: Optional[str] = None,
    sheet_name=0,
    header: int = 5,
    date_col: str = "Date",
    price_col: str = "PX_LAST",
) -> pd.Series:
    """
    Charge un fichier Excel de type Bloomberg/export maison.
    Retourne une Series indexée par Date avec comme nom le ticker.
    """
    file_path = Path(file_path)

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header,
        usecols=[date_col, price_col],
    )

    df = df.rename(columns={date_col: "Date", price_col: "Price"})
    df = df.dropna(subset=["Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    series_name = ticker if ticker is not None else sanitize_ticker(file_path.stem)
    s = df["Price"].astype(float)
    s.name = series_name
    return s


def load_price_panel(file_map: Dict[str, str | Path], **kwargs) -> pd.DataFrame:
    """
    file_map = {"AD1": "data/raw/AD1_COMB_Curncy.xlsx", ...}
    """
    series_list = []
    for ticker, path in file_map.items():
        s = load_single_price_series(path, ticker=ticker, **kwargs)
        series_list.append(s)

    prices = pd.concat(series_list, axis=1)
    prices = ensure_datetime_index(prices)
    prices = prices.sort_index()
    return prices


def load_price_panel_from_folder(
    folder: str | Path,
    pattern: str = "*.xlsx",
    sheet_name=0,
    header: int = 5,
    date_col: str = "Date",
    price_col: str = "PX_LAST",
) -> pd.DataFrame:
    folder = Path(folder)
    file_paths = sorted(folder.glob(pattern))

    series_list = []
    for file_path in file_paths:
        ticker = sanitize_ticker(file_path.stem)
        s = load_single_price_series(
            file_path=file_path,
            ticker=ticker,
            sheet_name=sheet_name,
            header=header,
            date_col=date_col,
            price_col=price_col,
        )
        series_list.append(s)

    if not series_list:
        raise ValueError(f"No files found in {folder} with pattern {pattern}")

    prices = pd.concat(series_list, axis=1)
    prices = ensure_datetime_index(prices)
    return prices
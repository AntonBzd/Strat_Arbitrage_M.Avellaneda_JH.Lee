from __future__ import annotations

import numpy as np
import pandas as pd


def compute_total_return(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return np.nan
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def compute_annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return np.nan

    compounded = (1.0 + daily_returns).prod()
    n = len(daily_returns)
    return float(compounded ** (periods_per_year / n) - 1.0)


def compute_annualized_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return np.nan
    return float(daily_returns.std(ddof=1) * np.sqrt(periods_per_year))


def compute_sharpe(daily_returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return np.nan

    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=1)

    if vol == 0 or np.isnan(vol):
        return np.nan

    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    dd = compute_drawdown_series(equity_curve)
    return float(dd.min()) if not dd.empty else np.nan


def compute_hit_ratio(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return np.nan
    return float((daily_returns > 0).mean())


def summarize_backtest(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    turnover: pd.Series | None = None,
    nb_trades: int | None = None
) -> pd.Series:
    summary = {
        "total_return": compute_total_return(equity_curve),
        "annualized_return": compute_annualized_return(daily_returns),
        "annualized_vol": compute_annualized_vol(daily_returns),
        "sharpe": compute_sharpe(daily_returns),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "hit_ratio": compute_hit_ratio(daily_returns),
        "nb_trades": nb_trades
               }

    if turnover is not None and not turnover.empty:
        summary["avg_turnover"] = float(turnover.mean())
        summary["median_turnover"] = float(turnover.median())

    return pd.Series(summary, name="performance")
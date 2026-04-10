from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    alpha_daily: float
    alpha_annual: float
    betas: pd.Series
    fitted: pd.Series
    residuals: pd.Series
    x_process: pd.Series
    x_process_ou_window: pd.Series


def fit_factor_regression(
    stock_returns: pd.Series,
    factor_returns: pd.DataFrame,
    dt_years: float = 1.0 / 252.0,
    ou_window: int = None,
) -> RegressionResult:

    y = stock_returns.rename("stock").astype(float)
    X = factor_returns.astype(float).copy()

    aligned = pd.concat([y, X], axis=1).dropna()
    y = aligned["stock"]
    X = aligned.drop(columns=["stock"])

    if len(aligned) < X.shape[1] + 5:
        raise ValueError("Not enough observations for regression")

    X_design = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(X_design, y.values, rcond=None)

    alpha_daily = float(coef[0])
    betas = pd.Series(coef[1:], index=X.columns, dtype=float)

    fitted_values = X_design @ coef
    fitted = pd.Series(fitted_values, index=y.index, name="fitted")

    residuals = y - fitted
    residuals.name = "residuals"

    x_process = residuals.cumsum()
    x_process.name = "x_process"

    if ou_window is not None and ou_window < len(residuals):
        residuals_ou = residuals.iloc[-ou_window:]
        x_process_ou = residuals_ou.cumsum()
        x_process_ou.name = "x_process_ou"
    else:
        x_process_ou = x_process.copy()
        x_process_ou.name = "x_process_ou"

    return RegressionResult(
        alpha_daily=alpha_daily,
        alpha_annual=alpha_daily / dt_years,
        betas=betas,
        fitted=fitted,
        residuals=residuals,
        x_process=x_process,
        x_process_ou_window=x_process_ou,
    )


def fit_residual_models(
    stock_returns_window: pd.DataFrame,
    factor_returns_window: pd.DataFrame,
    dt_years: float = 1.0 / 252.0,
    ou_window: int = None,
) -> Dict[str, RegressionResult]:
    results: Dict[str, RegressionResult] = {}

    for ticker in stock_returns_window.columns:
        stock_series = stock_returns_window[ticker].dropna()

        try:
            res = fit_factor_regression(
                stock_returns=stock_series,
                factor_returns=factor_returns_window,
                dt_years=dt_years,
                ou_window=ou_window,
            )
            results[ticker] = res
        except Exception:
            continue

    return results

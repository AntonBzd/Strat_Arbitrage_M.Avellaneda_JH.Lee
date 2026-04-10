from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from regressions import RegressionResult


@dataclass
class OUEstimate:
    a: float
    b: float
    kappa: float
    m_raw: float
    sigma: float
    sigma_eq: float
    tau_days: float
    var_zeta: float
    current_x: float
    is_valid: bool


def fit_ou_from_x(
    x_series: pd.Series,
    dt_years: float = 1.0 / 252.0,
) -> OUEstimate:
    x = x_series.dropna().astype(float)

    if len(x) < 10:
        return OUEstimate(
            a=np.nan, b=np.nan, kappa=np.nan, m_raw=np.nan,
            sigma=np.nan, sigma_eq=np.nan, tau_days=np.nan,
            var_zeta=np.nan, current_x=np.nan, is_valid=False
        )

    x_next = x.iloc[1:]
    x_prev = x.iloc[:-1]
    x_prev.index = x_next.index

    X_design = np.column_stack([np.ones(len(x_prev)), x_prev.values])
    coef, *_ = np.linalg.lstsq(X_design, x_next.values, rcond=None)

    a = float(coef[0])
    b = float(coef[1])

    fitted = X_design @ coef
    zeta = x_next.values - fitted
    var_zeta = float(np.var(zeta, ddof=1)) if len(zeta) > 1 else np.nan

    is_valid = (
        np.isfinite(a)
        and np.isfinite(b)
        and np.isfinite(var_zeta)
        and var_zeta > 0
        and 0.0 < b < 1.0
    )

    if not is_valid:
        return OUEstimate(
            a=a, b=b, kappa=np.nan, m_raw=np.nan,
            sigma=np.nan, sigma_eq=np.nan, tau_days=np.nan,
            var_zeta=var_zeta, current_x=float(x.iloc[-1]), is_valid=False
        )

    kappa = -np.log(b) / dt_years
    if not np.isfinite(kappa) or kappa <= 0:
        return OUEstimate(
            a=a, b=b, kappa=np.nan, m_raw=np.nan,
            sigma=np.nan, sigma_eq=np.nan, tau_days=np.nan,
            var_zeta=var_zeta, current_x=float(x.iloc[-1]), is_valid=False
        )

    m_raw = a / (1.0 - b)
    sigma_eq = np.sqrt(var_zeta / (1.0 - b**2))
    sigma = np.sqrt(var_zeta * 2.0 * kappa / (1.0 - b**2))

    tau_days = 252.0 / kappa # Characteristic reversion time, not a true half-life.

    is_valid = (
        np.isfinite(m_raw)
        and np.isfinite(sigma_eq)
        and np.isfinite(sigma)
        and sigma_eq > 0
        and tau_days > 0
    )

    return OUEstimate(
        a=a, b=b, kappa=kappa, m_raw=m_raw,
        sigma=sigma, sigma_eq=sigma_eq, tau_days=tau_days,
        var_zeta=var_zeta, current_x=float(x.iloc[-1]), is_valid=is_valid,
    )


def _apply_bayesian_shrinkage(df: pd.DataFrame, shrinkage_strength: float) -> pd.DataFrame:
    valid = df["is_valid"].copy()
    lam = shrinkage_strength

    if valid.any():
        kappa_prior = df.loc[valid, "kappa"].median()
        df.loc[valid, "kappa"] = (1 - lam) * df.loc[valid, "kappa"] + lam * kappa_prior
        df.loc[valid, "tau_days"] = 252.0 / df.loc[valid, "kappa"]

    df.loc[valid, "m_raw"] = (1 - lam) * df.loc[valid, "m_raw"]

    if valid.any():
        seq_prior = df.loc[valid, "sigma_eq"].median()
        df.loc[valid, "sigma_eq"] = (1 - lam) * df.loc[valid, "sigma_eq"] + lam * seq_prior

    return df


def build_ou_signal_table(
    regression_results: Dict[str, RegressionResult],
    dt_years: float = 1.0 / 252.0,
    max_mean_reversion_days: float = 30.0,
    center_means_cross_sectionally: bool = True,
    use_bayesian_shrinkage: bool = False,
    shrinkage_strength: float = 0.3,
) -> pd.DataFrame:
    rows = []

    for ticker, reg in regression_results.items():
        ou = fit_ou_from_x(reg.x_process_ou_window, dt_years=dt_years)

        rows.append({
            "ticker": ticker,
            "alpha_daily": reg.alpha_daily,
            "alpha_annual": reg.alpha_annual,
            "a": ou.a,
            "b": ou.b,
            "kappa": ou.kappa,
            "m_raw": ou.m_raw,
            "sigma": ou.sigma,
            "sigma_eq": ou.sigma_eq,
            "tau_days": ou.tau_days,
            "var_zeta": ou.var_zeta,
            "current_x": ou.current_x,
            "is_valid": ou.is_valid,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")

    if use_bayesian_shrinkage and shrinkage_strength > 0:
        df = _apply_bayesian_shrinkage(df, shrinkage_strength)

    if center_means_cross_sectionally:
        mean_m = df.loc[df["is_valid"], "m_raw"].mean()
        df["m"] = df["m_raw"] - mean_m
    else:
        df["m"] = df["m_raw"]

    df["is_fast"] = df["tau_days"] < max_mean_reversion_days
    df["s_score"] = (df["current_x"] - df["m"]) / df["sigma_eq"]

    tau_years = df["tau_days"] / 252.0
    df["modified_s_score"] = (
        df["s_score"] - (df["alpha_annual"] * tau_years / df["sigma_eq"])
    )

    finite_cols = ["s_score", "modified_s_score", "sigma_eq", "kappa", "tau_days"]
    df["is_usable"] = df["is_valid"] & df["is_fast"]
    for col in finite_cols:
        df["is_usable"] &= np.isfinite(df[col])

    return df
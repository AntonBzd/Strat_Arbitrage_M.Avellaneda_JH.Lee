from __future__ import annotations
import numpy as np
import pandas as pd
from config import StrategyConfig


def next_position_from_score(
    current_position: int,
    score: float,
    config: StrategyConfig,
    kappa: float = np.nan,
    sigma_eq: float = np.nan,
) -> int:
    if not np.isfinite(score):
        return 0

    if config.use_divergence_stop:
        if abs(score) > config.divergence_stop_zscore:
            return 0

    # Compute effective thresholds
    sbo, sso, sbc, ssc = _get_thresholds(config, kappa)

    # Flat -> potentially open
    if current_position == 0:
        if score < -sbo:
            return 1
        if score > sso:
            return -1
        return 0

    # Long -> potentially close
    if current_position > 0:
        if score > -ssc:
            if config.allow_reentry_same_day and score > sso:
                return -1
            return 0
        return 1

    # Short -> potentially close
    if current_position < 0:
        if score < sbc:
            if config.allow_reentry_same_day and score < -sbo:
                return 1
            return 0
        return -1

    return 0


def _get_thresholds(
    config: StrategyConfig,
    kappa: float,
) -> tuple[float, float, float, float]:

    if not config.use_adaptive_thresholds or not np.isfinite(kappa):
        return config.sbo, config.sso, config.sbc, config.ssc

    kappa_median = 40.0
    ratio = np.clip(kappa / kappa_median, 0.5, 2.0)
    scale = 1.0 / (ratio ** config.adaptive_kappa_scale)  # >1 if slow, <1 if fast

    sbo = config.adaptive_base_sbo * scale
    sso = config.adaptive_base_sbo * scale
    sbc = config.sbc * scale
    ssc = config.ssc * scale

    return sbo, sso, sbc, ssc


def generate_target_states(
    score_series: pd.Series,
    current_states: pd.Series,
    config: StrategyConfig,
    ou_table: pd.DataFrame = None,
) -> pd.Series:

    target = current_states.copy()

    for ticker in current_states.index:
        score = score_series.get(ticker, np.nan)
        current_position = int(current_states.loc[ticker])

        kappa = np.nan
        sigma_eq = np.nan
        if ou_table is not None and ticker in ou_table.index:
            kappa = ou_table.loc[ticker, "kappa"]
            sigma_eq = ou_table.loc[ticker, "sigma_eq"]

        target.loc[ticker] = next_position_from_score(
            current_position=current_position,
            score=score,
            config=config,
            kappa=kappa,
            sigma_eq=sigma_eq,
        )

    return target.astype(int)

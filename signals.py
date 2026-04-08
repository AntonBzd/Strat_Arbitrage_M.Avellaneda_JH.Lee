from __future__ import annotations
import numpy as np
import pandas as pd
from config import StrategyConfig


def next_position_from_score(
    current_position: int,
    score: float,
    config: StrategyConfig,
) -> int:
    if not np.isfinite(score):
        return 0

    # Flat -> potentially open
    if current_position == 0:
        if score < -config.sbo:
            return 1
        if score > config.sso:
            return -1
        return 0

    # Long -> potentially close
    if current_position > 0:
        if score > -config.ssc:
            if config.allow_reentry_same_day and score > config.sso:
                return -1
            return 0
        return 1

    # Short -> potentially close
    if current_position < 0:
        if score < config.sbc:
            if config.allow_reentry_same_day and score < -config.sbo:
                return 1
            return 0
        return -1

    return 0


def generate_target_states(
    score_series: pd.Series,
    current_states: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    target = current_states.copy()

    for ticker in current_states.index:
        score = score_series.get(ticker, np.nan)
        current_position = int(current_states.loc[ticker])
        target.loc[ticker] = next_position_from_score(
            current_position=current_position,
            score=score,
            config=config,
        )

    return target.astype(int)
from __future__ import annotations

import pandas as pd


def states_to_weights(
    states: pd.Series,
    long_leverage: float = 2.0,
    short_leverage: float = 2.0,
) -> pd.Series:
    """
    Transforme les états {-1, 0, +1} en poids neutres.
    """
    weights = pd.Series(0.0, index=states.index, dtype=float)

    longs = states[states > 0].index
    shorts = states[states < 0].index

    if len(longs) > 0:
        weights.loc[longs] = long_leverage / len(longs)

    if len(shorts) > 0:
        weights.loc[shorts] = -short_leverage / len(shorts)

    return weights


def compute_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    aligned_old, aligned_new = old_weights.align(new_weights, fill_value=0.0)
    return float((aligned_new - aligned_old).abs().sum())


def summarize_book(states: pd.Series, weights: pd.Series) -> dict:
    return {
        "n_longs": int((states > 0).sum()),
        "n_shorts": int((states < 0).sum()),
        "gross_long": float(weights[weights > 0].sum()),
        "gross_short": float(-weights[weights < 0].sum()),
        "net": float(weights.sum()),
        "gross": float(weights.abs().sum()),
    }


def compute_trade_count(old_states: pd.Series, new_states: pd.Series) -> int:
    aligned_old, aligned_new = old_states.align(new_states, fill_value=0)

    old_states_int = aligned_old.astype(int)
    new_states_int = aligned_new.astype(int)

    entries = ((old_states_int == 0) & (new_states_int != 0)).sum()
    exits = ((old_states_int != 0) & (new_states_int == 0)).sum()
    flips = ((old_states_int * new_states_int) == -1).sum()

    return int(entries + exits + 2 * flips)
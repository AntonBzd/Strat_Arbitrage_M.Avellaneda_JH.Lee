from __future__ import annotations

import numpy as np
import pandas as pd


def build_event_driven_raw_weights(
    current_weights: pd.Series,
    current_states: pd.Series,
    new_states: pd.Series,
    scores: pd.Series,
    entry_weight_mode: str = "constant",
    entry_weight_value: float = 0.05,
    score_cap: float = 3.0,
    rebalance_held_positions: bool = False,
) -> pd.Series:

    aligned_current_w = current_weights.reindex(new_states.index).fillna(0.0).astype(float)
    aligned_old_s = current_states.reindex(new_states.index).fillna(0).astype(int)
    aligned_new_s = new_states.reindex(new_states.index).fillna(0).astype(int)
    aligned_scores = scores.reindex(new_states.index).astype(float)

    w = aligned_current_w.copy()

    exits = (aligned_old_s != 0) & (aligned_new_s == 0)
    flips = (aligned_old_s * aligned_new_s) == -1
    new_entries = (aligned_old_s == 0) & (aligned_new_s != 0)

    w.loc[exits | flips] = 0.0

    if rebalance_held_positions:
        held_same_sign = (aligned_old_s == aligned_new_s) & (aligned_new_s != 0)
        if held_same_sign.any():
            if entry_weight_mode == "constant":
                w.loc[held_same_sign] = entry_weight_value * aligned_new_s.loc[held_same_sign]
            else:
                mag = aligned_scores.loc[held_same_sign].abs().clip(upper=score_cap).fillna(1.0)
                w.loc[held_same_sign] = entry_weight_value * mag * aligned_new_s.loc[held_same_sign]

    opens = new_entries | flips
    if opens.any():
        if entry_weight_mode == "constant":
            w.loc[opens] = entry_weight_value * aligned_new_s.loc[opens]
        else:
            mag = aligned_scores.loc[opens].abs().clip(upper=score_cap).fillna(1.0)
            w.loc[opens] = entry_weight_value * mag * aligned_new_s.loc[opens]

    return w.astype(float)


def build_beta_matrix(
    regression_results: dict,
    assets: list[str],
) -> pd.DataFrame:


    factor_names = None
    for _, res in regression_results.items():
        factor_names = list(res.betas.index)
        break

    if factor_names is None:
        return pd.DataFrame(index=[], columns=assets, dtype=float)

    beta_mat = pd.DataFrame(0.0, index=factor_names, columns=assets, dtype=float)
    for asset in assets:
        if asset in regression_results:
            beta_mat.loc[:, asset] = regression_results[asset].betas.reindex(factor_names).fillna(0.0)

    return beta_mat


def project_to_factor_neutral(
    raw_weights: pd.Series,
    beta_matrix: pd.DataFrame,
    neutralize_net_exposure: bool = True,
    ridge: float = 1e-8,
) -> pd.Series:
    """
    Orthogonal projection of raw stock weights onto the constraint set.
    """
    assets = list(raw_weights.index)
    q = raw_weights.reindex(assets).fillna(0.0).astype(float).values

    rows = []

    if neutralize_net_exposure:
        rows.append(np.ones(len(assets), dtype=float))

    if not beta_matrix.empty:
        B = beta_matrix.reindex(columns=assets).fillna(0.0).astype(float).values
        for k in range(B.shape[0]):
            rows.append(B[k])

    if len(rows) == 0:
        return raw_weights.copy().astype(float)

    A = np.vstack(rows)  # constraints in row form
    M = A @ A.T + ridge * np.eye(A.shape[0], dtype=float)
    adjustment = A.T @ np.linalg.solve(M, A @ q)
    q_proj = q - adjustment

    return pd.Series(q_proj, index=assets, dtype=float)


def scale_book_to_target_gross(
    weights: pd.Series,
    target_gross: float,
    max_single_position: float = 0.0,
) -> pd.Series:
    w = weights.copy().astype(float)

    gross = float(w.abs().sum())
    if gross > 0 and target_gross > 0:
        w *= target_gross / gross

    if max_single_position > 0:
        w = w.clip(lower=-max_single_position, upper=max_single_position)
        gross_after_clip = float(w.abs().sum())
        if gross_after_clip > 0 and gross_after_clip < target_gross:
            # do a global rescale only if it does not violate the cap after re-clipping
            scale = target_gross / gross_after_clip
            w = (w * scale).clip(lower=-max_single_position, upper=max_single_position)

    return w


def apply_partial_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rebalance_fraction: float = 1.0,
) -> pd.Series:
    aligned_old, aligned_new = current_weights.align(target_weights, fill_value=0.0)
    out = aligned_old + rebalance_fraction * (aligned_new - aligned_old)
    return out.astype(float)


def apply_no_trade_band(
    current_weights: pd.Series,
    target_weights: pd.Series,
    band: float = 0.0025,
) -> pd.Series:
    aligned_old, aligned_new = current_weights.align(target_weights, fill_value=0.0)
    delta = aligned_new - aligned_old
    out = aligned_new.copy()
    out.loc[delta.abs() < band] = aligned_old.loc[delta.abs() < band]
    return out.astype(float)


def compute_factor_exposure(
    weights: pd.Series,
    beta_matrix: pd.DataFrame,
) -> pd.Series:
    if beta_matrix.empty:
        return pd.Series(dtype=float)

    assets = list(weights.index)
    B = beta_matrix.reindex(columns=assets).fillna(0.0)
    return B @ weights.reindex(assets).fillna(0.0)


def compute_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    aligned_old, aligned_new = old_weights.align(new_weights, fill_value=0.0)
    return float((aligned_new - aligned_old).abs().sum())


def compute_market_impact_cost(
    old_weights: pd.Series,
    new_weights: pd.Series,
    daily_vol: pd.Series,
    impact_coefficient: float = 0.1,
    participation_rate: float = 0.01,
) -> float:
    aligned_old, aligned_new = old_weights.align(new_weights, fill_value=0.0)
    delta = (aligned_new - aligned_old).abs()

    fallback = float(daily_vol.median()) if len(daily_vol) > 0 else 0.02
    vol = daily_vol.reindex(delta.index).fillna(fallback)
    per_stock_cost = impact_coefficient * vol * np.sqrt(participation_rate) * delta
    return float(per_stock_cost.sum())


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
    old_int = aligned_old.astype(int)
    new_int = aligned_new.astype(int)

    entries = ((old_int == 0) & (new_int != 0)).sum()
    exits = ((old_int != 0) & (new_int == 0)).sum()
    flips = ((old_int * new_int) == -1).sum()

    return int(entries + exits + 2 * flips)
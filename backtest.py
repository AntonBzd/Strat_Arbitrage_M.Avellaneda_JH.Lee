from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from config import StrategyConfig
from returns import compute_simple_returns, get_rolling_window, filter_complete_assets
from pca_factors import run_pca_factor_model
from regressions import fit_residual_models
from ou_model import build_ou_signal_table
from signals import generate_target_states
from portfolio import (
    build_event_driven_raw_weights,
    build_beta_matrix,
    project_to_factor_neutral,
    scale_book_to_target_gross,
    apply_partial_rebalance,
    apply_no_trade_band,
    compute_factor_exposure,
    compute_turnover,
    compute_market_impact_cost,
    summarize_book,
    compute_trade_count,
)
from utils import assert_wide_price_panel, ensure_datetime_index


@dataclass
class DailySnapshot:
    date: pd.Timestamp
    next_date: pd.Timestamp
    universe: list[str]
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    eigenvectors: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_returns_window: pd.DataFrame
    stock_returns_window: pd.DataFrame
    residuals_window: pd.DataFrame
    x_process_window: pd.DataFrame
    ou_table: pd.DataFrame
    full_score: pd.Series
    target_states: pd.Series
    target_weights: pd.Series
    next_day_asset_contribution: pd.Series
    next_day_long_contribution: float
    next_day_short_contribution: float
    next_day_total_gross_contribution: float
    next_day_turnover: float
    next_day_net_return: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights_history: pd.DataFrame
    states_history: pd.DataFrame
    score_history: pd.DataFrame
    turnover: pd.Series
    diagnostics: pd.DataFrame
    raw_returns: pd.DataFrame
    snapshots: Dict[pd.Timestamp, DailySnapshot]
    long_contribution: pd.Series
    short_contribution: pd.Series
    gross_contribution: pd.Series
    contribution_by_asset: pd.DataFrame
    trade_count: pd.Series
    total_trade_count: int


def _should_store_snapshot(
    date: pd.Timestamp,
    config: StrategyConfig,
    snapshot_date_set: Optional[set[pd.Timestamp]],
) -> bool:
    if not config.store_snapshots:
        return False
    if snapshot_date_set is not None and date not in snapshot_date_set:
        return False
    if config.snapshot_start is not None and date < pd.Timestamp(config.snapshot_start):
        return False
    if config.snapshot_end is not None and date > pd.Timestamp(config.snapshot_end):
        return False
    return True


def _build_regression_windows(regression_results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals_window_df = pd.DataFrame({
        ticker: res.residuals for ticker, res in regression_results.items()
    }).sort_index()

    # use the OU sub-window x_process
    x_process_window_df = pd.DataFrame({
        ticker: res.x_process_ou_window for ticker, res in regression_results.items()
    }).sort_index()

    return residuals_window_df, x_process_window_df


def run_backtest(prices: pd.DataFrame, config: StrategyConfig) -> BacktestResult:
    config.validate()
    assert_wide_price_panel(prices)

    prices = ensure_datetime_index(prices)
    returns = compute_simple_returns(prices)

    all_assets = returns.columns.tolist()
    current_states = pd.Series(0, index=all_assets, dtype=int)
    current_weights = pd.Series(0.0, index=all_assets, dtype=float)

    score_history: dict[pd.Timestamp, pd.Series] = {}
    state_history: dict[pd.Timestamp, pd.Series] = {}
    weight_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}
    diagnostics_rows: list[dict[str, Any]] = []
    snapshots: dict[pd.Timestamp, DailySnapshot] = {}

    long_contribution_history: dict[pd.Timestamp, float] = {}
    short_contribution_history: dict[pd.Timestamp, float] = {}
    gross_contribution_history: dict[pd.Timestamp, float] = {}
    contribution_by_asset_history: dict[pd.Timestamp, pd.Series] = {}

    equity = config.initial_capital
    equity_curve: dict[pd.Timestamp, float] = {}
    daily_strategy_returns: dict[pd.Timestamp, float] = {}
    trade_count_history: dict[pd.Timestamp, int] = {}

    start_idx = max(config.pca_window, config.regression_window) - 1
    if len(returns) <= start_idx + 1:
        raise ValueError("Not enough data to run the backtest with the chosen windows")

    snapshot_date_set = None
    if config.snapshot_dates is not None:
        snapshot_date_set = {pd.Timestamp(d) for d in config.snapshot_dates}

    rolling_vol = returns.rolling(60).std() if config.use_market_impact else None

    equity_curve[returns.index[start_idx]] = equity

    for t_idx in range(start_idx, len(returns) - 1):
        date = returns.index[t_idx]
        next_date = returns.index[t_idx + 1]

        # PCA estimation
        pca_window = get_rolling_window(returns, end_idx=t_idx, lookback=config.pca_window)
        pca_window = filter_complete_assets(pca_window)

        if pca_window.shape[1] < config.min_assets:
            new_states = pd.Series(0, index=all_assets, dtype=int)
            new_weights = pd.Series(0.0, index=all_assets, dtype=float)

            trade_count = compute_trade_count(current_states, new_states)
            turnover = compute_turnover(current_weights, new_weights)
            next_ret = returns.iloc[t_idx + 1].reindex(all_assets).fillna(0.0)

            asset_contrib = (new_weights * next_ret).astype(float)
            long_contrib = float(asset_contrib[new_weights > 0].sum())
            short_contrib = float(asset_contrib[new_weights < 0].sum())
            gross_ret = float(asset_contrib.sum())

            if config.use_market_impact and rolling_vol is not None:
                daily_vol = rolling_vol.iloc[t_idx].reindex(all_assets).fillna(0.02)
                impact_cost = compute_market_impact_cost(
                    current_weights,
                    new_weights,
                    daily_vol,
                    impact_coefficient=config.impact_coefficient,
                    participation_rate=config.typical_participation_rate,
                )
            else:
                impact_cost = config.slippage * turnover

            net_ret = gross_ret - impact_cost
            equity *= (1.0 + net_ret)

            state_history[date] = new_states
            weight_history[date] = new_weights
            score_history[date] = pd.Series(np.nan, index=all_assets, dtype=float)
            turnover_history[date] = turnover
            trade_count_history[date] = trade_count

            long_contribution_history[next_date] = long_contrib
            short_contribution_history[next_date] = short_contrib
            gross_contribution_history[next_date] = gross_ret
            contribution_by_asset_history[next_date] = asset_contrib
            daily_strategy_returns[next_date] = net_ret
            equity_curve[next_date] = equity

            diagnostics_rows.append({
                "date": date,
                "n_assets": int(pca_window.shape[1]),
                "n_factors": np.nan,
                "selected_variance": np.nan,
                "n_longs": 0,
                "n_shorts": 0,
                "gross_long": 0.0,
                "gross_short": 0.0,
                "gross": 0.0,
                "net": 0.0,
                "turnover": turnover,
                "trade_count": trade_count,
                "impact_cost": impact_cost,
                "max_abs_factor_exposure": 0.0,
            })

            current_states = new_states
            current_weights = new_weights
            continue

        # PCA model
        pca_res = run_pca_factor_model(
            returns_window=pca_window,
            mode=config.pca_mode,
            n_factors_fixed=config.n_factors_fixed,
            explained_variance_target=config.explained_variance_target,
        )

        regression_lookback = min(config.regression_window, pca_window.shape[0])
        regression_stock_window = pca_window.iloc[-regression_lookback:].copy()
        factor_returns_regression = pca_res.factor_returns.iloc[-regression_lookback:].copy()

        common_assets = regression_stock_window.columns.intersection(pca_res.factor_weights.index)
        regression_stock_window = regression_stock_window[common_assets]

        regression_results = fit_residual_models(
            stock_returns_window=regression_stock_window,
            factor_returns_window=factor_returns_regression,
            dt_years=config.dt_years,
            ou_window=config.ou_window,
        )

        residuals_window_df, x_process_window_df = _build_regression_windows(regression_results)

        ou_table = build_ou_signal_table(
            regression_results=regression_results,
            dt_years=config.dt_years,
            max_mean_reversion_days=config.max_mean_reversion_days,
            center_means_cross_sectionally=True,
            use_bayesian_shrinkage=config.use_bayesian_shrinkage,
            shrinkage_strength=config.shrinkage_strength,
        )

        # Scores / signals
        full_score = pd.Series(np.nan, index=all_assets, dtype=float)
        if not ou_table.empty:
            score_col = "modified_s_score" if config.use_modified_score else "s_score"
            usable = ou_table[ou_table["is_usable"]].copy()
            if not usable.empty:
                full_score.loc[usable.index] = usable[score_col]

        new_states = generate_target_states(
            score_series=full_score,
            current_states=current_states,
            config=config,
            ou_table=ou_table if config.use_adaptive_thresholds else None,
        )

        trade_count = compute_trade_count(current_states, new_states)

        # Raw weights
        use_event_driven_weights = getattr(config, "use_event_driven_weights", True)
        entry_weight_mode = getattr(config, "entry_weight_mode", "constant")
        entry_weight_value = getattr(config, "entry_weight_value", 0.05)
        rebalance_held_positions = getattr(config, "rebalance_held_positions", False)

        if use_event_driven_weights:
            raw_weights = build_event_driven_raw_weights(
                current_weights=current_weights.reindex(all_assets).fillna(0.0),
                current_states=current_states.reindex(all_assets).fillna(0).astype(int),
                new_states=new_states.reindex(all_assets).fillna(0).astype(int),
                scores=full_score.reindex(all_assets),
                entry_weight_mode=entry_weight_mode,
                entry_weight_value=entry_weight_value,
                score_cap=config.signal_weight_cap,
                rebalance_held_positions=rebalance_held_positions,
            ).reindex(all_assets).fillna(0.0)
        else:
            # fallback if you deliberately want old behavior
            raw_weights = pd.Series(0.0, index=all_assets, dtype=float)
            longs = new_states[new_states > 0].index
            shorts = new_states[new_states < 0].index
            if len(longs) > 0:
                raw_weights.loc[longs] = config.long_leverage / len(longs)
            if len(shorts) > 0:
                raw_weights.loc[shorts] = -config.short_leverage / len(shorts)

        # Factor-neutral adjustment
        use_factor_neutral_overlay = getattr(config, "use_factor_neutral_overlay", True)
        neutralize_net_exposure = getattr(config, "neutralize_net_exposure", True)
        factor_neutral_ridge = getattr(config, "factor_neutral_ridge", 1e-8)

        beta_matrix = build_beta_matrix(
            regression_results=regression_results,
            assets=all_assets,
        )

        if use_factor_neutral_overlay:
            target_weights = project_to_factor_neutral(
                raw_weights=raw_weights,
                beta_matrix=beta_matrix,
                neutralize_net_exposure=neutralize_net_exposure,
                ridge=factor_neutral_ridge,
            )
        else:
            target_weights = raw_weights.copy()

        # Size the post projection book
        if config.gross_sizing_mode == "target":
            gross_target = config.long_leverage + config.short_leverage
            target_weights = scale_book_to_target_gross(
                weights=target_weights,
                target_gross=gross_target,
                max_single_position=config.max_single_position,
            ).reindex(all_assets).fillna(0.0)

        elif config.gross_sizing_mode == "natural":
            target_weights = target_weights.clip(
                lower=-config.max_single_position,
                upper=config.max_single_position,
            ).reindex(all_assets).fillna(0.0)

        else:
            raise ValueError(f"Unknown gross_sizing_mode: {config.gross_sizing_mode}")


        # Smooth / no-trade band
        hedge_rebalance_fraction = getattr(config, "hedge_rebalance_fraction", 1.0)
        target_weights = apply_partial_rebalance(
            current_weights=current_weights.reindex(all_assets).fillna(0.0),
            target_weights=target_weights,
            rebalance_fraction=hedge_rebalance_fraction,
        ).reindex(all_assets).fillna(0.0)

        use_no_trade_band = getattr(config, "use_no_trade_band", True)
        no_trade_band = getattr(config, "no_trade_band", 0.0025)

        if use_no_trade_band:
            new_weights = apply_no_trade_band(
                current_weights=current_weights.reindex(all_assets).fillna(0.0),
                target_weights=target_weights,
                band=no_trade_band,
            ).reindex(all_assets).fillna(0.0)
        else:
            new_weights = target_weights.copy()

        turnover = compute_turnover(current_weights, new_weights)

        final_factor_exposure = compute_factor_exposure(new_weights, beta_matrix)
        max_abs_factor_exposure = (
            float(final_factor_exposure.abs().max())
            if len(final_factor_exposure) > 0 else 0.0
        )

        # Costs
        if config.use_market_impact and rolling_vol is not None:
            daily_vol = rolling_vol.iloc[t_idx].reindex(all_assets).fillna(0.02)
            impact_cost = compute_market_impact_cost(
                current_weights,
                new_weights,
                daily_vol,
                impact_coefficient=config.impact_coefficient,
                participation_rate=config.typical_participation_rate,
            )
        else:
            impact_cost = config.slippage * turnover

        # Next-day PnL
        next_ret = returns.iloc[t_idx + 1].reindex(all_assets).fillna(0.0)
        asset_contrib = (new_weights * next_ret).astype(float)

        long_contrib = float(asset_contrib[new_weights > 0].sum())
        short_contrib = float(asset_contrib[new_weights < 0].sum())
        gross_ret = float(asset_contrib.sum())
        net_ret = gross_ret - impact_cost

        equity *= (1.0 + net_ret)

        # Store histories
        state_history[date] = new_states
        weight_history[date] = new_weights
        score_history[date] = full_score
        turnover_history[date] = turnover

        long_contribution_history[next_date] = long_contrib
        short_contribution_history[next_date] = short_contrib
        gross_contribution_history[next_date] = gross_ret
        contribution_by_asset_history[next_date] = asset_contrib
        daily_strategy_returns[next_date] = net_ret
        equity_curve[next_date] = equity
        trade_count_history[date] = trade_count

        # Diagnostics
        book = summarize_book(new_states, new_weights)
        selected_variance = float(
            pca_res.cumulative_explained_variance[pca_res.n_factors - 1]
        )

        diagnostics_row = {
            "date": date,
            "n_assets": int(pca_window.shape[1]),
            "n_factors": int(pca_res.n_factors),
            "selected_variance": selected_variance,
            "n_longs": book["n_longs"],
            "n_shorts": book["n_shorts"],
            "gross_long": book["gross_long"],
            "gross_short": book["gross_short"],
            "gross": book["gross"],
            "net": book["net"],
            "turnover": turnover,
            "trade_count": trade_count,
            "impact_cost": impact_cost,
            "max_abs_factor_exposure": max_abs_factor_exposure,
        }

        for factor_name, exp_value in final_factor_exposure.items():
            diagnostics_row[f"beta_{factor_name}"] = float(exp_value)

        diagnostics_rows.append(diagnostics_row)

        # Snapshot
        if _should_store_snapshot(date, config, snapshot_date_set):
            factor_returns_window = pca_res.factor_returns.iloc[-config.ou_window:].copy()
            ou_stock_window = pca_window.iloc[-config.ou_window:][common_assets].copy()

            snapshots[date] = DailySnapshot(
                date=date,
                next_date=next_date,
                universe=list(common_assets),
                correlation_matrix=pca_res.correlation_matrix.copy(),
                eigenvalues=pca_res.eigenvalues.copy(),
                eigenvectors=pca_res.eigenvectors.copy(),
                factor_weights=pca_res.factor_weights.copy(),
                factor_returns_window=factor_returns_window,
                stock_returns_window=ou_stock_window,
                residuals_window=residuals_window_df.copy(),
                x_process_window=x_process_window_df.copy(),
                ou_table=ou_table.copy(),
                full_score=full_score.copy(),
                target_states=new_states.copy(),
                target_weights=new_weights.copy(),
                next_day_asset_contribution=asset_contrib.copy(),
                next_day_long_contribution=long_contrib,
                next_day_short_contribution=short_contrib,
                next_day_total_gross_contribution=gross_ret,
                next_day_turnover=turnover,
                next_day_net_return=net_ret,
            )

        current_states = new_states
        current_weights = new_weights

    # Results
    equity_curve = pd.Series(equity_curve).sort_index()
    daily_returns = pd.Series(daily_strategy_returns).sort_index()
    turnover = pd.Series(turnover_history).sort_index()
    states_history = pd.DataFrame(state_history).T.sort_index()
    weights_history = pd.DataFrame(weight_history).T.sort_index()
    score_history = pd.DataFrame(score_history).T.sort_index()
    diagnostics = pd.DataFrame(diagnostics_rows).set_index("date").sort_index()
    long_contribution = pd.Series(long_contribution_history).sort_index()
    short_contribution = pd.Series(short_contribution_history).sort_index()
    gross_contribution = pd.Series(gross_contribution_history).sort_index()
    contribution_by_asset = pd.DataFrame(contribution_by_asset_history).T.sort_index()
    trade_count = pd.Series(trade_count_history).sort_index()
    total_trade_count = int(trade_count.sum())

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights_history=weights_history,
        states_history=states_history,
        score_history=score_history,
        turnover=turnover,
        diagnostics=diagnostics,
        raw_returns=returns,
        snapshots=snapshots,
        long_contribution=long_contribution,
        short_contribution=short_contribution,
        gross_contribution=gross_contribution,
        contribution_by_asset=contribution_by_asset,
        trade_count=trade_count,
        total_trade_count=total_trade_count,
    )
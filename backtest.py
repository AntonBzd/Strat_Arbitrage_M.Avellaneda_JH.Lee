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
from portfolio import states_to_weights, compute_turnover, summarize_book, compute_trade_count
from utils import assert_wide_price_panel, ensure_datetime_index


@dataclass
class DailySnapshot:
    date: pd.Timestamp
    next_date: pd.Timestamp
    universe: list[str]

    # PCA block
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    eigenvectors: pd.DataFrame
    factor_weights: pd.DataFrame
    factor_returns_window: pd.DataFrame
    stock_returns_window: pd.DataFrame

    # Regression / OU block
    residuals_window: pd.DataFrame
    x_process_window: pd.DataFrame
    ou_table: pd.DataFrame

    # Signal / portfolio block
    full_score: pd.Series
    target_states: pd.Series
    target_weights: pd.Series

    # Next-day realized contribution
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

    # Diagnostics / introspection
    snapshots: Dict[pd.Timestamp, DailySnapshot]

    # Contribution analysis
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

    x_process_window_df = pd.DataFrame({
        ticker: res.x_process for ticker, res in regression_results.items()
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

    start_idx = max(config.pca_window, config.ou_window) - 1
    if len(returns) <= start_idx + 1:
        raise ValueError("Not enough data to run the backtest with the chosen windows")

    snapshot_date_set = None
    if config.snapshot_dates is not None:
        snapshot_date_set = {pd.Timestamp(d) for d in config.snapshot_dates}

    equity_curve[returns.index[start_idx]] = equity

    for t_idx in range(start_idx, len(returns) - 1):
        date = returns.index[t_idx]
        next_date = returns.index[t_idx + 1]

        # ----------------------------
        # 1) PCA estimation window
        # ----------------------------
        pca_window = get_rolling_window(returns, end_idx=t_idx, lookback=config.pca_window)
        pca_window = filter_complete_assets(pca_window)

        # Branch: universe too small -> flat book
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
            net_ret = gross_ret - config.slippage * turnover

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
                "gross": 0.0,
                "turnover": turnover,
                "trade_count": trade_count,
            })

            current_states = new_states
            current_weights = new_weights
            continue

        # ----------------------------
        # 2) PCA block
        # ----------------------------
        pca_res = run_pca_factor_model(
            returns_window=pca_window,
            mode=config.pca_mode,
            n_factors_fixed=config.n_factors_fixed,
            explained_variance_target=config.explained_variance_target,
        )

        # ----------------------------
        # 3) Last 60 days for regression / OU block
        # ----------------------------
        ou_stock_window = pca_window.iloc[-config.ou_window:].copy()
        factor_returns_window = pca_res.factor_returns.iloc[-config.ou_window:].copy()

        common_assets = ou_stock_window.columns.intersection(pca_res.factor_weights.index)
        ou_stock_window = ou_stock_window[common_assets]

        regression_results = fit_residual_models(
            stock_returns_window=ou_stock_window,
            factor_returns_window=factor_returns_window,
            dt_years=config.dt_years,
        )

        residuals_window_df, x_process_window_df = _build_regression_windows(regression_results)

        ou_table = build_ou_signal_table(
            regression_results=regression_results,
            dt_years=config.dt_years,
            max_mean_reversion_days=config.max_mean_reversion_days,
            center_means_cross_sectionally=True,
        )

        # ----------------------------
        # 4) Full score on the whole universe
        # ----------------------------
        full_score = pd.Series(np.nan, index=all_assets, dtype=float)

        if not ou_table.empty:
            score_col = "modified_s_score" if config.use_modified_score else "s_score"
            usable = ou_table[ou_table["is_usable"]].copy()

            if not usable.empty:
                full_score.loc[usable.index] = usable[score_col]

        # ----------------------------
        # 5) Signals
        # ----------------------------
        new_states = generate_target_states(
            score_series=full_score,
            current_states=current_states,
            config=config,
        )

        trade_count = compute_trade_count(current_states, new_states)

        # ----------------------------
        # 6) Portfolio weights
        # ----------------------------
        new_weights = states_to_weights(
            states=new_states,
            long_leverage=config.long_leverage,
            short_leverage=config.short_leverage,
        ).reindex(all_assets).fillna(0.0)

        turnover = compute_turnover(current_weights, new_weights)

        # ----------------------------
        # 7) Next-day PnL attribution
        # ----------------------------
        next_ret = returns.iloc[t_idx + 1].reindex(all_assets).fillna(0.0)
        asset_contrib = (new_weights * next_ret).astype(float)

        long_contrib = float(asset_contrib[new_weights > 0].sum())
        short_contrib = float(asset_contrib[new_weights < 0].sum())
        gross_ret = float(asset_contrib.sum())
        net_ret = gross_ret - config.slippage * turnover

        equity *= (1.0 + net_ret)

        # ----------------------------
        # 8) Save histories
        # ----------------------------
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

        book = summarize_book(new_states, new_weights)
        selected_variance = float(
            pca_res.cumulative_explained_variance[pca_res.n_factors - 1]
        )

        diagnostics_rows.append({
            "date": date,
            "n_assets": int(pca_window.shape[1]),
            "n_factors": int(pca_res.n_factors),
            "selected_variance": selected_variance,
            "n_longs": book["n_longs"],
            "n_shorts": book["n_shorts"],
            "gross": book["gross"],
            "turnover": turnover,
            "trade_count": trade_count,
        })

        # ----------------------------
        # 9) Store optional snapshot
        # ----------------------------
        if _should_store_snapshot(date, config, snapshot_date_set):
            snapshots[date] = DailySnapshot(
                date=date,
                next_date=next_date,
                universe=list(common_assets),

                correlation_matrix=pca_res.correlation_matrix.copy(),
                eigenvalues=pca_res.eigenvalues.copy(),
                eigenvectors=pca_res.eigenvectors.copy(),
                factor_weights=pca_res.factor_weights.copy(),
                factor_returns_window=factor_returns_window.copy(),
                stock_returns_window=ou_stock_window.copy(),

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
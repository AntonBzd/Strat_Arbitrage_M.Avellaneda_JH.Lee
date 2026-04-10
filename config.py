from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StrategyConfig:
    # Data / estimation windows
    pca_window: int = 252
    ou_window: int = 60
    regression_window: int = 90      
    dt_years: float = 1.0 / 252.0

    # PCA selection
    pca_mode: str = "fixed"  # "fixed" or "variance"
    n_factors_fixed: int = 15
    explained_variance_target: float = 0.55

    # OU / signal filtering
    max_mean_reversion_days: float = 30.0

    # Signal thresholds
    sbo: float = 1.25
    sso: float = 1.25
    sbc: float = 0.75
    ssc: float = 0.50

    # Divergence stop-loss
    use_divergence_stop: bool = True
    divergence_stop_zscore: float = 4.0  

    # Portfolio / costs
    long_leverage: float = 1.0
    short_leverage: float = 1.0
    slippage: float = 0.0005
    initial_capital: float = 1_000_000.0
    gross_sizing_mode: str = "natural" # "target" or "natural"

    # Backtest controls
    min_assets: int = 1
    use_modified_score: bool = False
    allow_reentry_same_day: bool = False

    # Factor-neutral portfolio construction
    use_factor_neutral_overlay: bool = True
    factor_neutral_ridge: float = 1e-8
    neutralize_net_exposure: bool = True

    # Event-driven / turnover control
    use_event_driven_weights: bool = True
    entry_weight_mode: str = "constant"   # "constant" or "score"
    entry_weight_value: float = 0.05      
    rebalance_held_positions: bool = False
    use_no_trade_band: bool = True
    no_trade_band: float = 0.0025         # 25 bps on weights
    hedge_rebalance_fraction: float = 1.0 # 1.0 = full rebalance, <1 = partial

    # Snapshot controls
    store_snapshots: bool = False
    snapshot_dates: Optional[list[str]] = None
    snapshot_start: Optional[str] = None
    snapshot_end: Optional[str] = None

    # Outputs
    output_dir: str = "outputs"

    # Adaptive thresholds from OU parameters
    use_adaptive_thresholds: bool = False
    adaptive_base_sbo: float = 1.25     
    adaptive_kappa_scale: float = 0.5     # faster kappa -> tighter threshold

    # Signal-weighted sizing
    use_signal_weighting: bool = False     # weight by |s_score| instead of equal weight
    signal_weight_cap: float = 3.0     

    # Concentration limits 
    max_sector_weight: float = 0.30     
    max_single_position: float = 0.05     

    ### 
    # Market impact cost model 
    use_market_impact: bool = False
    impact_coefficient: float = 0.1            # cost = impact_coeff * sigma * sqrt(participation)
    typical_participation_rate: float = 0.01  # assume 1% of ADV

    # Bayesian shrinkage on OU params 
    use_bayesian_shrinkage: bool = False
    shrinkage_strength: float = 0.3       # 0 = no shrinkage, 1 = full prior
    ###
    

    def validate(self) -> None:
        if self.pca_mode not in {"fixed", "variance"}:
            raise ValueError("pca_mode must be 'fixed' or 'variance'")
        if self.n_factors_fixed <= 0:
            raise ValueError("n_factors_fixed must be > 0")
        if not (0.0 < self.explained_variance_target <= 1.0):
            raise ValueError("explained_variance_target must be in (0, 1]")
        if self.pca_window <= 1 or self.ou_window <= 2:
            raise ValueError("Windows are too short")
        if self.regression_window < self.ou_window:
            raise ValueError("regression_window must be >= ou_window")
        if self.long_leverage < 0 or self.short_leverage < 0:
            raise ValueError("Leverage must be non-negative")
        if self.slippage < 0:
            raise ValueError("slippage must be non-negative")
        if self.min_assets <= 0:
            raise ValueError("min_assets must be > 0")
        if self.divergence_stop_zscore <= self.sso:
            raise ValueError("divergence_stop must be > sso")
        if self.entry_weight_mode not in {"constant", "score"}:
            raise ValueError("entry_weight_mode must be 'constant' or 'score'")
        if self.entry_weight_value < 0:
            raise ValueError("entry_weight_value must be non-negative")
        if self.no_trade_band < 0:
            raise ValueError("no_trade_band must be non-negative")
        if not (0.0 <= self.hedge_rebalance_fraction <= 1.0):
            raise ValueError("hedge_rebalance_fraction must be in [0, 1]")
        if self.gross_sizing_mode not in {"natural", "target"}:
            raise ValueError("gross_sizing_mode must be 'natural' or 'target'")

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def make_pca_config() -> StrategyConfig:
    return StrategyConfig()


def make_improved_config() -> StrategyConfig:
    """Config with all improvements turned on."""
    return StrategyConfig(
        regression_window=90,
        use_adaptive_thresholds=False,
        use_divergence_stop=True,
        divergence_stop_zscore=4.0,
        use_signal_weighting=True,
        signal_weight_cap=3.0,
        use_bayesian_shrinkage=True,
        shrinkage_strength=0.3,
        n_factors_fixed=15,
    )

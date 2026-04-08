from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StrategyConfig:
    # Data / estimation windows
    pca_window: int = 252
    ou_window: int = 60
    dt_years: float = 1.0 / 252.0

    # PCA selection
    pca_mode: str = "fixed"  # "fixed" or "variance"
    n_factors_fixed: int = 10
    explained_variance_target: float = 0.55

    # OU / signal filtering
    max_mean_reversion_days: float = 30.0

    # Signal thresholds
    sbo: float = 1.25  # buy to open if s < -sbo
    sso: float = 1.25  # sell to open if s > +sso
    sbc: float = 0.75  # close short if s < +sbc
    ssc: float = 0.50  # close long if s > -ssc

    # Portfolio / costs
    long_leverage: float = 1.0
    short_leverage: float = 1.0
    slippage: float = 0.0005
    initial_capital: float = 1_000_000.0

    # Backtest controls
    min_assets: int = 1
    use_modified_score: bool = False
    allow_reentry_same_day: bool = False

    # Snapshot controls
    store_snapshots: bool = False
    snapshot_dates: Optional[list[str]] = None
    snapshot_start: Optional[str] = None
    snapshot_end: Optional[str] = None

    # Outputs
    output_dir: str = "outputs"

    def validate(self) -> None:
        if self.pca_mode not in {"fixed", "variance"}:
            raise ValueError("pca_mode must be 'fixed' or 'variance'")
        if self.n_factors_fixed <= 0:
            raise ValueError("n_factors_fixed must be > 0")
        if not (0.0 < self.explained_variance_target <= 1.0):
            raise ValueError("explained_variance_target must be in (0, 1]")
        if self.pca_window <= 1 or self.ou_window <= 2:
            raise ValueError("Windows are too short")
        if self.long_leverage < 0 or self.short_leverage < 0:
            raise ValueError("Leverage must be non-negative")
        if self.slippage < 0:
            raise ValueError("slippage must be non-negative")
        if self.min_assets <= 0:
            raise ValueError("min_assets must be > 0")

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def make_pca_config() -> StrategyConfig:
    cfg = StrategyConfig()
    return cfg

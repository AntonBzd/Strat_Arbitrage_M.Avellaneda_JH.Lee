from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from returns import standardize_returns_window


@dataclass
class PCAResult:
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    eigenvectors: pd.DataFrame
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    n_factors: int
    factor_weights: pd.DataFrame
    factor_returns: pd.DataFrame


def compute_correlation_matrix(standardized_returns: pd.DataFrame) -> pd.DataFrame:
    return standardized_returns.corr()


def eigen_decompose_correlation(corr: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    eigvals, eigvecs = np.linalg.eigh(corr.values)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvecs_df = pd.DataFrame(
        eigvecs,
        index=corr.index,
        columns=[f"factor_{i+1}" for i in range(len(eigvals))]
    )
    return eigvals, eigvecs_df


def select_number_of_factors(
    eigenvalues: np.ndarray,
    mode: str = "fixed",
    n_factors_fixed: int = 15,
    explained_variance_target: float = 0.55,
) -> int:
    if mode == "fixed":
        return min(n_factors_fixed, len(eigenvalues))

    if mode == "variance":
        cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return int(np.searchsorted(cumulative, explained_variance_target) + 1)

    raise ValueError("mode must be 'fixed' or 'variance'")


def build_eigenportfolio_weights(
    eigenvectors: pd.DataFrame,
    asset_vols: pd.Series,
    n_factors: int,
) -> pd.DataFrame:
    selected = eigenvectors.iloc[:, :n_factors].copy()
    safe_vols = asset_vols.replace(0.0, np.nan)
    weights = selected.div(safe_vols, axis=0)
    weights = weights.dropna(axis=0, how="any")
    return weights


def compute_factor_returns(
    raw_returns_window: pd.DataFrame,
    factor_weights: pd.DataFrame,
) -> pd.DataFrame:
    aligned_returns = raw_returns_window[factor_weights.index].copy()
    factor_returns = aligned_returns @ factor_weights

    factor_returns.columns = factor_weights.columns
    return factor_returns


def run_pca_factor_model(
    returns_window: pd.DataFrame,
    mode: str = "fixed",
    n_factors_fixed: int = 15,
    explained_variance_target: float = 0.55,
) -> PCAResult:
    """
    returns_window: DataFrame (dates x assets) sans NaN sur les actifs utilisés.
    """
    standardized = standardize_returns_window(returns_window)
    aligned_assets = standardized.columns
    raw_returns = returns_window[aligned_assets].copy()

    corr = compute_correlation_matrix(standardized)
    eigvals, eigvecs = eigen_decompose_correlation(corr)

    explained_var = eigvals / np.sum(eigvals)
    cum_explained = np.cumsum(explained_var) # donne bien la trace de la matrice ? 

    n_factors = select_number_of_factors(
        eigenvalues=eigvals,
        mode=mode,
        n_factors_fixed=n_factors_fixed,
        explained_variance_target=explained_variance_target,
    )

    asset_vols = raw_returns.std(axis=0, ddof=1) # donne bien la vol par stock ?
    factor_weights = build_eigenportfolio_weights(
        eigenvectors=eigvecs,
        asset_vols=asset_vols,
        n_factors=n_factors,
    )

    factor_returns = compute_factor_returns(
        raw_returns_window=raw_returns,
        factor_weights=factor_weights,
    )

    return PCAResult(
        correlation_matrix=corr,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        explained_variance_ratio=explained_var,
        cumulative_explained_variance=cum_explained,
        n_factors=n_factors,
        factor_weights=factor_weights,
        factor_returns=factor_returns,
    )
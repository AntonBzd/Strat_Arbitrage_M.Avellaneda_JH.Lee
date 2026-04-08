from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_correlation_matrix(corr: pd.DataFrame, title: str = "Correlation Matrix", figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Assets")
    ax.set_ylabel("Assets")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_scree(eigenvalues: np.ndarray, title: str = "PCA Eigenvalues", figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    plt.tight_layout()
    plt.show()


def plot_explained_variance(eigenvalues: np.ndarray, title: str = "Explained Variance", figsize=(8, 4)):
    explained = eigenvalues / np.sum(eigenvalues)
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(explained) + 1), explained, marker="o", label="Explained variance")
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", label="Cumulative explained variance")
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance ratio")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_factor_loadings(
    series: pd.Series,
    top_n: int = 20,
    title: str = "Factor Loadings",
    figsize=(10, 4),
):
    s = series.sort_values(key=lambda x: x.abs(), ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Loading")
    plt.tight_layout()
    plt.show()


def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve", figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    equity_curve.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.show()


def plot_drawdown(drawdown_series: pd.Series, title: str = "Drawdown", figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    drawdown_series.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


def plot_number_of_factors(n_factors: pd.Series, title: str = "Number of PCA Factors", figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    n_factors.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Number of factors")
    plt.tight_layout()
    plt.show()


def plot_strategy_comparison(curves: dict[str, pd.Series], title: str = "Strategy Comparison", figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, curve in curves.items():
        curve.plot(ax=ax, label=name)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_score_distribution(scores: pd.Series, title: str = "Score Distribution", figsize=(8, 4), bins: int = 30):
    fig, ax = plt.subplots(figsize=figsize)
    scores.dropna().hist(ax=ax, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Score")
    plt.tight_layout()
    plt.show()


def plot_full_score(full_score: pd.Series, title: str = "Full Score", figsize=(10, 4), top_n: int = 20):
    s = full_score.dropna().sort_values()
    if s.empty:
        print("No valid scores to plot.")
        return

    low = s.head(top_n)
    high = s.tail(top_n)
    combined = pd.concat([low, high])

    fig, ax = plt.subplots(figsize=figsize)
    combined.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.show()


def plot_ou_process(
    x_process: pd.Series,
    equilibrium_mean: float,
    title: str = "OU State Process",
    figsize=(10, 4),
):
    fig, ax = plt.subplots(figsize=figsize)
    x_process.plot(ax=ax, label="X process")
    ax.axhline(equilibrium_mean, linestyle="--", label="Estimated equilibrium mean")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(
    residuals: pd.Series,
    title: str = "Residuals",
    figsize=(10, 4),
):
    fig, ax = plt.subplots(figsize=figsize)
    residuals.plot(ax=ax)
    ax.axhline(0.0, linestyle="--")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_long_short_contribution(
    total_returns: pd.Series,
    long_contribution: pd.Series,
    short_contribution: pd.Series,
    title: str = "Long vs Short Contribution",
    figsize=(10, 4),
):
    total_curve = (1.0 + total_returns.fillna(0.0)).cumprod()
    long_curve = (1.0 + long_contribution.fillna(0.0)).cumprod()
    short_curve = (1.0 + short_contribution.fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=figsize)
    total_curve.plot(ax=ax, label="Total")
    long_curve.plot(ax=ax, label="Long contribution")
    short_curve.plot(ax=ax, label="Short contribution")
    ax.set_title(title)
    ax.set_ylabel("Cumulative growth")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_asset_contribution_bar(
    contribution_by_asset: pd.Series,
    title: str = "Asset Contribution",
    figsize=(10, 5),
):
    s = contribution_by_asset.sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    s.plot(kind="bar", ax=ax) 
    ax.set_title(title)
    ax.set_ylabel("Contribution")
    plt.tight_layout()
    plt.show()
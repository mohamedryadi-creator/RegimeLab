"""Reusable figures for notebooks and the report.

Notebooks must contain no plotting logic beyond calling these helpers on
experiment outputs — that keeps figures reproducible and testable. Matplotlib
is imported lazily (``notebooks`` extra); every function returns the Figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimelab.backtest.metrics import drawdown_series


def _plt():
    import matplotlib.pyplot as plt

    return plt


def wealth_curves(
    net_returns: pd.DataFrame, title: str = "Cumulative wealth, net of costs", logy: bool = True
):
    """Compounded wealth (start = 1) for each column of net returns."""
    plt = _plt()
    wealth = (1.0 + net_returns.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    wealth.plot(ax=ax, logy=logy, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("wealth (start = 1)")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def drawdown_curves(net_returns: pd.DataFrame, title: str = "Drawdowns"):
    plt = _plt()
    dd = net_returns.apply(drawdown_series)
    fig, ax = plt.subplots(figsize=(10, 4))
    dd.plot(ax=ax, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("drawdown")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def regime_overlay(
    prices: pd.Series,
    labels: pd.Series,
    turbulent_probability: pd.Series | None = None,
    turbulent_label: float = 1.0,
    title: str = "Price with detected turbulent regimes",
):
    """Log price with turbulent periods shaded; optional filtered-probability panel."""
    plt = _plt()
    n_rows = 2 if turbulent_probability is not None else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 6 if n_rows == 2 else 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]} if n_rows == 2 else None,
    )
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(prices.index, prices, linewidth=1.0, color="black")
    ax.set_yscale("log")
    mask = labels.reindex(prices.index) == turbulent_label
    ax.fill_between(
        prices.index,
        prices.min(),
        prices.max(),
        where=mask,
        alpha=0.25,
        color="tab:red",
        linewidth=0,
        label="turbulent (out-of-sample label)",
    )
    ax.set_title(title)
    ax.set_ylabel("price (log scale)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    if turbulent_probability is not None:
        ax2 = axes[1]
        ax2.plot(
            turbulent_probability.index, turbulent_probability, linewidth=0.8, color="tab:red"
        )
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("P(turbulent | past)")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def bootstrap_forest(
    table: pd.DataFrame, title: str = "Sharpe differences with 95% bootstrap CIs"
):
    """Horizontal point + interval plot from a bootstrap_sharpe_diffs table."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.5))
    y = np.arange(len(table))[::-1]
    ax.errorbar(
        table["diff"],
        y,
        xerr=[table["diff"] - table["ci_low"], table["ci_high"] - table["diff"]],
        fmt="o",
        capsize=3,
        color="tab:blue",
    )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_xlabel("ΔSharpe (annualized)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def sensitivity_heatmap(
    pivot: pd.DataFrame, title: str = "Vol-gate sensitivity (Sharpe)", fmt: str = ".2f"
):
    """Annotated heatmap of a parameter grid (e.g. vol window x quantile)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(i) for i in pivot.index])
    ax.set_xlabel(pivot.columns.name or "quantile")
    ax.set_ylabel(pivot.index.name or "vol_window")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, format(pivot.iloc[i, j], fmt), ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def subperiod_bars(sharpe: pd.DataFrame, title: str = "Sharpe by sub-period"):
    """Grouped bars: rows = eras, columns = strategies."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    sharpe.plot.bar(ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Sharpe (annualized)")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def turnover_sharpe_frontier(
    table: pd.DataFrame, title: str = "Turnover vs. Sharpe (net of costs)"
):
    """Labeled scatter of annual turnover against Sharpe, one point per strategy.

    Expects columns ``avg_annual_turnover`` and ``sharpe``, indexed by strategy
    name — e.g. the @headline-cost rows of an experiment summary table.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(table["avg_annual_turnover"], table["sharpe"], color="tab:blue", zorder=3)
    for name, row in table.iterrows():
        ax.annotate(
            str(name),
            (row["avg_annual_turnover"], row["sharpe"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("average annual turnover (×)")
    ax.set_ylabel("Sharpe (annualized)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure, creating parent directories; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

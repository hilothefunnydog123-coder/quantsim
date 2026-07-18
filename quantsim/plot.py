"""Equity-curve plotting (optional — requires matplotlib)."""
from __future__ import annotations

import numpy as np

from quantsim.backtest import BacktestResult


def plot_backtest(result: BacktestResult, path: str, title: str | None = None) -> None:
    """Save a two-panel chart: equity vs buy-and-hold, and drawdown."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires matplotlib — install with: pip install matplotlib"
        ) from exc

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(color="#21262d", linewidth=0.6)

    ax_eq.plot(result.dates, result.equity, color="#3fb950", linewidth=1.6,
               label=f"{result.strategy_name} (Sharpe {result.metrics['sharpe']:.2f})")
    ax_eq.plot(result.dates, result.benchmark, color="#58a6ff", linewidth=1.2,
               alpha=0.85,
               label=f"buy & hold (Sharpe {result.metrics['benchmark']['sharpe']:.2f})")
    ax_eq.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                 fontsize=9)
    ax_eq.set_title(title or f"{result.strategy_name} vs buy & hold",
                    color="#e6edf3", fontsize=11)

    dd = result.equity / np.maximum.accumulate(result.equity) - 1.0
    ax_dd.fill_between(result.dates, dd * 100, 0, color="#f85149", alpha=0.55)
    ax_dd.set_ylabel("drawdown %", color="#8b949e", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

"""Event-driven daily backtester with a tiny, honest core.

The one rule that matters: the weight chosen with data through day t earns the
return from day t to day t+1 — never day t's own return. No lookahead, ever.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class Strategy(ABC):
    """Maps price history to a target portfolio weight.

    Subclasses set ``warmup`` (bars needed before the first signal) and
    implement ``target_weight``, returning a desired exposure in [-1, 1]:
    1.0 = fully long, 0.0 = flat, -1.0 = fully short.
    """

    warmup: int = 0
    name: str = "strategy"

    @abstractmethod
    def target_weight(self, closes: np.ndarray) -> float:
        """Desired weight given all closes up to and including today."""

    def reset(self) -> None:
        """Clear any internal state before a fresh backtest run."""


@dataclass
class BacktestResult:
    strategy_name: str
    dates: list[dt.date]
    equity: np.ndarray          # portfolio value, same length as dates
    benchmark: np.ndarray       # buy-and-hold of the same capital
    weights: np.ndarray         # weight held from bar t to t+1 (length n-1)
    metrics: dict = field(default_factory=dict)


def max_drawdown_curve(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    return float((equity / peaks - 1.0).min())


def _series_metrics(equity: np.ndarray, periods_per_year: int) -> dict:
    rets = equity[1:] / equity[:-1] - 1.0
    years = len(rets) / periods_per_year
    total = float(equity[-1] / equity[0] - 1.0)
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)) \
        if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,  # rf assumed 0 — documented in the README
        "max_drawdown": max_drawdown_curve(equity),
    }


def run_backtest(
    closes: np.ndarray,
    strategy: Strategy,
    dates: list[dt.date] | None = None,
    initial: float = 10_000.0,
    cost_bps: float = 1.0,
    periods_per_year: int = 252,
    execution=None,
) -> BacktestResult:
    """Backtest ``strategy`` over a daily close series.

    Transaction costs are charged as ``cost_bps`` basis points on turnover
    (the change in absolute weight), the standard first-order cost model.

    Pass an ``execution`` model (e.g. :class:`quantsim.execution.BookExecution`)
    to replace the flat fee with order-book mechanics: each rebalance is sized
    in shares from current equity and executed against synthetic book
    liquidity, so large or frequent trades pay realistic market impact.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n < max(3, strategy.warmup + 2):
        raise ValueError(
            f"need at least {max(3, strategy.warmup + 2)} bars for {strategy.name} "
            f"(warmup={strategy.warmup}), got {n}"
        )
    if dates is None:
        dates = [dt.date(1970, 1, 1) + dt.timedelta(days=i) for i in range(n)]

    strategy.reset()
    bar_returns = closes[1:] / closes[:-1] - 1.0     # bar_returns[t]: t -> t+1
    weights = np.zeros(n - 1)
    for t in range(strategy.warmup, n - 1):
        # Only closes[: t + 1] are visible — the no-lookahead boundary.
        weights[t] = float(np.clip(strategy.target_weight(closes[: t + 1]), -1.0, 1.0))

    turnover = np.abs(np.diff(np.concatenate([[0.0], weights])))
    if execution is None:
        strat_returns = weights * bar_returns - turnover * (cost_bps / 1e4)
        equity = np.concatenate([[initial], initial * np.cumprod(1.0 + strat_returns)])
    else:
        # Equity-dependent costs: size each rebalance in shares and pay the
        # slippage of walking the synthetic book with that order.
        strat_returns = np.empty(n - 1)
        equity_t = initial
        equity = np.empty(n)
        equity[0] = initial
        for t in range(n - 1):
            cost_frac = 0.0
            if turnover[t] > 0:
                shares = turnover[t] * equity_t / closes[t]
                cost_frac = (
                    turnover[t]
                    * execution.slippage_bps(shares, mid=float(closes[t]))
                    / 1e4
                )
            strat_returns[t] = weights[t] * bar_returns[t] - cost_frac
            equity_t *= 1.0 + strat_returns[t]
            equity[t + 1] = equity_t
    benchmark = initial * closes / closes[0]

    active = weights != 0
    active_rets = strat_returns[active]
    metrics = _series_metrics(equity, periods_per_year)
    metrics.update(
        exposure=float(active.mean()),
        n_trades=int((turnover > 0).sum()),
        win_rate=float((active_rets > 0).mean()) if active_rets.size else 0.0,
        benchmark=_series_metrics(benchmark, periods_per_year),
    )
    return BacktestResult(
        strategy_name=strategy.name,
        dates=list(dates),
        equity=equity,
        benchmark=benchmark,
        weights=weights,
        metrics=metrics,
    )

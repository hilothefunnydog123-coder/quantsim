"""Monte Carlo engines and risk metrics."""
from __future__ import annotations

import numpy as np


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    years: float,
    steps_per_year: int = 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion price paths.

    Uses the exact log-space solution, so paths are unbiased for any step size:
        S(t+dt) = S(t) * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z)

    Returns an array of shape ``(n_paths, n_steps + 1)`` where column 0 is ``s0``.
    """
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")

    rng = np.random.default_rng(seed)
    n_steps = int(round(years * steps_per_year))
    dt = 1.0 / steps_per_year

    shocks = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(log_paths)
    return paths


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough decline of each path, as negative fractions."""
    peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / peaks - 1.0
    return drawdowns.min(axis=1)


def summarize(paths: np.ndarray, years: float, risk_free: float = 0.03) -> dict:
    """Compute portfolio risk/return metrics across simulated paths."""
    s0 = float(paths[0, 0])
    finals = paths[:, -1]
    total_return = finals / s0 - 1.0
    cagr = (finals / s0) ** (1.0 / years) - 1.0

    var_95 = float(np.percentile(total_return, 5))
    tail = total_return[total_return <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95

    drawdowns = max_drawdown(paths)
    cagr_std = float(cagr.std(ddof=1))
    sharpe = (float(cagr.mean()) - risk_free) / cagr_std if cagr_std > 0 else float("nan")

    return {
        "n_paths": int(paths.shape[0]),
        "median_final": float(np.median(finals)),
        "p5_final": float(np.percentile(finals, 5)),
        "p95_final": float(np.percentile(finals, 95)),
        "mean_cagr": float(cagr.mean()),
        "prob_loss": float((total_return < 0).mean()),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "sharpe": sharpe,
        "median_max_drawdown": float(np.median(drawdowns)),
        "worst_max_drawdown": float(drawdowns.min()),
    }


def ascii_histogram(values: np.ndarray, bins: int = 22, width: int = 44) -> str:
    """Render a horizontal ASCII histogram, one bin per line."""
    counts, edges = np.histogram(values, bins=bins)
    top = counts.max() if counts.max() > 0 else 1
    lines = []
    for count, lo in zip(counts, edges[:-1]):
        bar_len = int(round(count / top * width))
        if count and bar_len == 0:
            bar_len = 1
        lines.append(f"{lo:>14,.0f} │{'█' * bar_len}{' ' * (width - bar_len)}│ {count}")
    return "\n".join(lines)

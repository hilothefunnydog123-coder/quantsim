"""Built-in strategies. Each is deliberately simple and readable — the point is
a correct reference implementation you can fork, not a black box."""
from __future__ import annotations

import numpy as np

from quantsim.backtest import Strategy


class BuyAndHold(Strategy):
    """Always fully invested. The benchmark everyone has to beat."""

    name = "buyhold"
    warmup = 0

    def target_weight(self, closes: np.ndarray) -> float:
        return 1.0


class SMACrossover(Strategy):
    """Long when the fast moving average is above the slow one, else flat.

    The classic trend filter: slow to enter, slow to exit, but sidesteps the
    deepest drawdowns of prolonged bear markets.
    """

    name = "sma"

    def __init__(self, fast: int = 20, slow: int = 100):
        if not 0 < fast < slow:
            raise ValueError("require 0 < fast < slow")
        self.fast = fast
        self.slow = slow
        self.warmup = slow

    def target_weight(self, closes: np.ndarray) -> float:
        fast_ma = closes[-self.fast:].mean()
        slow_ma = closes[-self.slow:].mean()
        return 1.0 if fast_ma > slow_ma else 0.0


class Momentum(Strategy):
    """Time-series momentum: long if the trailing return is positive, else flat.

    Default lookback of 126 bars ≈ 6 months of daily data.
    """

    name = "momentum"

    def __init__(self, lookback: int = 126):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self.warmup = lookback

    def target_weight(self, closes: np.ndarray) -> float:
        trailing = closes[-1] / closes[-self.lookback - 1] - 1.0
        return 1.0 if trailing > 0 else 0.0


class MeanReversion(Strategy):
    """Buy dips: enter long when price z-score vs its SMA drops below
    ``entry_z``, exit once it recovers above ``exit_z``. Stateful.
    """

    name = "meanrev"

    def __init__(self, window: int = 20, entry_z: float = -1.0, exit_z: float = 0.0):
        if window < 3:
            raise ValueError("window must be >= 3")
        if entry_z >= exit_z:
            raise ValueError("entry_z must be below exit_z")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.warmup = window
        self._long = False

    def reset(self) -> None:
        self._long = False

    def target_weight(self, closes: np.ndarray) -> float:
        recent = closes[-self.window:]
        std = recent.std(ddof=1)
        if std == 0:
            return 1.0 if self._long else 0.0
        z = (closes[-1] - recent.mean()) / std
        if self._long and z >= self.exit_z:
            self._long = False
        elif not self._long and z <= self.entry_z:
            self._long = True
        return 1.0 if self._long else 0.0


def make_strategy(name: str, **kwargs) -> Strategy:
    """Build a strategy by registry name — the hook the CLI uses."""
    registry = {
        "buyhold": BuyAndHold,
        "sma": SMACrossover,
        "momentum": Momentum,
        "meanrev": MeanReversion,
    }
    if name not in registry:
        raise ValueError(f"unknown strategy {name!r} — choose from {sorted(registry)}")
    return registry[name](**kwargs) if kwargs else registry[name]()

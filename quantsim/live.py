"""Live paper trading: run the SAME Strategy you backtested, one cycle a day.

The whole point of the stack: :func:`run_cycle` takes any
:class:`quantsim.backtest.Strategy` — the identical object
:func:`quantsim.backtest.run_backtest` evaluates on history — computes today's
target weight, and places one rebalancing order with a broker. Backtest it,
then deploy it, with zero reimplementation drift.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from quantsim.backtest import Strategy
from quantsim.broker import PaperBroker


@dataclass
class CycleResult:
    ts: str
    symbol: str
    price: float
    weight: float
    current_qty: float
    target_qty: int
    order_qty: float
    equity: float


def run_cycle(
    broker,
    closes,
    symbol: str,
    strategy: Strategy,
    state_dir: str | Path = "state",
) -> CycleResult:
    """Compute the target position and place one rebalancing order.

    Live trading is long-only: strategy weights are clipped to [0, 1].
    Idempotent by design — run it twice on the same data and the second run
    places no order, because the position already matches the target. That is
    what makes it safe to run from an unattended scheduler.
    """
    closes = np.asarray(closes, dtype=float)
    if len(closes) < strategy.warmup + 1:
        raise ValueError(
            f"{strategy.name} needs {strategy.warmup + 1} bars, got {len(closes)}"
        )
    price = float(closes[-1])
    if isinstance(broker, PaperBroker):
        broker.mark(symbol, price)

    # Replay the full history exactly like the backtester does, so stateful
    # strategies (e.g. MeanReversion) arrive at today with identical state:
    # the live weight is bit-for-bit the backtest's final weight.
    strategy.reset()
    weight = 0.0
    for t in range(strategy.warmup, len(closes)):
        weight = float(np.clip(strategy.target_weight(closes[: t + 1]), 0.0, 1.0))
    equity = broker.equity()
    target_qty = math.floor(equity * weight / price)
    current_qty = broker.position(symbol)
    order_qty = target_qty - current_qty

    if order_qty != 0:
        broker.submit(symbol, order_qty, price)

    result = CycleResult(
        ts=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        symbol=symbol,
        price=price,
        weight=weight,
        current_qty=current_qty,
        target_qty=target_qty,
        order_qty=order_qty,
        equity=broker.equity(),
    )
    history = Path(state_dir) / "history.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    return result


def load_history(state_dir: str | Path = "state") -> list[dict]:
    history = Path(state_dir) / "history.jsonl"
    if not history.exists():
        return []
    return [json.loads(line) for line in history.read_text().splitlines() if line.strip()]

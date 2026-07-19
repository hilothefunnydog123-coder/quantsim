"""Limit order book with price-time priority matching.

The core data structure inside every exchange: resting limit orders queue
FIFO at each price level; incoming orders match against the best opposing
price first, then by arrival order within a level. Trades always execute at
the resting (maker) order's price.

Used standalone, or as the liquidity model behind
:class:`quantsim.execution.BookExecution` for slippage-aware backtests.
"""
from __future__ import annotations

from bisect import insort
from dataclasses import dataclass


@dataclass
class Order:
    id: str
    side: str          # "buy" | "sell"
    price: float
    qty: float         # remaining quantity, decreases as filled
    seq: int           # arrival sequence — the "time" in price-time priority


@dataclass(frozen=True)
class Trade:
    price: float       # execution price = maker's limit price
    qty: float
    maker_id: str
    taker_id: str
    taker_side: str
    seq: int


@dataclass(frozen=True)
class BookLevel:
    price: float
    qty: float
    orders: int


def _validate(value: float, name: str) -> None:
    if not (value > 0) or value != value or value in (float("inf"),):
        raise ValueError(f"{name} must be a positive finite number, got {value}")


class OrderBook:
    """Price-time priority matching engine."""

    def __init__(self) -> None:
        # price level -> FIFO queue of orders; sort keys kept in ascending
        # lists (bids negated) so the best price is always index 0
        self._levels: dict[str, dict[float, list[Order]]] = {"buy": {}, "sell": {}}
        self._keys: dict[str, list[float]] = {"buy": [], "sell": []}
        self._by_id: dict[str, Order] = {}
        self._seq = 0
        self._auto = 0

    # -- public API ----------------------------------------------------------

    @property
    def open_orders(self) -> int:
        return len(self._by_id)

    def limit(self, side: str, price: float, qty: float, id: str | None = None) -> list[Trade]:
        """Submit a limit order: match while it crosses, rest the remainder."""
        self._check_side(side)
        _validate(price, "price")
        _validate(qty, "qty")
        order = self._new_order(side, price, qty, id)
        trades = self._match(order, price)
        if order.qty > 0:
            self._rest(order)
        return trades

    def market(self, side: str, qty: float, id: str | None = None) -> list[Trade]:
        """Submit a market order: fill at any price, best first; the
        unfillable remainder is discarded (immediate-or-cancel)."""
        self._check_side(side)
        _validate(qty, "qty")
        order = self._new_order(side, float("nan"), qty, id)
        return self._match(order, float("inf") if side == "buy" else 0.0)

    def cancel(self, id: str) -> bool:
        """Cancel a resting order. False if unknown or already filled."""
        order = self._by_id.get(id)
        if order is None:
            return False
        queue = self._levels[order.side][order.price]
        queue.remove(order)
        if not queue:
            del self._levels[order.side][order.price]
            self._keys[order.side].remove(self._key(order.side, order.price))
        del self._by_id[id]
        return True

    def best_bid(self) -> float | None:
        keys = self._keys["buy"]
        return -keys[0] if keys else None

    def best_ask(self) -> float | None:
        keys = self._keys["sell"]
        return keys[0] if keys else None

    def spread(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        return ask - bid if bid is not None and ask is not None else None

    def midpoint(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    def depth(self, levels: int = 10) -> dict[str, list[BookLevel]]:
        """Aggregated snapshot: best ``levels`` price levels per side."""
        out: dict[str, list[BookLevel]] = {}
        for side, name in (("buy", "bids"), ("sell", "asks")):
            out[name] = [
                BookLevel(
                    price=(-k if side == "buy" else k),
                    qty=sum(o.qty for o in self._levels[side][-k if side == "buy" else k]),
                    orders=len(self._levels[side][-k if side == "buy" else k]),
                )
                for k in self._keys[side][:levels]
            ]
        return out

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _check_side(side: str) -> None:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    @staticmethod
    def _key(side: str, price: float) -> float:
        return -price if side == "buy" else price

    def _new_order(self, side: str, price: float, qty: float, id: str | None) -> Order:
        if id is None:
            self._auto += 1
            id = f"o{self._auto}"
        if id in self._by_id:
            raise ValueError(f"duplicate order id {id!r}")
        self._seq += 1
        return Order(id=id, side=side, price=price, qty=qty, seq=self._seq)

    def _match(self, taker: Order, limit_price: float) -> list[Trade]:
        trades: list[Trade] = []
        opposing = "sell" if taker.side == "buy" else "buy"
        keys = self._keys[opposing]
        levels = self._levels[opposing]

        while taker.qty > 0 and keys:
            best_price = -keys[0] if opposing == "buy" else keys[0]
            crosses = best_price <= limit_price if taker.side == "buy" else best_price >= limit_price
            if not crosses:
                break
            queue = levels[best_price]
            while taker.qty > 0 and queue:
                maker = queue[0]
                fill = min(taker.qty, maker.qty)
                maker.qty -= fill
                taker.qty -= fill
                self._seq += 1
                trades.append(Trade(
                    price=best_price, qty=fill, maker_id=maker.id,
                    taker_id=taker.id, taker_side=taker.side, seq=self._seq,
                ))
                if maker.qty == 0:
                    queue.pop(0)
                    del self._by_id[maker.id]
            if not queue:
                del levels[best_price]
                keys.pop(0)
        return trades

    def _rest(self, order: Order) -> None:
        queue = self._levels[order.side].get(order.price)
        if queue is not None:
            queue.append(order)
        else:
            self._levels[order.side][order.price] = [order]
            insort(self._keys[order.side], self._key(order.side, order.price))
        self._by_id[order.id] = order

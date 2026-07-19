"""Execution models: what does it actually cost to trade?

The default backtest cost model (flat bps on turnover) is a fine first-order
guess. :class:`BookExecution` replaces the guess with mechanics: every
rebalance is executed as a market order walking a synthetic limit order book
(:class:`quantsim.orderbook.OrderBook`), so the cost of a trade *emerges* from
crossing the spread and eating through finite liquidity — small orders pay
about half the spread, big orders pay visibly more. That's market impact.
"""
from __future__ import annotations

from quantsim.orderbook import OrderBook


class BookExecution:
    """Synthetic-liquidity execution model for slippage-aware backtests.

    A fresh book is seeded around the mid price with ``levels`` price levels
    per side, ``level_shares`` shares at each, spaced ``tick_bps`` apart
    beyond a ``spread_bps``-wide inside spread. A market order for N shares
    walks that ladder; realized slippage is the volume-weighted fill price
    versus mid.

    ``level_shares`` is the liquidity knob: think "shares available near the
    touch". Backtesting a small account on SPY? Liquidity dwarfs your orders
    and costs collapse to ~half the spread. Simulate a thin small-cap by
    turning it down and watching impact eat the strategy's edge.
    """

    def __init__(
        self,
        level_shares: float = 10_000.0,
        levels: int = 200,
        spread_bps: float = 2.0,
        tick_bps: float = 1.0,
    ):
        if level_shares <= 0 or levels < 1:
            raise ValueError("level_shares and levels must be positive")
        if spread_bps < 0 or tick_bps <= 0:
            raise ValueError("spread_bps must be >= 0 and tick_bps > 0")
        self.level_shares = level_shares
        self.levels = levels
        self.spread_bps = spread_bps
        self.tick_bps = tick_bps

    def _build_book(self, mid: float) -> OrderBook:
        book = OrderBook()
        half_spread = mid * self.spread_bps / 2e4
        tick = mid * self.tick_bps / 1e4
        for i in range(self.levels):
            book.limit("sell", mid + half_spread + i * tick, self.level_shares)
            book.limit("buy", mid - half_spread - i * tick, self.level_shares)
        return book

    def slippage_bps(self, shares: float, mid: float = 100.0) -> float:
        """Cost of a market order for ``shares``, in bps of notional vs mid."""
        if shares <= 0:
            return 0.0
        book = self._build_book(mid)
        trades = book.market("buy", shares)  # symmetric book: buy == sell cost
        filled = sum(t.qty for t in trades)
        notional = sum(t.qty * t.price for t in trades)
        # anything beyond the whole ladder pays the worst available price
        if filled < shares:
            worst = max(t.price for t in trades)
            notional += (shares - filled) * worst
        avg_price = notional / shares
        return (avg_price - mid) / mid * 1e4

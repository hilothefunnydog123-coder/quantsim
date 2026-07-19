"""Brokers for live paper trading: a local simulator and Alpaca's paper API.

Both expose the same three calls the live engine needs:
    equity()                          -> float
    position(symbol)                  -> float (signed share count)
    submit(symbol, qty_delta, price)  -> None
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ALPACA_BASE = "https://paper-api.alpaca.markets"


class PaperBroker:
    """Local simulation: instant fills at the provided price, state on disk."""

    def __init__(self, state_path: str | Path, initial_cash: float = 100_000.0):
        self.state_path = Path(state_path)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"cash": initial_cash, "positions": {}, "last_prices": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def mark(self, symbol: str, price: float) -> None:
        """Record the latest price so equity marks to market."""
        self.state["last_prices"][symbol] = price
        self._save()

    def equity(self) -> float:
        positions_value = sum(
            qty * self.state["last_prices"].get(sym, 0.0)
            for sym, qty in self.state["positions"].items()
        )
        return self.state["cash"] + positions_value

    def position(self, symbol: str) -> float:
        return float(self.state["positions"].get(symbol, 0.0))

    def submit(self, symbol: str, qty_delta: float, price: float) -> None:
        if qty_delta == 0:
            return
        self.state["cash"] -= qty_delta * price
        new_qty = self.position(symbol) + qty_delta
        if new_qty == 0:
            self.state["positions"].pop(symbol, None)
        else:
            self.state["positions"][symbol] = new_qty
        self.state["last_prices"][symbol] = price
        self._save()


class AlpacaBroker:
    """Alpaca paper-trading API via plain urllib — no SDK required.

    Reads credentials from APCA_API_KEY_ID / APCA_API_SECRET_KEY (the same
    variables Alpaca's own tooling uses). Paper endpoint only by default:
    real-money trading would need a deliberate base-URL change.
    """

    def __init__(self, base_url: str = ALPACA_BASE):
        self.base_url = base_url.rstrip("/")
        self.key = os.environ.get("APCA_API_KEY_ID", "")
        self.secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise RuntimeError(
                "set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                "(free paper keys: https://app.alpaca.markets)"
            )

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def equity(self) -> float:
        return float(self._request("GET", "/v2/account")["equity"])

    def position(self, symbol: str) -> float:
        try:
            return float(self._request("GET", f"/v2/positions/{symbol}")["qty"])
        except urllib.error.HTTPError as err:
            if err.code == 404:  # no open position
                return 0.0
            raise

    def submit(self, symbol: str, qty_delta: float, price: float) -> None:
        if qty_delta == 0:
            return
        self._request("POST", "/v2/orders", {
            "symbol": symbol,
            "qty": str(abs(int(qty_delta))),
            "side": "buy" if qty_delta > 0 else "sell",
            "type": "market",
            "time_in_force": "day",
        })

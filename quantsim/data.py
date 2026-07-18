"""Market data loading: yfinance when available, Stooq's free CSV endpoint as a
zero-dependency fallback, and plain CSV files for offline work."""
from __future__ import annotations

import csv
import datetime as dt
import io
import urllib.request
from dataclasses import dataclass

import numpy as np

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"


@dataclass
class PriceSeries:
    """Daily close prices for one symbol, oldest first."""

    symbol: str
    dates: list[dt.date]
    closes: np.ndarray

    def __post_init__(self) -> None:
        self.closes = np.asarray(self.closes, dtype=float)
        if len(self.dates) != len(self.closes):
            raise ValueError("dates and closes must have equal length")
        if len(self.closes) < 2:
            raise ValueError(f"{self.symbol}: need at least 2 price points")

    def __len__(self) -> int:
        return len(self.closes)

    def since(self, start: dt.date) -> "PriceSeries":
        idx = next((i for i, d in enumerate(self.dates) if d >= start), None)
        if idx is None:
            raise ValueError(f"{self.symbol}: no data on or after {start}")
        return PriceSeries(self.symbol, self.dates[idx:], self.closes[idx:])


def load_csv(path: str, symbol: str = "CSV", date_col: str = "Date",
             close_col: str = "Close") -> PriceSeries:
    """Load a CSV with (at least) date and close columns, e.g. a Yahoo export."""
    with open(path, newline="") as f:
        return _parse_csv(f, symbol, date_col, close_col)


def _parse_csv(f, symbol: str, date_col: str, close_col: str) -> PriceSeries:
    reader = csv.DictReader(f)
    dates: list[dt.date] = []
    closes: list[float] = []
    for row in reader:
        raw_close = (row.get(close_col) or "").strip()
        raw_date = (row.get(date_col) or "").strip()
        if not raw_close or not raw_date or raw_close.lower() in ("null", "nan"):
            continue
        dates.append(dt.date.fromisoformat(raw_date[:10]))
        closes.append(float(raw_close))
    pairs = sorted(zip(dates, closes))
    return PriceSeries(symbol, [d for d, _ in pairs], np.array([c for _, c in pairs]))


def fetch_stooq(symbol: str, timeout: float = 30.0) -> PriceSeries:
    """Fetch full daily history from Stooq (free, no API key, no dependencies).

    US tickers get a `.us` suffix automatically: SPY -> spy.us.
    """
    stooq_symbol = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    req = urllib.request.Request(
        STOOQ_URL.format(symbol=stooq_symbol),
        headers={"User-Agent": "quantsim (github.com/hilothefunnydog123-coder/quantsim)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    if not text.startswith("Date"):
        raise RuntimeError(f"Stooq returned no data for {symbol!r}")
    return _parse_csv(io.StringIO(text), symbol.upper(), "Date", "Close")


def fetch_yfinance(symbol: str, start: str | None = None) -> PriceSeries:
    """Fetch adjusted daily closes via yfinance (optional dependency)."""
    import yfinance as yf  # deferred so quantsim works without it

    frame = yf.download(symbol, start=start, progress=False, auto_adjust=True)
    if frame is None or len(frame) == 0:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}")
    closes = frame["Close"]
    if hasattr(closes, "columns"):  # multi-index frame for single symbol
        closes = closes[symbol]
    dates = [ts.date() for ts in closes.index]
    return PriceSeries(symbol.upper(), dates, closes.to_numpy(dtype=float))


def fetch(symbol: str, start: str | None = None) -> PriceSeries:
    """Fetch daily closes: yfinance if installed, otherwise Stooq."""
    errors = []
    try:
        return fetch_yfinance(symbol, start=start)
    except ImportError:
        pass
    except Exception as exc:  # network / bad symbol — fall through to Stooq
        errors.append(f"yfinance: {exc}")
    try:
        series = fetch_stooq(symbol)
    except Exception as exc:
        errors.append(f"stooq: {exc}")
        raise RuntimeError(
            f"could not fetch {symbol!r} from any source ({'; '.join(errors)})"
        ) from exc
    if start:
        series = series.since(dt.date.fromisoformat(start))
    return series

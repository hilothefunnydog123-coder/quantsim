"""quantsim — Monte Carlo simulation and strategy backtesting for your terminal."""
from quantsim.backtest import BacktestResult, Strategy, run_backtest
from quantsim.data import PriceSeries, fetch, fetch_stooq, load_csv
from quantsim.simulate import ascii_histogram, max_drawdown, simulate_gbm, summarize
from quantsim.strategies import (
    BuyAndHold,
    MeanReversion,
    Momentum,
    SMACrossover,
    make_strategy,
)

__version__ = "0.2.0"
__all__ = [
    "simulate_gbm", "summarize", "max_drawdown", "ascii_histogram",
    "Strategy", "run_backtest", "BacktestResult",
    "PriceSeries", "fetch", "fetch_stooq", "load_csv",
    "BuyAndHold", "SMACrossover", "Momentum", "MeanReversion", "make_strategy",
    "__version__",
]

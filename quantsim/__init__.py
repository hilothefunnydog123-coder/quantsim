"""quantsim — a full quant stack: simulate, backtest, and paper-trade
the same strategy objects, with an order-book matching engine underneath."""
from quantsim.backtest import BacktestResult, Strategy, run_backtest
from quantsim.broker import AlpacaBroker, PaperBroker
from quantsim.data import PriceSeries, fetch, fetch_stooq, load_csv
from quantsim.execution import BookExecution
from quantsim.live import CycleResult, load_history, run_cycle
from quantsim.orderbook import OrderBook
from quantsim.simulate import ascii_histogram, max_drawdown, simulate_gbm, summarize
from quantsim.strategies import (
    BuyAndHold,
    MeanReversion,
    Momentum,
    SMACrossover,
    make_strategy,
)

__version__ = "0.3.0"
__all__ = [
    "simulate_gbm", "summarize", "max_drawdown", "ascii_histogram",
    "Strategy", "run_backtest", "BacktestResult",
    "PriceSeries", "fetch", "fetch_stooq", "load_csv",
    "BuyAndHold", "SMACrossover", "Momentum", "MeanReversion", "make_strategy",
    "OrderBook", "BookExecution",
    "PaperBroker", "AlpacaBroker", "run_cycle", "load_history", "CycleResult",
    "__version__",
]

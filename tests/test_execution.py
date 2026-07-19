import numpy as np

from quantsim import BookExecution, BuyAndHold, run_backtest, simulate_gbm
from quantsim.backtest import Strategy


def test_small_orders_pay_about_half_the_spread():
    ex = BookExecution(level_shares=10_000, spread_bps=2.0)
    tiny = ex.slippage_bps(1)
    assert 0.9 < tiny < 1.1  # half of a 2 bps spread


def test_impact_grows_with_order_size():
    ex = BookExecution(level_shares=1_000, spread_bps=2.0, tick_bps=1.0)
    costs = [ex.slippage_bps(shares) for shares in (100, 1_000, 10_000, 100_000)]
    assert costs == sorted(costs)
    assert costs[-1] > costs[0] * 5  # big orders pay real impact


def test_impact_shrinks_with_liquidity():
    thin = BookExecution(level_shares=100).slippage_bps(10_000)
    deep = BookExecution(level_shares=100_000).slippage_bps(10_000)
    assert thin > deep


def test_backtest_with_book_execution_costs_more_for_churny_strategies():
    closes = simulate_gbm(100, 0.07, 0.18, years=3, n_paths=1, seed=11)[0]

    class Churner(Strategy):
        name = "churner"
        warmup = 0

        def target_weight(self, closes):
            return float(len(closes) % 2)

    frictionless = run_backtest(closes, Churner(), cost_bps=0.0)
    # thin book: the same strategy trades identically but pays impact
    impacted = run_backtest(closes, Churner(),
                            execution=BookExecution(level_shares=50))
    assert np.array_equal(frictionless.weights, impacted.weights)
    assert impacted.equity[-1] < frictionless.equity[-1]


def test_buy_and_hold_is_nearly_unaffected_by_deep_books():
    closes = simulate_gbm(100, 0.07, 0.18, years=3, n_paths=1, seed=12)[0]
    flat = run_backtest(closes, BuyAndHold(), cost_bps=0.0)
    booked = run_backtest(closes, BuyAndHold(),
                          execution=BookExecution(level_shares=1e9))
    # one entry trade paying ~half a 2bps spread on a deep book: tiny drag
    ratio = booked.equity[-1] / flat.equity[-1]
    assert 0.999 < ratio < 1.0

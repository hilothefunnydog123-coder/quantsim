import numpy as np
import pytest

from quantsim import (
    BuyAndHold,
    MeanReversion,
    Momentum,
    SMACrossover,
    make_strategy,
    run_backtest,
    simulate_gbm,
)
from quantsim.backtest import Strategy, max_drawdown_curve


def synthetic_closes(n=1_000, seed=7):
    return simulate_gbm(100, 0.07, 0.18, years=n / 252, n_paths=1, seed=seed)[0]


def test_buy_and_hold_tracks_benchmark_minus_entry_cost():
    closes = synthetic_closes()
    result = run_backtest(closes, BuyAndHold(), initial=10_000, cost_bps=0.0)
    assert np.allclose(result.equity, result.benchmark)
    # with costs, the single entry trade costs exactly cost_bps once
    with_costs = run_backtest(closes, BuyAndHold(), initial=10_000, cost_bps=10.0)
    assert with_costs.equity[-1] < result.equity[-1]
    assert with_costs.metrics["n_trades"] == 1


def test_no_lookahead_is_enforced():
    """A strategy that could see tomorrow's bar would ace this series; the
    engine must only give it data through today."""
    seen_lengths = []

    class Recorder(Strategy):
        name = "recorder"
        warmup = 0

        def target_weight(self, closes):
            seen_lengths.append(len(closes))
            return 0.0

    closes = synthetic_closes(n=50)
    run_backtest(closes, Recorder())
    # at decision step t the strategy saw exactly t+1 bars, never more
    assert seen_lengths == list(range(1, len(closes)))


def test_flat_strategy_never_gains_or_loses():
    class Flat(Strategy):
        name = "flat"
        warmup = 0

        def target_weight(self, closes):
            return 0.0

    result = run_backtest(synthetic_closes(), Flat(), initial=5_000)
    assert np.allclose(result.equity, 5_000)
    assert result.metrics["n_trades"] == 0
    assert result.metrics["exposure"] == 0.0


def test_sma_goes_long_in_uptrend_and_flat_in_downtrend():
    up = np.linspace(100, 200, 300)
    result_up = run_backtest(up, SMACrossover(fast=5, slow=20))
    assert result_up.weights[25:].mean() > 0.95  # fully long once warmed up

    down = np.linspace(200, 100, 300)
    result_down = run_backtest(down, SMACrossover(fast=5, slow=20))
    assert result_down.weights[25:].mean() < 0.05  # stays out of the decline
    assert result_down.equity[-1] > result_down.benchmark[-1]


def test_momentum_positions_match_trailing_return_sign():
    closes = np.concatenate([np.linspace(100, 150, 200), np.linspace(150, 90, 200)])
    result = run_backtest(closes, Momentum(lookback=50))
    assert result.weights[60:190].mean() > 0.9   # long during the rally
    assert result.weights[300:].mean() < 0.1     # flat deep into the selloff


def test_mean_reversion_enters_after_dip_and_exits_on_recovery():
    flat = np.full(60, 100.0) + np.sin(np.arange(60)) * 0.5  # gentle noise
    dip = np.concatenate([flat, [92.0, 91.0], np.linspace(92, 101, 20)])
    result = run_backtest(dip, MeanReversion(window=20))
    assert result.weights[60:64].max() == 1.0    # bought the dip
    assert result.weights[-3:].max() == 0.0      # exited after recovery


def test_costs_scale_with_turnover():
    closes = synthetic_closes()

    class Churner(Strategy):  # alternates daily: worst-case turnover
        name = "churner"
        warmup = 0

        def target_weight(self, closes):
            return float(len(closes) % 2)

    cheap = run_backtest(closes, Churner(), cost_bps=0.0)
    pricey = run_backtest(closes, Churner(), cost_bps=20.0)
    assert pricey.equity[-1] < cheap.equity[-1] * 0.9


def test_max_drawdown_curve():
    assert max_drawdown_curve(np.array([100.0, 110, 120, 130])) == 0.0
    assert max_drawdown_curve(np.array([100.0, 50, 75])) == pytest.approx(-0.5)


def test_metrics_sanity():
    result = run_backtest(synthetic_closes(), SMACrossover(fast=10, slow=50))
    m = result.metrics
    assert -1 <= m["max_drawdown"] <= 0
    assert 0 <= m["exposure"] <= 1
    assert 0 <= m["win_rate"] <= 1
    assert m["n_trades"] >= 1
    assert set(m["benchmark"]) == {
        "total_return", "cagr", "volatility", "sharpe", "max_drawdown",
    }


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        run_backtest(np.array([100.0, 101, 102]), SMACrossover(fast=5, slow=20))


def test_registry():
    assert make_strategy("sma", fast=5, slow=10).warmup == 10
    with pytest.raises(ValueError):
        make_strategy("nope")

import json

import numpy as np
import pytest

from quantsim import (
    MeanReversion,
    PaperBroker,
    SMACrossover,
    load_history,
    run_backtest,
    run_cycle,
)
from quantsim.report import render_report


def uptrend(n=200, start=100.0):
    return [start * (1.004 ** i) for i in range(n)]


def downtrend(n=200, start=100.0):
    return [start * (0.996 ** i) for i in range(n)]


def make_broker(tmp_path, cash=100_000.0):
    return PaperBroker(tmp_path / "portfolio.json", initial_cash=cash)


def test_enters_long_in_uptrend(tmp_path):
    broker = make_broker(tmp_path)
    result = run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    assert result.weight == 1.0
    assert result.order_qty > 0
    assert broker.position("SPY") == result.target_qty
    assert result.target_qty * result.price <= result.equity
    assert result.equity == pytest.approx(100_000.0)  # cash conserved through fill


def test_stays_flat_in_downtrend(tmp_path):
    broker = make_broker(tmp_path)
    result = run_cycle(broker, downtrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    assert result.weight == 0.0
    assert result.order_qty == 0
    assert broker.position("SPY") == 0


def test_idempotent_second_run_places_no_order(tmp_path):
    broker = make_broker(tmp_path)
    first = run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    second = run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    assert first.order_qty > 0
    assert second.order_qty == 0


def test_exits_when_trend_flips(tmp_path):
    broker = make_broker(tmp_path)
    run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    assert broker.position("SPY") > 0
    flipped = uptrend() + downtrend(120, start=uptrend()[-1])
    result = run_cycle(broker, flipped, "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    assert result.weight == 0.0
    assert broker.position("SPY") == 0
    assert broker.state["cash"] == pytest.approx(broker.equity())


def test_live_weight_matches_backtests_final_weight(tmp_path):
    """The core promise of the stack: deploy exactly what you backtested —
    including stateful strategies, thanks to the history replay."""
    closes = np.concatenate([
        np.full(60, 100.0) + np.sin(np.arange(60)) * 0.5,
        [92.0, 91.0, 90.5],
    ])
    strategy = MeanReversion(window=20)
    backtest_weight = run_backtest(closes, MeanReversion(window=20)).weights[-1]
    live = run_cycle(make_broker(tmp_path), closes, "XYZ", strategy, state_dir=tmp_path)
    assert live.weight == backtest_weight == 1.0  # both in the dip-buying state


def test_state_persists_and_history_appends(tmp_path):
    broker = make_broker(tmp_path)
    run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    reloaded = PaperBroker(tmp_path / "portfolio.json")
    assert reloaded.position("SPY") > 0
    history = load_history(tmp_path)
    assert len(history) == 2
    raw = (tmp_path / "history.jsonl").read_text().splitlines()
    assert all(json.loads(line) for line in raw)


def test_short_history_raises(tmp_path):
    with pytest.raises(ValueError):
        run_cycle(make_broker(tmp_path), uptrend(20), "SPY", SMACrossover(10, 50),
                  state_dir=tmp_path)


def test_report_renders_svg_and_handles_empty(tmp_path):
    broker = make_broker(tmp_path)
    run_cycle(broker, uptrend(), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    run_cycle(broker, uptrend(220), "SPY", SMACrossover(10, 50), state_dir=tmp_path)
    out = tmp_path / "report.html"
    render_report(load_history(tmp_path), out)
    html = out.read_text()
    assert "<svg" in html and "total return" in html and "SPY" in html
    render_report([], tmp_path / "empty.html")
    assert "No cycles recorded" in (tmp_path / "empty.html").read_text()

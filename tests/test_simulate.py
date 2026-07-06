import numpy as np
import pytest

from quantsim import ascii_histogram, max_drawdown, simulate_gbm, summarize


def test_shape_and_initial_value():
    paths = simulate_gbm(100, 0.07, 0.2, years=2, steps_per_year=252, n_paths=500, seed=1)
    assert paths.shape == (500, 505)
    assert np.allclose(paths[:, 0], 100)
    assert (paths > 0).all()


def test_seed_reproducibility():
    a = simulate_gbm(100, 0.05, 0.15, years=1, n_paths=200, seed=42)
    b = simulate_gbm(100, 0.05, 0.15, years=1, n_paths=200, seed=42)
    assert np.array_equal(a, b)


def test_zero_volatility_is_deterministic_growth():
    paths = simulate_gbm(100, 0.05, 0.0, years=1, n_paths=10, seed=0)
    assert np.allclose(paths[:, -1], 100 * np.exp(0.05), rtol=1e-10)


def test_drift_matches_expectation():
    # E[S_T] = s0 * exp(mu * T) for GBM
    paths = simulate_gbm(100, 0.07, 0.2, years=5, n_paths=200_000, seed=7)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(0.07 * 5), rel=0.02)


def test_max_drawdown_is_non_positive():
    paths = simulate_gbm(100, 0.07, 0.3, years=3, n_paths=1_000, seed=3)
    dd = max_drawdown(paths)
    assert dd.shape == (1_000,)
    assert (dd <= 0).all()
    assert (dd >= -1).all()


def test_summarize_keys_and_ranges():
    paths = simulate_gbm(10_000, 0.07, 0.18, years=10, n_paths=2_000, seed=11)
    s = summarize(paths, years=10)
    assert s["n_paths"] == 2_000
    assert 0 <= s["prob_loss"] <= 1
    assert s["cvar_95"] <= s["var_95"]
    assert s["p5_final"] <= s["median_final"] <= s["p95_final"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_gbm(-1, 0.05, 0.2, years=1)
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, -0.2, years=1)
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, 0.2, years=0)


def test_ascii_histogram_renders_all_bins():
    values = np.random.default_rng(0).normal(0, 1, 1_000)
    out = ascii_histogram(values, bins=10)
    assert len(out.splitlines()) == 10

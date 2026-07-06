# 📈 quantsim

A **Monte Carlo market simulator** for your terminal. Simulate thousands of geometric Brownian motion price paths and get institutional-style risk analytics — **VaR, CVaR, Sharpe ratio, max drawdown** — plus an ASCII histogram of outcomes, in under a second.

Built on NumPy. One dependency, fully vectorized, fully tested.

## 🚀 Quickstart

```bash
pip install git+https://github.com/hilothefunnydog123-coder/quantsim.git
quantsim --initial 10000 --mu 0.07 --sigma 0.18 --years 10 --paths 10000
```

```
quantsim — 10,000 GBM paths · μ=+7.00% σ=18.0% · 10y horizon

Outcomes (start: 10,000)
  median final value              17,197
  5th–95th percentile              7,314 – 42,927
  mean CAGR                       +5.36%
  P(end below start)               17.1%

Risk
  VaR 95% (total return)         -26.86%
  CVaR 95%                       -37.85%
  Sharpe (vs rf=+3.00%)             0.40
  median max drawdown            -33.42%
  worst max drawdown             -75.10%

Distribution of final values
         3,061 │████████                                    │ 512
         9,270 │████████████████████████████████████████████│ 3,204
        ...
```

*(numbers vary run to run — pass `--seed 42` for reproducible output)*

## 🐍 Use it as a library

```python
from quantsim import simulate_gbm, summarize

paths = simulate_gbm(s0=10_000, mu=0.07, sigma=0.18, years=10, n_paths=50_000, seed=42)
stats = summarize(paths, years=10)

print(stats["var_95"])              # 95% value-at-risk on total return
print(stats["median_max_drawdown"]) # median worst peak-to-trough decline
```

`simulate_gbm` returns a `(n_paths, n_steps + 1)` NumPy array, so everything composes with the scientific Python stack — plot it, feed it to pandas, whatever you need.

## 🧮 The math

Paths follow the exact log-space GBM solution, so there is no discretization bias at any step size:

```
S(t+Δt) = S(t) · exp[(μ − σ²/2)·Δt + σ·√Δt·Z],   Z ~ N(0, 1)
```

- **VaR 95%** — the 5th percentile of total return: the loss you'd exceed only 1-in-20 times
- **CVaR 95%** — the *average* loss inside that worst 5% tail (always ≤ VaR)
- **Max drawdown** — worst peak-to-trough decline along each path, computed with a vectorized running maximum
- **Sharpe** — mean excess CAGR over the risk-free rate, divided by CAGR dispersion

## ✅ Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite checks statistical correctness, not just plumbing — e.g. that the simulated mean converges to the analytical `E[S_T] = S₀·e^(μT)` and that zero volatility collapses to deterministic growth.

## 📄 License

MIT

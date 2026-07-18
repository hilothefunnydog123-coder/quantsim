"""Command-line interface for quantsim: `quantsim mc` and `quantsim backtest`."""
from __future__ import annotations

import argparse

from quantsim.simulate import ascii_histogram, simulate_gbm, summarize

BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _run_mc(args: argparse.Namespace) -> None:
    paths = simulate_gbm(
        args.initial, args.mu, args.sigma, args.years, n_paths=args.paths, seed=args.seed
    )
    s = summarize(paths, args.years, risk_free=args.risk_free)

    print(f"\n{BOLD}{CYAN}quantsim mc{RESET} — {s['n_paths']:,} GBM paths · "
          f"μ={_pct(args.mu)} σ={args.sigma * 100:.1f}% · {args.years:g}y horizon\n")

    print(f"{BOLD}Outcomes{RESET} (start: {args.initial:,.0f})")
    print(f"  median final value      {s['median_final']:>14,.0f}")
    print(f"  5th–95th percentile     {s['p5_final']:>14,.0f} – {s['p95_final']:,.0f}")
    print(f"  mean CAGR               {_pct(s['mean_cagr']):>14}")
    print(f"  P(end below start)      {s['prob_loss'] * 100:>13.1f}%\n")

    print(f"{BOLD}Risk{RESET}")
    print(f"  VaR 95% (total return)  {_pct(s['var_95']):>14}")
    print(f"  CVaR 95%                {_pct(s['cvar_95']):>14}")
    print(f"  {f'Sharpe (rf {_pct(args.risk_free)})':<24}{s['sharpe']:>14.2f}")
    print(f"  median max drawdown     {_pct(s['median_max_drawdown']):>14}")
    print(f"  worst max drawdown      {_pct(s['worst_max_drawdown']):>14}\n")

    print(f"{BOLD}Distribution of final values{RESET}")
    print(f"{DIM}{ascii_histogram(paths[:, -1])}{RESET}\n")


def _run_backtest(args: argparse.Namespace) -> None:
    from quantsim.backtest import run_backtest
    from quantsim.data import fetch, load_csv
    from quantsim.strategies import make_strategy

    kwargs = {}
    if args.strategy == "sma":
        kwargs = {"fast": args.fast, "slow": args.slow}
    elif args.strategy == "momentum":
        kwargs = {"lookback": args.lookback}
    elif args.strategy == "meanrev":
        kwargs = {"window": args.window}
    strategy = make_strategy(args.strategy, **kwargs)

    if args.csv:
        series = load_csv(args.csv)
    else:
        print(f"{DIM}fetching {args.symbol} daily history…{RESET}")
        series = fetch(args.symbol, start=args.start)

    result = run_backtest(
        series.closes, strategy, dates=series.dates,
        initial=args.initial, cost_bps=args.cost_bps,
    )
    m = result.metrics
    b = m["benchmark"]
    label = args.csv or args.symbol

    print(f"\n{BOLD}{CYAN}quantsim backtest{RESET} — {strategy.name} on {label} · "
          f"{len(series):,} bars · {series.dates[0]} → {series.dates[-1]} · "
          f"costs {args.cost_bps:g} bps\n")
    print(f"{BOLD}{'metric':<16}{strategy.name:>14}{'buy & hold':>14}{RESET}")
    rows = [
        ("total return", _pct(m["total_return"]), _pct(b["total_return"])),
        ("CAGR", _pct(m["cagr"]), _pct(b["cagr"])),
        ("volatility", _pct(m["volatility"]), _pct(b["volatility"])),
        ("Sharpe (rf=0)", f"{m['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
        ("max drawdown", _pct(m["max_drawdown"]), _pct(b["max_drawdown"])),
        ("exposure", f"{m['exposure'] * 100:.0f}%", "100%"),
        ("trades", str(m["n_trades"]), "1"),
    ]
    for name, sv, bv in rows:
        print(f"{name:<16}{sv:>14}{bv:>14}")
    print(f"\nfinal value: {BOLD}{result.equity[-1]:,.0f}{RESET} "
          f"{DIM}(buy & hold: {result.benchmark[-1]:,.0f}, "
          f"started with {args.initial:,.0f}){RESET}")

    if args.plot:
        from quantsim.plot import plot_backtest
        plot_backtest(result, args.plot, title=f"{strategy.name} vs buy & hold — {label}")
        print(f"chart saved to {args.plot}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantsim",
        description="Monte Carlo simulation and strategy backtesting in your terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mc = sub.add_parser("mc", help="Monte Carlo GBM simulation with risk metrics")
    mc.add_argument("--initial", type=float, default=10_000)
    mc.add_argument("--mu", type=float, default=0.07, help="expected annual return")
    mc.add_argument("--sigma", type=float, default=0.18, help="annual volatility")
    mc.add_argument("--years", type=float, default=10)
    mc.add_argument("--paths", type=int, default=10_000)
    mc.add_argument("--risk-free", type=float, default=0.03)
    mc.add_argument("--seed", type=int, default=None)
    mc.set_defaults(func=_run_mc)

    bt = sub.add_parser("backtest", help="backtest a strategy on real market data")
    bt.add_argument("--symbol", default="SPY", help="ticker to fetch (default SPY)")
    bt.add_argument("--csv", help="backtest a local CSV (Date,Close columns) instead")
    bt.add_argument("--start", default="2015-01-01", help="history start date")
    bt.add_argument("--strategy", default="sma",
                    choices=["buyhold", "sma", "momentum", "meanrev"])
    bt.add_argument("--fast", type=int, default=20, help="sma: fast window")
    bt.add_argument("--slow", type=int, default=100, help="sma: slow window")
    bt.add_argument("--lookback", type=int, default=126, help="momentum: lookback bars")
    bt.add_argument("--window", type=int, default=20, help="meanrev: z-score window")
    bt.add_argument("--initial", type=float, default=10_000)
    bt.add_argument("--cost-bps", type=float, default=1.0,
                    help="transaction cost in basis points of turnover")
    bt.add_argument("--plot", metavar="PATH.png",
                    help="save an equity/drawdown chart (requires matplotlib)")
    bt.set_defaults(func=_run_backtest)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

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


def _strategy_from_args(args: argparse.Namespace):
    from quantsim.strategies import make_strategy

    kwargs = {}
    if args.strategy == "sma":
        kwargs = {"fast": args.fast, "slow": args.slow}
    elif args.strategy == "momentum":
        kwargs = {"lookback": args.lookback}
    elif args.strategy == "meanrev":
        kwargs = {"window": args.window}
    return make_strategy(args.strategy, **kwargs)


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", default="sma",
                        choices=["buyhold", "sma", "momentum", "meanrev"])
    parser.add_argument("--fast", type=int, default=20, help="sma: fast window")
    parser.add_argument("--slow", type=int, default=100, help="sma: slow window")
    parser.add_argument("--lookback", type=int, default=126, help="momentum: lookback bars")
    parser.add_argument("--window", type=int, default=20, help="meanrev: z-score window")


def _run_backtest(args: argparse.Namespace) -> None:
    from quantsim.backtest import run_backtest
    from quantsim.data import fetch, load_csv

    strategy = _strategy_from_args(args)

    if args.csv:
        series = load_csv(args.csv)
    else:
        print(f"{DIM}fetching {args.symbol} daily history…{RESET}")
        series = fetch(args.symbol, start=args.start)

    execution = None
    cost_label = f"costs {args.cost_bps:g} bps"
    if args.execution == "book":
        from quantsim.execution import BookExecution
        execution = BookExecution(level_shares=args.level_shares)
        cost_label = f"costs via order book (level_shares={args.level_shares:g})"

    result = run_backtest(
        series.closes, strategy, dates=series.dates,
        initial=args.initial, cost_bps=args.cost_bps, execution=execution,
    )
    m = result.metrics
    b = m["benchmark"]
    label = args.csv or args.symbol

    print(f"\n{BOLD}{CYAN}quantsim backtest{RESET} — {strategy.name} on {label} · "
          f"{len(series):,} bars · {series.dates[0]} → {series.dates[-1]} · "
          f"{cost_label}\n")
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


def _live_run(args: argparse.Namespace) -> None:
    from pathlib import Path

    from quantsim.broker import AlpacaBroker, PaperBroker
    from quantsim.data import fetch, load_csv
    from quantsim.live import run_cycle

    strategy = _strategy_from_args(args)
    if args.csv:
        series = load_csv(args.csv)
    else:
        print(f"{DIM}fetching {args.symbol} daily history…{RESET}")
        series = fetch(args.symbol)

    if args.live:
        broker = AlpacaBroker()
        mode = "alpaca-paper"
    else:
        broker = PaperBroker(Path(args.state_dir) / "portfolio.json",
                             initial_cash=args.initial)
        mode = "local-sim"

    result = run_cycle(broker, series.closes, args.symbol, strategy,
                       state_dir=args.state_dir)
    action = (f"{'BUY' if result.order_qty > 0 else 'SELL'} "
              f"{abs(result.order_qty):g} {result.symbol}"
              if result.order_qty else "HOLD (already at target)")
    print(f"{BOLD}quantsim live{RESET} [{mode}] {result.ts}")
    print(f"  {result.symbol} @ {result.price:,.2f} · {strategy.name} weight {result.weight:g}")
    print(f"  position {result.current_qty:g} → target {result.target_qty} · {BOLD}{action}{RESET}")
    print(f"  equity: {BOLD}{result.equity:,.2f}{RESET}")


def _live_status(args: argparse.Namespace) -> None:
    from quantsim.live import load_history

    history = load_history(args.state_dir)
    if not history:
        print("no cycles recorded yet — run `quantsim live run` first")
        return
    first, last = history[0], history[-1]
    total = last["equity"] / first["equity"] - 1.0
    print(f"{BOLD}quantsim live status{RESET} — {len(history)} cycle(s), "
          f"{first['ts'][:10]} → {last['ts'][:10]}")
    print(f"  equity {BOLD}{last['equity']:,.2f}{RESET} ({total * 100:+.2f}% since start)")
    print(f"  position {last['target_qty']:g} {last['symbol']} · "
          f"last signal weight {last['weight']:g}")
    for entry in history[-min(args.tail, len(history)):]:
        order = (f"{'+' if entry['order_qty'] > 0 else ''}{entry['order_qty']:g}"
                 if entry["order_qty"] else "·")
        print(f"  {DIM}{entry['ts'][:16]}  {entry['symbol']} @ {entry['price']:>10,.2f}  "
              f"w={entry['weight']:g}  order {order:>6}  eq {entry['equity']:>12,.2f}{RESET}")


def _live_report(args: argparse.Namespace) -> None:
    from quantsim.live import load_history
    from quantsim.report import render_report

    render_report(load_history(args.state_dir), args.out)
    print(f"report written to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantsim",
        description="A full quant stack in your terminal: Monte Carlo simulation, "
        "strategy backtesting with order-book execution, and live paper trading.",
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
    _add_strategy_args(bt)
    bt.add_argument("--initial", type=float, default=10_000)
    bt.add_argument("--cost-bps", type=float, default=1.0,
                    help="transaction cost in basis points of turnover")
    bt.add_argument("--execution", choices=["flat", "book"], default="flat",
                    help="'book' prices each trade by walking a synthetic "
                         "limit order book (market impact) instead of flat bps")
    bt.add_argument("--level-shares", type=float, default=10_000,
                    help="book execution: liquidity per price level, in shares")
    bt.add_argument("--plot", metavar="PATH.png",
                    help="save an equity/drawdown chart (requires matplotlib)")
    bt.set_defaults(func=_run_backtest)

    live = sub.add_parser(
        "live", help="paper-trade the same strategies (local sim or Alpaca)"
    )
    live_sub = live.add_subparsers(dest="live_command", required=True)

    run_p = live_sub.add_parser("run", help="execute one trading cycle")
    run_p.add_argument("--symbol", default="SPY")
    _add_strategy_args(run_p)
    run_p.add_argument("--csv", help="use a local Date,Close CSV instead of fetching")
    run_p.add_argument("--live", action="store_true",
                       help="trade the Alpaca paper account (needs APCA_* env vars)")
    run_p.add_argument("--state-dir", default="state")
    run_p.add_argument("--initial", type=float, default=100_000,
                       help="starting cash for the local simulator")
    run_p.set_defaults(func=_live_run)

    status_p = live_sub.add_parser("status", help="show position and P&L history")
    status_p.add_argument("--state-dir", default="state")
    status_p.add_argument("--tail", type=int, default=10)
    status_p.set_defaults(func=_live_status)

    report_p = live_sub.add_parser("report", help="render the HTML P&L report")
    report_p.add_argument("--state-dir", default="state")
    report_p.add_argument("--out", default="docs/report.html")
    report_p.set_defaults(func=_live_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Command-line interface for quantsim."""
from __future__ import annotations

import argparse

from quantsim.simulate import ascii_histogram, simulate_gbm, summarize

BOLD = "\033[1m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="quantsim",
        description="Monte Carlo market simulator: GBM price paths with VaR/CVaR, "
        "Sharpe and max-drawdown analytics.",
    )
    parser.add_argument("--initial", type=float, default=10_000, help="starting portfolio value")
    parser.add_argument("--mu", type=float, default=0.07, help="expected annual return (e.g. 0.07)")
    parser.add_argument("--sigma", type=float, default=0.18, help="annual volatility (e.g. 0.18)")
    parser.add_argument("--years", type=float, default=10, help="investment horizon in years")
    parser.add_argument("--paths", type=int, default=10_000, help="number of simulated paths")
    parser.add_argument("--risk-free", type=float, default=0.03, help="annual risk-free rate")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = parser.parse_args(argv)

    paths = simulate_gbm(
        args.initial, args.mu, args.sigma, args.years, n_paths=args.paths, seed=args.seed
    )
    s = summarize(paths, args.years, risk_free=args.risk_free)

    print(f"\n{BOLD}{CYAN}quantsim{RESET} — {s['n_paths']:,} GBM paths · "
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


if __name__ == "__main__":
    main()

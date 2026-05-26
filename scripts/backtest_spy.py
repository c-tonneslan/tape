"""Backtest the mean reversion strategy.

Usage:
    .venv/bin/python scripts/backtest_spy.py
    .venv/bin/python scripts/backtest_spy.py --equity 1500
    .venv/bin/python scripts/backtest_spy.py --equity 1500 --symbols SLV,GLD,XLF,KRE,XOP
    .venv/bin/python scripts/backtest_spy.py --equity 1500 --years 3 --csv equity.csv
    .venv/bin/python scripts/backtest_spy.py --symbols SPY --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)

# Self-bootstrap sys.path so we run on the macOS-hidden .pth quirk too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tape.backtest import backtest  # noqa: E402
from tape.config import load_settings  # noqa: E402
from tape.data.market_data import MarketDataProvider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="SLV,GLD,XLF,KRE,XOP",
                        help="comma-separated tickers (cheap ETFs work best for a small account)")
    parser.add_argument("--equity", type=float, default=1500.0)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--start", type=_parse_date,
                        help="YYYY-MM-DD; overrides --years if given with --end")
    parser.add_argument("--end", type=_parse_date,
                        help="YYYY-MM-DD; overrides --years if given with --start")
    parser.add_argument("--csv", help="dump the daily equity curve to this path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    settings = load_settings()
    provider = MarketDataProvider(settings)

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=365 * args.years)

    bars_by_symbol = {}
    for sym in symbols:
        print(f"fetching {sym} {start.date()} -> {end.date()}...", end=" ", flush=True)
        bars = provider.historical_daily_bars(sym, start=start, end=end)
        print(f"{len(bars)} bars")
        if not bars.empty:
            bars_by_symbol[sym] = bars

    if not bars_by_symbol:
        print("no data fetched")
        return 1

    print()
    result = backtest(bars_by_symbol, starting_equity=args.equity)

    span_days = (end - start).days
    print(f"mean reversion, ${args.equity:,.0f} starting, "
          f"{start.date()} -> {end.date()} ({span_days / 365:.1f}y), "
          f"{', '.join(bars_by_symbol)}:")
    print(result.summary())

    monthly = result.monthly_equity().dropna()
    if not monthly.empty:
        print("\nmonth-end equity:")
        prev = args.equity
        for ts, eq in monthly.items():
            delta = eq - prev
            sign = "+" if delta >= 0 else "-"
            print(f"  {ts.date()}  ${eq:>9,.2f}   "
                  f"{sign}${abs(delta):>7,.2f}   "
                  f"({(eq / args.equity - 1):+.2%} from start)")
            prev = eq

    if result.trades:
        print(f"\nall trades ({len(result.trades)}):")
        for t in result.trades:
            print(
                f"  {t.entry_time.date()} -> {t.exit_time.date()}  "
                f"{t.symbol:<4}  qty {t.qty:>3}  "
                f"entry {t.entry_price:>7.2f}  exit {t.exit_price:>7.2f}  "
                f"P/L ${t.pnl:>+7.2f}  ({t.exit_reason})"
            )

    if args.csv:
        result.equity_series().to_csv(args.csv, header=True)
        print(f"\nwrote daily equity curve to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke test for the Alpaca paper connection.

Loads settings from .env, asks the data provider for SPY's most recent
daily bar, and prints it. Useful as a "does this even work" check before
any strategy or risk work.
"""

from __future__ import annotations

from tape.config import load_settings
from tape.data import MarketDataProvider


def main() -> int:
    settings = load_settings()
    provider = MarketDataProvider(settings)

    symbol = "SPY"
    bar = provider.latest_daily_bar(symbol)
    if bar is None:
        print(f"No recent daily bar for {symbol}. Check your network / Alpaca status.")
        return 1

    print(f"{symbol} latest daily bar ({bar.timestamp:%Y-%m-%d}):")
    print(f"  open  = {bar.open:,.2f}")
    print(f"  high  = {bar.high:,.2f}")
    print(f"  low   = {bar.low:,.2f}")
    print(f"  close = {bar.close:,.2f}")
    print(f"  vol   = {bar.volume:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

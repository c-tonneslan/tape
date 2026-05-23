"""Smoke test for the Alpaca paper connection.

Loads settings from .env, asks the data provider for SPY's most recent
daily bar, and prints it. Useful as a "does this even work" check before
any strategy or risk work.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make this script work even if the editable install isn't visible. macOS
# sets UF_HIDDEN on hatchling's editable .pth file, which Python 3.13's
# site.py treats as "hidden" and skips, so `import tape` fails when running
# the script outside a context that already has the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tape.config import load_settings  # noqa: E402
from tape.data import MarketDataProvider  # noqa: E402


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

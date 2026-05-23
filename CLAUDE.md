# Project: tape — paper-trading bot

## What this is

A Python paper-trading bot built against the Alpaca paper API. The bot is
built iteratively. It will never touch a live trading account until
explicitly authorized in this file.

## Trading constraints (the bot's constitution)

- **Paper trading only.** The bot refuses to start if the configured Alpaca
  base URL isn't the paper URL.
- **Max risk per trade:** 1% of account value (computed from stop distance,
  not from notional).
- **Max total exposure:** 20% of account across all open positions.
- **Stop loss:** 2× ATR(14) below entry for longs. Shorts paper-only.
- **Daily loss limit:** 3% of start-of-day equity → halt trading for the day.
- **Max drawdown circuit breaker:** 15% from peak → halt indefinitely, require
  manual review.
- **Kill switch:** operable externally; closes all positions, cancels all
  orders, sets a halted flag.
- **Watchlist:** start with SPY, QQQ, IWM (liquid US ETFs). Expand later.
- **First strategy:** mean reversion. RSI(14) below 30 long, confirmation that
  price is 1.5+ standard deviations below the 20-day moving average.

## Code rules

- Python 3.11+.
- Type hints on public functions.
- Pytest for any new logic. No untested logic merged.
- All numeric trading parameters in `tape/config/settings.py`, never
  hardcoded in strategy/risk/execution modules.
- API keys live in `.env`, never committed.
- The risk manager mediates every order. Strategy modules never call the
  execution engine directly.

## Architecture (six modules)

```
tape/
  config/      settings, parameters
  data/        market data fetching (Alpaca + yfinance fallback)
  strategies/  signal logic (pure functions, no side effects)
  risk/        position sizing, pre-trade checks, kill switch
  execution/   broker API interaction
  journal/     trade logging and reports
tests/         unit tests for everything in tape/
scripts/       one-off scripts (smoke tests, manual runs)
```

The package is named `journal/` rather than `logging/` so it doesn't
shadow Python's stdlib `logging`.

## Forbidden

- Hardcoded API keys.
- Live trading. Paper only until this line is removed.
- Any strategy with more than 6 tunable parameters.
- Removing risk checks "just to try something."
- Letting the LLM directly initiate orders. The LLM gets veto power, not
  initiation power.

## Writing style

Same as the global account rules: contractions, no em-dashes, no
"comprehensive"/"robust"/"leverage"/"streamline"/"facilitate"/"seamless", no
AI disclosure in PRs or commit messages unless a maintainer explicitly asks.

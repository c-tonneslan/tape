# tape

A Python paper-trading bot for the Alpaca paper API. Built iteratively, paper
only, with risk management at the center.

This is day-1 scaffolding. The pieces are stubbed; the architecture is
there.

## Setup

```sh
git clone https://github.com/c-tonneslan/tape.git
cd tape

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Then fill in your Alpaca paper keys from
# https://app.alpaca.markets/paper/dashboard/overview
```

## Smoke test

Once `.env` has your paper keys:

```sh
python scripts/hello_alpaca.py
```

It fetches the most recent SPY daily bar through the Alpaca paper API and
prints it. If you see the bar, the connection works.

## Layout

```
tape/
  config/      parameters
  data/        market data fetching
  strategies/  signal logic (pure functions)
  risk/        position sizing + pre-trade checks + kill switch
  execution/   broker API interaction
  journal/     trade logging
tests/         pytest
scripts/       smoke tests, manual runs
```

## Where we are

- [x] Scaffolding, package layout, env handling
- [x] Position-sizing function (1% per-trade rule)
- [x] Smoke-test script that hits Alpaca paper
- [ ] Mean-reversion strategy
- [ ] Real risk manager (daily loss limit, max drawdown, kill switch)
- [ ] Backtest harness
- [ ] Live paper trading loop
- [ ] AI-augmented daily briefing

## License

MIT

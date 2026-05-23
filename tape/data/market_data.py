"""Market data fetching through the Alpaca paper data API.

Only enough is here today to support the smoke test: fetch the latest daily
bar for a symbol. The full provider (historical ranges, streaming, caching,
yfinance fallback) lands when we need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tape.config import Settings


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. Symbol-agnostic."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider:
    """Wraps the Alpaca historical data client."""

    def __init__(self, settings: Settings) -> None:
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def latest_daily_bar(self, symbol: str) -> Bar | None:
        """Return the most recent daily bar for `symbol`, or None if there is
        no bar in the last ten calendar days (long weekends, holidays).
        """
        # Alpaca's free historical feed has a ~15-minute SIP delay, so asking
        # for "right now" can return nothing on intraday queries. Daily bars
        # are safer for a smoke test, and a 10-day window covers any holiday
        # cluster.
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=10)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        response = self._client.get_stock_bars(request)
        bars = response.data.get(symbol, [])
        if not bars:
            return None

        last = bars[-1]
        return Bar(
            symbol=symbol,
            timestamp=last.timestamp,
            open=float(last.open),
            high=float(last.high),
            low=float(last.low),
            close=float(last.close),
            volume=float(last.volume),
        )

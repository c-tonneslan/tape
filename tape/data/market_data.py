"""Market data fetching through the Alpaca paper data API.

Live + recent daily bars come from Alpaca (paper account, IEX feed). For
older history we fall back to yfinance, because the free paper-account
IEX feed only goes back to roughly mid-2020 and a backtest that wants
2018 needs to come from somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tape.config import Settings

# When Alpaca's first bar comes back significantly after the requested
# start, or when there's a hole inside the Alpaca series bigger than this,
# we treat the gap as "old enough to fetch from yfinance." A week of
# slack covers weekends, holidays, and slow settlement without spuriously
# firing the fallback.
_ALPACA_GAP_TOLERANCE = timedelta(days=7)


def _alpaca_needs_help(df: pd.DataFrame, *, start: datetime) -> bool:
    """True if the Alpaca response is missing enough of the requested
    range that the yfinance fallback is worth running.
    """
    if df.empty:
        return True
    if df.index[0] - start.astimezone(UTC) > _ALPACA_GAP_TOLERANCE:
        return True
    if len(df) >= 2:
        diffs = df.index.to_series().diff().iloc[1:]
        if (diffs > _ALPACA_GAP_TOLERANCE).any():
            return True
    return False


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
        # Free Alpaca paper accounts can't query recent SIP data (the
        # consolidated feed) — the API returns "subscription does not permit
        # querying recent SIP data". Use the IEX feed, which is always
        # available on free accounts and is fine for the smoke test and any
        # daily-bar work. We can switch to SIP later if/when the account
        # subscription supports it.
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=10)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
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

    def historical_daily_bars(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Daily OHLCV bars for `symbol` between `start` and `end`.

        Returns a DataFrame indexed by timestamp with open/high/low/close/volume
        columns, sorted ascending. Empty DataFrame if neither source has
        data for the range.

        Strategy: ask Alpaca first. If Alpaca came back empty, has a late
        first bar, or has an internal hole bigger than the tolerance, fetch
        the full range from yfinance and use it to fill the dates Alpaca
        doesn't have. Live/recent data still flows through Alpaca for
        paper-trading parity; 2018 bars come from yfinance.
        """
        df = self._alpaca_bars(symbol, start=start, end=end)

        if not _alpaca_needs_help(df, start=start):
            return df

        yfin = self._yfinance_bars(symbol, start=start, end=end)
        if yfin.empty:
            return df
        if df.empty:
            return yfin

        # Prefer Alpaca where both sources have a date; use yfinance only
        # to plug holes. Compare normalized (midnight-UTC) dates because
        # yfinance returns date-only timestamps and Alpaca returns the
        # session open in Eastern time, so a raw .isin would treat the
        # same calendar day as two different rows.
        alpaca_dates = df.index.normalize()
        yfin_dates = yfin.index.normalize()
        gap_filler = yfin[~yfin_dates.isin(alpaca_dates)]
        return pd.concat([df, gap_filler]).sort_index()

    def _alpaca_bars(
        self, symbol: str, *, start: datetime, end: datetime
    ) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        response = self._client.get_stock_bars(request)
        df = response.df
        if df.empty:
            return df
        # Alpaca returns a MultiIndex (symbol, timestamp). Drop the symbol
        # level for single-symbol queries so the caller gets a flat
        # timestamp-indexed DataFrame.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def _yfinance_bars(
        self, symbol: str, *, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """yfinance daily bars normalized to look like an Alpaca response."""
        df = yf.download(
            symbol,
            start=start.date(),
            end=end.date(),
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return df
        # yfinance hands back a MultiIndex on columns even for a single
        # ticker. Flatten and lowercase so the schema matches Alpaca.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df.columns = [str(c).lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        # yfinance returns a tz-naive DatetimeIndex; localize to UTC so it
        # plays with Alpaca's tz-aware timestamps in a concat.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

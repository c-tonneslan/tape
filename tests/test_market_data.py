"""Tests for the historical-bar fallback logic.

These don't hit the network. The Alpaca side is stubbed by monkeypatching
the private `_alpaca_bars` helper, and the yfinance side is stubbed by
patching `yf.download` at the module level.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tape.data import market_data
from tape.data.market_data import MarketDataProvider


@pytest.fixture()
def provider(monkeypatch):
    """A MarketDataProvider whose Alpaca client constructor is stubbed
    out (we never use the real client in these tests).
    """

    class _StubClient:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(market_data, "StockHistoricalDataClient", _StubClient)

    class _StubSettings:
        alpaca_api_key = "test"
        alpaca_secret_key = "test"

    return MarketDataProvider(_StubSettings())


def _alpaca_frame(closes: list[float], start: str) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(closes), tz="UTC")
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr, "low": arr, "close": arr, "volume": np.full_like(arr, 1.0)},
        index=idx,
    )


def _yfinance_raw_frame(closes: list[float], start: str) -> pd.DataFrame:
    """Same shape yfinance returns: capitalized columns with a (col, ticker)
    MultiIndex on columns, tz-naive index. Used to feed the patched
    yf.download.
    """
    idx = pd.bdate_range(start=start, periods=len(closes))
    arr = np.asarray(closes, dtype=float)
    df = pd.DataFrame(
        {
            "Open": arr,
            "High": arr,
            "Low": arr,
            "Close": arr,
            "Adj Close": arr,
            "Volume": np.full_like(arr, 1.0),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_product([df.columns, ["SPY"]])
    return df


def test_alpaca_full_coverage_skips_yfinance(provider, monkeypatch):
    start = datetime(2024, 1, 2, tzinfo=UTC)
    end = datetime(2024, 1, 10, tzinfo=UTC)
    alpaca = _alpaca_frame([100.0, 101.0, 102.0, 103.0, 104.0], start="2024-01-02")
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: alpaca)

    def _fail(*_, **__):
        raise AssertionError("yfinance should not be called when Alpaca covers the range")

    monkeypatch.setattr(market_data.yf, "download", _fail)

    out = provider.historical_daily_bars("SPY", start=start, end=end)
    assert len(out) == 5
    assert out["close"].iloc[0] == 100.0
    assert out["close"].iloc[-1] == 104.0


def test_falls_back_to_yfinance_when_alpaca_empty(provider, monkeypatch):
    start = datetime(2018, 1, 2, tzinfo=UTC)
    end = datetime(2018, 1, 10, tzinfo=UTC)
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: pd.DataFrame())

    yfin = _yfinance_raw_frame([200.0, 201.0, 202.0, 203.0, 204.0], start="2018-01-02")
    calls: list[tuple] = []

    def _yf(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return yfin

    monkeypatch.setattr(market_data.yf, "download", _yf)

    out = provider.historical_daily_bars("SPY", start=start, end=end)
    assert len(out) == 5
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].iloc[0] == 200.0
    assert out.index.tz is not None
    assert calls and calls[0][0] == "SPY"


def test_fills_pre_alpaca_prefix_with_yfinance(provider, monkeypatch):
    start = datetime(2018, 1, 2, tzinfo=UTC)
    end = datetime(2020, 8, 1, tzinfo=UTC)

    # Alpaca only covers mid-2020 onwards.
    alpaca = _alpaca_frame([300.0, 301.0, 302.0], start="2020-07-27")
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: alpaca)

    # yfinance covers the requested 2018 prefix.
    yfin = _yfinance_raw_frame([200.0, 201.0, 202.0], start="2018-01-02")
    monkeypatch.setattr(market_data.yf, "download", lambda *_, **__: yfin)

    out = provider.historical_daily_bars("SPY", start=start, end=end)
    # Three yfinance bars + three Alpaca bars, in order.
    assert len(out) == 6
    assert out["close"].iloc[0] == 200.0
    assert out["close"].iloc[-1] == 302.0
    # Sorted ascending across the join.
    assert out.index.is_monotonic_increasing


def test_returns_empty_when_both_sources_have_nothing(provider, monkeypatch):
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: pd.DataFrame())
    monkeypatch.setattr(market_data.yf, "download", lambda *_, **__: pd.DataFrame())

    out = provider.historical_daily_bars(
        "SPY",
        start=datetime(2018, 1, 2, tzinfo=UTC),
        end=datetime(2018, 1, 10, tzinfo=UTC),
    )
    assert out.empty


def test_fills_internal_alpaca_hole_with_yfinance(provider, monkeypatch):
    # Real shape we see for SPY: one stray 2018-11 bar, then nothing
    # until mid-2020. The middle hole needs to come from yfinance.
    start = datetime(2018, 1, 2, tzinfo=UTC)
    end = datetime(2020, 8, 5, tzinfo=UTC)

    alpaca_idx = pd.DatetimeIndex(["2018-11-01", "2020-07-27", "2020-07-28"], tz="UTC")
    alpaca = pd.DataFrame(
        {
            "open": [100.0, 200.0, 201.0],
            "high": [100.0, 200.0, 201.0],
            "low": [100.0, 200.0, 201.0],
            "close": [100.0, 200.0, 201.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=alpaca_idx,
    )
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: alpaca)

    # yfinance covers the full range; the dates that overlap Alpaca's
    # 2018-11-01 and 2020-07-27/28 should NOT replace the Alpaca rows.
    yfin = _yfinance_raw_frame(
        list(np.linspace(50.0, 60.0, 7)),
        start="2018-10-29",
    )
    monkeypatch.setattr(market_data.yf, "download", lambda *_, **__: yfin)

    out = provider.historical_daily_bars("SPY", start=start, end=end)

    # Alpaca rows untouched; the yfinance rows that didn't collide are
    # spliced in to plug the hole.
    assert out.loc[pd.Timestamp("2018-11-01", tz="UTC"), "close"] == 100.0
    assert out.loc[pd.Timestamp("2020-07-27", tz="UTC"), "close"] == 200.0
    assert len(out) > len(alpaca)
    assert out.index.is_monotonic_increasing


def test_overlapping_dates_are_not_double_counted(provider, monkeypatch):
    # yfinance returns its index at midnight UTC; Alpaca's daily bars are
    # stamped at the US session open (04:00 or 05:00 UTC). A naive isin
    # would treat the same calendar day as two different rows, so a year
    # of SPY would end up with ~500 bars instead of ~250.
    start = datetime(2018, 1, 2, tzinfo=UTC)
    end = datetime(2018, 1, 10, tzinfo=UTC)
    alpaca_idx = pd.DatetimeIndex(
        ["2018-01-02 05:00", "2018-01-03 05:00", "2018-01-04 05:00"], tz="UTC"
    )
    alpaca = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=alpaca_idx,
    )
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: alpaca)

    # Force the fallback path by inserting an artificial internal hole
    # past the tolerance; otherwise no yfinance call.
    monkeypatch.setattr(market_data, "_alpaca_needs_help", lambda *_, **__: True)

    yfin = _yfinance_raw_frame([50.0, 51.0, 52.0], start="2018-01-02")
    monkeypatch.setattr(market_data.yf, "download", lambda *_, **__: yfin)

    out = provider.historical_daily_bars("SPY", start=start, end=end)
    # Same three calendar days. Alpaca wins on each; yfinance contributes
    # nothing because every yfinance date is also in Alpaca after
    # normalization.
    assert len(out) == 3
    assert (out["close"] == [100.0, 101.0, 102.0]).all()


def test_small_alpaca_gap_does_not_trigger_yfinance(provider, monkeypatch):
    # Alpaca's first bar is two business days after the requested start.
    # That's within the slack tolerance (holidays, weekends), so we keep
    # the Alpaca answer as-is.
    start = datetime(2024, 1, 2, tzinfo=UTC)
    end = datetime(2024, 1, 10, tzinfo=UTC)
    alpaca = _alpaca_frame([100.0, 101.0, 102.0], start="2024-01-04")
    monkeypatch.setattr(provider, "_alpaca_bars", lambda *_, **__: alpaca)

    def _fail(*_, **__):
        raise AssertionError("yfinance should not be called for a 2-day gap")

    monkeypatch.setattr(market_data.yf, "download", _fail)
    out = provider.historical_daily_bars("SPY", start=start, end=end)
    assert len(out) == 3

"""Pure indicator functions used by strategies.

Each function takes a pandas Series (or a few) and returns a pandas Series.
No state, no I/O. Tested in isolation.

The EMAs use Wilder's smoothing (alpha = 1/period). That's the classic
choice for RSI and ATR; it matches what TradingView, ta-lib, and most
broker platforms produce, so backtest values line up with what a discretionary
trader would see on a chart.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Returns a series the same length as `close`. The first `period` values
    are NaN since RSI needs a full window before it can be computed.
    """
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # When avg_loss is 0, RSI is defined as 100 (all gains, no losses). The
    # straight division would produce inf; clamp it explicitly.
    rs = avg_gain / avg_loss
    return (100.0 - (100.0 / (1.0 + rs))).where(avg_loss != 0, 100.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (lower, middle, upper) Bollinger Bands. Middle is the simple
    moving average; the bands are `num_std` standard deviations away.
    """
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    lower = middle - num_std * std
    upper = middle + num_std * std
    return lower, middle, upper

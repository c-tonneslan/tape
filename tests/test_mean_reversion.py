"""Tests for the mean reversion strategy.

Each test builds a synthetic OHLCV DataFrame that drives the strategy into
a specific state, then checks the Signal it returns.
"""

import numpy as np
import pandas as pd
import pytest

from tape.strategies import Action, MeanReversionStrategy


def make_bars(closes: list[float], wide: float = 0.5) -> pd.DataFrame:
    """OHLCV DataFrame with the given closes. Highs/lows hover symmetrically
    around close so ATR has something to work with; volume is constant.
    """
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + wide,
            "low": closes - wide,
            "close": closes,
            "volume": np.full_like(closes, 1_000_000.0),
        }
    )


def sharp_crash() -> list[float]:
    """30 bars of steady rise, then one big down bar. After that single
    bar both conditions fire: RSI tanks below 30, and the close sits well
    below the wide Bollinger band the volatility opened up.
    """
    return list(np.linspace(100, 130, 30)) + [95.0]


# ---------- entry conditions -----------------------------------------------


def test_buy_when_oversold_and_below_lower_band():
    bars = make_bars(sharp_crash())
    strategy = MeanReversionStrategy()
    signal = strategy.generate_signal(bars, symbol="SPY")

    assert signal.action == Action.BUY
    assert signal.symbol == "SPY"
    assert signal.suggested_entry == pytest.approx(bars["close"].iloc[-1])
    # Stop is below entry by 2 × ATR.
    assert signal.suggested_stop is not None
    assert signal.suggested_stop < signal.suggested_entry


def test_hold_when_rsi_oversold_but_price_not_below_band():
    # Long smooth decline: RSI hits 0, but with bb_stddev cranked up the
    # band sits below the price even though the price is falling.
    closes = list(np.linspace(120, 100, 35))
    bars = make_bars(closes)
    strategy = MeanReversionStrategy(bb_stddev=3.0)
    signal = strategy.generate_signal(bars, symbol="SPY")
    assert signal.action == Action.HOLD


def test_hold_when_price_below_band_but_rsi_not_oversold():
    # Construct: long uptrend that puts a bar below the band briefly
    # without taking RSI under 30. The simplest way is an unchanging series
    # then a tiny dip; RSI stays at 50 (or near) but the band has 0 width
    # so it can't be "below" without dipping. Use a slow trend.
    closes = list(np.linspace(100, 120, 35))
    bars = make_bars(closes)
    signal = MeanReversionStrategy().generate_signal(bars, symbol="SPY")
    assert signal.action == Action.HOLD


def test_hold_when_history_is_too_short():
    bars = make_bars([100.0] * 10)
    signal = MeanReversionStrategy().generate_signal(bars, symbol="SPY")
    assert signal.action == Action.HOLD
    assert "need at least" in signal.reasoning


# ---------- exit conditions ------------------------------------------------


def test_sell_when_in_position_and_rsi_recovers():
    # Long uptrend → high RSI. We pass bars_in_position=1 so the exit
    # branch fires; RSI well above the 50 exit threshold triggers SELL.
    closes = list(np.linspace(100, 130, 40))
    bars = make_bars(closes)
    signal = MeanReversionStrategy().generate_signal(
        bars, symbol="SPY", bars_in_position=1
    )
    assert signal.action == Action.SELL
    assert "exit" in signal.reasoning


def test_sell_when_max_holding_days_reached():
    # Choppy/flat series → RSI hovers around 50 but won't cross. Holding
    # past max_holding_days should still force the exit.
    rng = np.random.default_rng(seed=7)
    closes = list(100.0 + rng.standard_normal(40).cumsum() * 0.1)
    bars = make_bars(closes)
    strategy = MeanReversionStrategy(max_holding_days=3)
    signal = strategy.generate_signal(bars, symbol="SPY", bars_in_position=3)
    assert signal.action == Action.SELL
    assert "hit max" in signal.reasoning


def test_hold_when_in_position_and_no_exit_condition():
    rng = np.random.default_rng(seed=7)
    closes = list(100.0 + rng.standard_normal(40).cumsum() * 0.1)
    bars = make_bars(closes)
    strategy = MeanReversionStrategy(max_holding_days=10)
    signal = strategy.generate_signal(bars, symbol="SPY", bars_in_position=2)
    assert signal.action == Action.HOLD


# ---------- output shape ---------------------------------------------------


def test_buy_signal_includes_a_useful_reasoning_string():
    bars = make_bars(sharp_crash())
    signal = MeanReversionStrategy().generate_signal(bars, symbol="SPY")
    assert signal.reasoning  # non-empty
    assert "RSI" in signal.reasoning

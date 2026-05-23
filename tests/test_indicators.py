import numpy as np
import pandas as pd

from tape.strategies.indicators import atr, bollinger_bands, rsi


def test_rsi_is_nan_until_the_window_fills():
    close = pd.Series(range(1, 50), dtype=float)
    result = rsi(close, period=14)
    # The first `period` entries can't have a value (no full window yet);
    # everything afterwards should.
    assert result.iloc[:14].isna().all()
    assert not result.iloc[14:].isna().any()


def test_rsi_is_100_when_every_change_is_a_gain():
    # Monotonically rising series → all gains, no losses → RSI = 100.
    close = pd.Series(range(1, 50), dtype=float)
    result = rsi(close, period=14)
    assert result.iloc[-1] == 100.0


def test_rsi_is_zero_when_every_change_is_a_loss():
    close = pd.Series(range(50, 1, -1), dtype=float)
    result = rsi(close, period=14)
    assert result.iloc[-1] == 0.0


def test_rsi_sits_around_50_for_alternating_up_down():
    # Each step alternates +1 then -1, so gains and losses average out.
    moves = np.array([1.0, -1.0] * 50)
    close = pd.Series(100.0 + moves.cumsum())
    result = rsi(close, period=14)
    # Won't be exactly 50 because Wilder's EMA has memory, but should be
    # close.
    assert 40.0 < result.iloc[-1] < 60.0


def test_atr_window_and_sensitivity():
    # Constant 1-wide bars → ATR converges to 1 once the window is full.
    n = 30
    high = pd.Series([2.0] * n)
    low = pd.Series([1.0] * n)
    close = pd.Series([1.5] * n)
    result = atr(high, low, close, period=14)
    # True Range is defined from bar 0 (max-with-NaN-prev-close falls back
    # to high-low), so the EMA hits 14 observations at index 13.
    assert result.iloc[:13].isna().all()
    assert abs(result.iloc[-1] - 1.0) < 1e-9


def test_bollinger_bands_have_correct_relationships():
    # Random walk; bands should be in order and centered on the SMA.
    rng = np.random.default_rng(seed=42)
    close = pd.Series(100.0 + rng.standard_normal(60).cumsum())
    lower, middle, upper = bollinger_bands(close, period=20, num_std=2.0)

    # First 19 are NaN; the rest must satisfy lower <= middle <= upper.
    valid = ~middle.isna()
    assert (lower[valid] <= middle[valid]).all()
    assert (middle[valid] <= upper[valid]).all()

    # Distance from middle to upper equals distance from middle to lower
    # (by definition of the bands).
    assert np.allclose((upper[valid] - middle[valid]), (middle[valid] - lower[valid]))


def test_bollinger_bands_collapse_to_a_point_for_constant_series():
    close = pd.Series([50.0] * 30)
    lower, middle, upper = bollinger_bands(close, period=20, num_std=2.0)
    # std is 0 for a constant series, so all three bands sit on top of each
    # other at 50.
    assert (middle.dropna() == 50.0).all()
    assert (lower.dropna() == 50.0).all()
    assert (upper.dropna() == 50.0).all()

"""Mean reversion on liquid ETFs.

Long signal:
    RSI(14) < oversold
    AND close <= lower Bollinger Band (20-day SMA, 1.5σ).

Exit (when already long):
    RSI(14) >= exit
    OR we've held the position for `max_holding_days` bars.

Stop loss is set at entry - `stop_atr_multiplier` * ATR(14). The execution
side is responsible for actually placing the stop; the strategy just
suggests where it should go.

The strategy is a pure function on the bar history. No I/O, no API calls,
no state. Position-tracking lives in the caller — the strategy is told
"are we currently in a position, and if so for how many bars" via
`bars_in_position`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Action, Signal, Strategy
from .indicators import atr, bollinger_bands, rsi


@dataclass
class MeanReversionStrategy(Strategy):
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_exit: float = 50.0
    bb_period: int = 20
    bb_stddev: float = 1.5
    atr_period: int = 14
    stop_atr_multiplier: float = 2.0
    max_holding_days: int = 5

    def generate_signal(
        self,
        bars: pd.DataFrame,
        *,
        symbol: str = "",
        bars_in_position: int | None = None,
    ) -> Signal:
        """Decide what to do given the bar history.

        Args:
            bars: time-sorted OHLCV DataFrame with at least
                `max(bb_period, rsi_period, atr_period) + 1` rows. Earlier
                bars are first; the last row is "now".
            symbol: the symbol the bars belong to. Passed through onto the
                returned Signal so the caller knows what to act on.
            bars_in_position: None when we're flat, otherwise the number of
                bars we've been holding the current long.
        """
        min_history = max(self.bb_period, self.rsi_period, self.atr_period) + 1
        if len(bars) < min_history:
            return Signal(
                action=Action.HOLD,
                symbol=symbol,
                reasoning=f"need at least {min_history} bars, got {len(bars)}",
            )

        close = bars["close"]
        rsi_series = rsi(close, period=self.rsi_period)
        lower_band, _, _ = bollinger_bands(close, period=self.bb_period, num_std=self.bb_stddev)
        atr_series = atr(bars["high"], bars["low"], close, period=self.atr_period)

        current_close = float(close.iloc[-1])
        current_rsi = float(rsi_series.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_atr = float(atr_series.iloc[-1])

        if bars_in_position is None:
            return self._maybe_enter(
                symbol=symbol,
                close=current_close,
                rsi_value=current_rsi,
                lower_band=current_lower,
                atr_value=current_atr,
            )
        return self._maybe_exit(
            symbol=symbol,
            rsi_value=current_rsi,
            bars_held=bars_in_position,
        )

    def _maybe_enter(
        self,
        *,
        symbol: str,
        close: float,
        rsi_value: float,
        lower_band: float,
        atr_value: float,
    ) -> Signal:
        oversold = rsi_value < self.rsi_oversold
        below_band = close <= lower_band
        if oversold and below_band:
            stop = close - self.stop_atr_multiplier * atr_value
            return Signal(
                action=Action.BUY,
                symbol=symbol,
                suggested_entry=close,
                suggested_stop=stop,
                reasoning=(
                    f"RSI {rsi_value:.1f} below {self.rsi_oversold}, "
                    f"close {close:.2f} at/below lower band {lower_band:.2f}"
                ),
                confidence=1.0,
            )
        return Signal(
            action=Action.HOLD,
            symbol=symbol,
            reasoning=(
                f"no entry: RSI {rsi_value:.1f} "
                f"({'oversold' if oversold else 'not oversold'}), "
                f"close {close:.2f} "
                f"({'<=' if below_band else '>'}lower band {lower_band:.2f})"
            ),
        )

    def _maybe_exit(self, *, symbol: str, rsi_value: float, bars_held: int) -> Signal:
        if rsi_value >= self.rsi_exit:
            return Signal(
                action=Action.SELL,
                symbol=symbol,
                reasoning=f"RSI {rsi_value:.1f} at/above exit {self.rsi_exit}",
                confidence=1.0,
            )
        if bars_held >= self.max_holding_days:
            return Signal(
                action=Action.SELL,
                symbol=symbol,
                reasoning=f"held {bars_held} bars, hit max {self.max_holding_days}",
                confidence=1.0,
            )
        return Signal(
            action=Action.HOLD,
            symbol=symbol,
            reasoning=(
                f"holding: RSI {rsi_value:.1f} below exit {self.rsi_exit}, "
                f"{bars_held}/{self.max_holding_days} bars held"
            ),
        )

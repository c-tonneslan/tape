"""Tests for the backtest harness.

These exercise the mechanics of `backtest()` (fills at next-bar open, stop
handling, multi-symbol shared cash, the end-of-data force close, halt
gating, etc.) without depending on the mean reversion strategy. Tests
plug in a `ScriptedStrategy` that emits the action you want at a given
bar index, which keeps the failures pointing at the harness rather than
at indicator math.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from tape.backtest import backtest
from tape.risk import RiskManager
from tape.strategies import Action
from tape.strategies.base import Signal


# ---------- helpers --------------------------------------------------------


def bars_from_closes(closes, start: str = "2024-01-02") -> pd.DataFrame:
    """OHLCV with closes you provide and ohlc collapsed to close. Daily
    business-day index so resample("ME") in `monthly_equity` has something
    sensible to look at.
    """
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.full_like(closes, 1_000_000.0),
        },
        index=idx,
    )


def bars_with_gap_low(closes, gap_bar: int, gap_low: float) -> pd.DataFrame:
    """Same as bars_from_closes but the bar at index `gap_bar` has its low
    pushed down to `gap_low` (open/high/close left at the close). That lets
    a test drop the low under a stop without disturbing entry pricing.
    """
    bars = bars_from_closes(closes)
    bars.loc[bars.index[gap_bar], "low"] = gap_low
    return bars


@dataclass
class ScriptedStrategy:
    """Minimal stand-in for a real Strategy that emits actions from a dict.

    Has the three period attributes the harness looks at to compute
    `min_history`. Actions are keyed on the 0-indexed bar number; missing
    keys mean HOLD. BUY signals propose an entry at the current close and a
    stop at `stop_pct_below` * close.
    """

    bb_period: int = 5
    rsi_period: int = 5
    atr_period: int = 5
    actions: dict[int, Action] = field(default_factory=dict)
    stop_pct_below: float = 0.95

    def generate_signal(
        self,
        bars: pd.DataFrame,
        *,
        symbol: str = "",
        bars_in_position: int | None = None,
    ) -> Signal:
        bar_idx = len(bars) - 1
        action = self.actions.get(bar_idx, Action.HOLD)
        if action == Action.BUY and bars_in_position is None:
            close = float(bars["close"].iloc[-1])
            return Signal(
                action=Action.BUY,
                symbol=symbol,
                suggested_entry=close,
                suggested_stop=close * self.stop_pct_below,
                confidence=1.0,
            )
        if action == Action.SELL and bars_in_position is not None:
            return Signal(action=Action.SELL, symbol=symbol, confidence=1.0)
        return Signal(action=Action.HOLD, symbol=symbol)


# ---------- happy path -----------------------------------------------------


def test_buy_then_sell_records_a_round_trip():
    closes = [100.0] * 6 + [100.0, 102.0, 104.0, 106.0]
    bars = bars_from_closes(closes)
    strategy = ScriptedStrategy(actions={6: Action.BUY, 8: Action.SELL})

    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.symbol == "SPY"
    assert trade.exit_reason == "signal"
    # BUY at bar 6 means we fill at bar 7's open. SELL at bar 8 fills at bar 9's open.
    assert trade.entry_price == pytest.approx(bars["open"].iloc[7])
    assert trade.exit_price == pytest.approx(bars["open"].iloc[9])
    assert trade.pnl > 0
    assert result.ending_equity > result.starting_equity


def test_fills_happen_at_next_bar_open_not_signal_close():
    closes = [100.0] * 6 + [100.0, 95.0, 100.0]
    bars = bars_from_closes(closes)
    strategy = ScriptedStrategy(actions={6: Action.BUY}, stop_pct_below=0.90)

    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)

    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(95.0)
    assert result.trades[0].entry_price != pytest.approx(100.0)


# ---------- stops ----------------------------------------------------------


def test_stop_fires_when_intraday_low_breaches_stop_price():
    closes = [100.0] * 6 + [100.0, 100.0, 99.0, 99.0]
    bars = bars_with_gap_low(closes, gap_bar=8, gap_low=90.0)
    strategy = ScriptedStrategy(actions={6: Action.BUY}, stop_pct_below=0.95)

    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(95.0)
    assert trade.pnl < 0


# ---------- end of data ----------------------------------------------------


def test_open_position_at_end_of_data_is_force_closed():
    closes = [100.0] * 6 + [100.0, 102.0]
    bars = bars_from_closes(closes)
    strategy = ScriptedStrategy(actions={6: Action.BUY})

    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.trades[0].exit_price == pytest.approx(bars["close"].iloc[-1])


# ---------- multi-symbol --------------------------------------------------


def test_multi_symbol_shares_cash_across_books():
    closes = [100.0] * 6 + [100.0, 105.0, 105.0]
    bars_a = bars_from_closes(closes)
    bars_b = bars_from_closes(closes)
    # 0.5% risk-per-trade with a 5%-below stop sizes each entry to about
    # 9 shares at $105, or $945 notional. A 25% total cap gives room for
    # both books to fill.
    strategy = ScriptedStrategy(actions={6: Action.BUY}, stop_pct_below=0.95)
    rm = RiskManager(max_risk_per_trade=0.005, max_total_exposure=0.25)

    result = backtest(
        {"AAA": bars_a, "BBB": bars_b},
        strategy=strategy,
        risk_manager=rm,
        starting_equity=10_000.0,
    )

    assert len(result.trades) == 2
    symbols = {t.symbol for t in result.trades}
    assert symbols == {"AAA", "BBB"}


def test_multi_symbol_respects_total_exposure_cap():
    # Same per-trade sizing, but a 15% cap that only leaves room for one
    # of the two ~10%-of-equity positions to fill on a given bar.
    closes = [100.0] * 6 + [100.0, 100.0, 100.0]
    bars_a = bars_from_closes(closes)
    bars_b = bars_from_closes(closes)
    strategy = ScriptedStrategy(actions={6: Action.BUY}, stop_pct_below=0.95)
    rm = RiskManager(max_risk_per_trade=0.005, max_total_exposure=0.15)

    result = backtest(
        {"AAA": bars_a, "BBB": bars_b},
        strategy=strategy,
        risk_manager=rm,
        starting_equity=10_000.0,
    )

    open_symbols = {t.symbol for t in result.trades}
    # AAA is processed first alphabetically and uses up the budget; BBB
    # then fails the exposure check and never opens.
    assert "AAA" in open_symbols
    assert "BBB" not in open_symbols


# ---------- halts ----------------------------------------------------------


def test_no_new_entry_after_drawdown_halt():
    # Force a halt before the entry bar by tripping the RiskManager
    # directly; the harness should then refuse the BUY.
    closes = [100.0] * 6 + [100.0, 102.0]
    bars = bars_from_closes(closes)
    strategy = ScriptedStrategy(actions={6: Action.BUY})
    rm = RiskManager()
    rm.halt("test halt")

    result = backtest(bars, symbol="SPY", strategy=strategy, risk_manager=rm,
                      starting_equity=10_000.0)

    assert result.trades == []
    assert result.ending_equity == pytest.approx(10_000.0)


# ---------- empty / degenerate ---------------------------------------------


def test_dataframe_input_requires_symbol():
    bars = bars_from_closes([100.0] * 10)
    with pytest.raises(ValueError):
        backtest(bars, strategy=ScriptedStrategy(), starting_equity=10_000.0)


def test_too_little_history_produces_no_trades():
    bars = bars_from_closes([100.0] * 4)
    strategy = ScriptedStrategy(actions={3: Action.BUY})
    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)
    assert result.trades == []


# ---------- result shape ---------------------------------------------------


def test_summary_includes_the_headline_metrics():
    closes = [100.0] * 6 + [100.0, 102.0, 104.0, 106.0]
    bars = bars_from_closes(closes)
    strategy = ScriptedStrategy(actions={6: Action.BUY, 8: Action.SELL})

    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)
    text = result.summary()
    for label in ("start equity", "end equity", "P/L", "trades", "win rate",
                  "max drawdown", "CAGR", "Sharpe"):
        assert label in text


def test_sharpe_is_zero_when_equity_curve_is_flat():
    bars = bars_from_closes([100.0] * 30)
    strategy = ScriptedStrategy()  # always HOLD
    result = backtest(bars, symbol="SPY", strategy=strategy, starting_equity=10_000.0)
    assert result.sharpe() == 0.0
    assert result.cagr() == 0.0

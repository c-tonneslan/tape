"""Tests for the RiskManager class.

Position sizing math has its own file; this one is about the gate logic and
the halt conditions.
"""

import pytest

from tape.risk import AccountState, Position, RiskManager
from tape.strategies import Action, Signal


def buy_signal(
    *, symbol: str = "SPY", entry: float = 100.0, stop: float = 95.0
) -> Signal:
    return Signal(
        action=Action.BUY,
        symbol=symbol,
        suggested_entry=entry,
        suggested_stop=stop,
        reasoning="test",
        confidence=1.0,
    )


def empty_account(equity: float = 100_000.0, cash: float | None = None) -> AccountState:
    return AccountState(equity=equity, cash=cash if cash is not None else equity)


# ---------- initial state --------------------------------------------------


def test_fresh_manager_is_not_halted():
    rm = RiskManager()
    assert not rm.is_halted()
    assert rm.peak_equity is None


# ---------- halt conditions ------------------------------------------------


def test_daily_loss_limit_trips_halt():
    rm = RiskManager(daily_loss_limit=0.03)
    rm.start_of_day(equity=100_000)
    rm.update_equity(equity=96_900)  # 3.1% drawdown on the day

    assert rm.is_halted()
    assert "daily loss" in rm.halt_reason


def test_intraday_drawdown_just_under_limit_doesnt_halt():
    rm = RiskManager(daily_loss_limit=0.03)
    rm.start_of_day(equity=100_000)
    rm.update_equity(equity=97_500)  # 2.5%, under the 3% limit

    assert not rm.is_halted()


def test_max_drawdown_from_peak_trips_halt():
    rm = RiskManager(max_drawdown=0.15)
    rm.start_of_day(equity=100_000)
    rm.update_equity(equity=120_000)  # new peak
    rm.update_equity(equity=101_000)  # ~15.8% off peak

    assert rm.is_halted()
    assert "drawdown" in rm.halt_reason


def test_peak_equity_tracks_high_water_mark():
    rm = RiskManager()
    rm.start_of_day(equity=100_000)
    rm.update_equity(equity=110_000)
    rm.update_equity(equity=105_000)  # drawdown but new peak stays

    assert rm.peak_equity == 110_000


def test_manual_halt_sets_reason():
    rm = RiskManager()
    rm.halt("kill switch tripped from CLI")
    assert rm.is_halted()
    assert rm.halt_reason == "kill switch tripped from CLI"


def test_resume_clears_a_halt():
    rm = RiskManager()
    rm.halt("manual")
    rm.resume()
    assert not rm.is_halted()


# ---------- evaluate: pre-trade gate ---------------------------------------


def test_evaluate_approves_a_normal_buy():
    rm = RiskManager()
    rm.start_of_day(equity=100_000)
    decision = rm.evaluate(buy_signal(), empty_account())

    assert decision.approved
    assert decision.shares == 200  # 1% of $100k = $1k risk / $5 stop dist = 200
    assert "approved" in decision.reason


def test_evaluate_rejects_when_halted():
    rm = RiskManager()
    rm.start_of_day(equity=100_000)
    rm.halt("manual")
    decision = rm.evaluate(buy_signal(), empty_account())

    assert not decision.approved
    assert decision.shares == 0
    assert "halted" in decision.reason


def test_evaluate_rejects_non_buy_signals():
    rm = RiskManager()
    sell = Signal(action=Action.SELL, symbol="SPY", reasoning="exit")
    hold = Signal(action=Action.HOLD, symbol="SPY", reasoning="no edge")

    assert not rm.evaluate(sell, empty_account()).approved
    assert not rm.evaluate(hold, empty_account()).approved


def test_evaluate_rejects_buy_without_entry_or_stop():
    rm = RiskManager()
    signal = Signal(action=Action.BUY, symbol="SPY", suggested_entry=100.0)
    decision = rm.evaluate(signal, empty_account())

    assert not decision.approved
    assert "missing" in decision.reason


def test_evaluate_rejects_when_position_size_rounds_to_zero():
    rm = RiskManager(max_risk_per_trade=0.0001)  # $10 risk budget on $100k
    # $50 stop distance → can't afford a single share.
    decision = rm.evaluate(
        buy_signal(entry=200.0, stop=150.0),
        empty_account(equity=100_000),
    )
    assert not decision.approved
    assert "0 shares" in decision.reason


def test_evaluate_rejects_when_total_exposure_would_exceed_cap():
    rm = RiskManager(max_total_exposure=0.20)
    # $100k account, 20% cap = $20k. Already $19k in another position.
    existing = Position(symbol="QQQ", qty=50, entry_price=380.0, market_value=19_000.0)
    account = AccountState(
        equity=100_000.0, cash=100_000.0, positions=(existing,)
    )
    # 200 shares @ $100 = $20k notional → pushes us over.
    decision = rm.evaluate(buy_signal(), account)

    assert not decision.approved
    assert "exposure" in decision.reason


def test_evaluate_rejects_when_cash_is_short():
    rm = RiskManager()
    # Plenty of equity (positions count too) but cash is locked up.
    account = AccountState(equity=100_000.0, cash=500.0)
    decision = rm.evaluate(buy_signal(), account)

    assert not decision.approved
    assert "cash" in decision.reason


def test_evaluate_rejects_when_equity_is_zero_or_negative():
    rm = RiskManager()
    decision = rm.evaluate(buy_signal(), AccountState(equity=0.0, cash=0.0))
    assert not decision.approved


# ---------- end-to-end -----------------------------------------------------


def test_halt_blocks_subsequent_evaluate_calls():
    rm = RiskManager(daily_loss_limit=0.03)
    rm.start_of_day(equity=100_000)
    # Approve once.
    first = rm.evaluate(buy_signal(), empty_account())
    assert first.approved

    # Equity craters; next evaluate should refuse.
    rm.update_equity(equity=95_000)
    second = rm.evaluate(buy_signal(), empty_account(equity=95_000))
    assert not second.approved
    assert "halted" in second.reason


@pytest.mark.parametrize(
    "equity,expected_halt",
    [
        (97_500, False),  # 2.5% loss, under limit
        (97_000, True),   # exactly 3% loss, hits limit
        (90_000, True),   # 10% loss, well past
    ],
)
def test_daily_loss_threshold_is_inclusive(equity, expected_halt):
    rm = RiskManager(daily_loss_limit=0.03)
    rm.start_of_day(equity=100_000)
    rm.update_equity(equity=equity)
    assert rm.is_halted() is expected_halt

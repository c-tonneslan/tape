"""Risk management.

The `RiskManager` is the only thing allowed to greenlight an order. Strategy
modules emit signals; the risk manager decides whether the trade fits inside
the per-trade risk cap, the total exposure cap, and isn't blocked by a halt
condition.

The halt conditions are the layers from the bot's constitution:
- Daily loss limit (default 3% of start-of-day equity) → halt for the day.
- Max drawdown (default 15% from peak equity) → halt indefinitely.
- Kill switch → halt indefinitely until explicitly cleared.

The class is intentionally I/O-free. The caller feeds it equity updates and
account snapshots; it returns decisions. Wiring it to the broker is the
execution module's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..strategies.base import Action, Signal


def calculate_position_size(
    account_value: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.01,
) -> int:
    """Number of shares to buy so that the loss on a stop-out equals
    `risk_pct` of `account_value`.

    The math is:

        max_dollar_loss = account_value * risk_pct
        per_share_loss  = |entry_price - stop_loss_price|
        shares          = floor(max_dollar_loss / per_share_loss)

    Returns 0 (a refusal) when the trade has no defined risk, when the
    inputs are non-positive, or when one share already costs more than the
    permitted dollar loss. The caller should treat 0 as "skip this trade,"
    not as "fill any amount."

    The 1% rule (Van Tharp, Mark Douglas) bounds the *loss* at 1% of the
    account, not the position size. A 1%-loss trade with a tight 0.5% stop
    is a position roughly 20% of the account in notional terms. That's
    expected and correct — it's the loss that's capped.
    """
    if account_value <= 0 or entry_price <= 0 or risk_pct <= 0:
        return 0

    per_share_risk = abs(entry_price - stop_loss_price)
    if per_share_risk == 0:
        return 0

    max_dollar_loss = account_value * risk_pct
    return int(max_dollar_loss // per_share_risk)


@dataclass(frozen=True)
class Position:
    """A single open position. Market value is qty * current price."""

    symbol: str
    qty: int
    entry_price: float
    market_value: float


@dataclass(frozen=True)
class AccountState:
    """Snapshot of the account at decision time."""

    equity: float
    cash: float
    positions: tuple[Position, ...] = ()


@dataclass(frozen=True)
class RiskDecision:
    """Result of `RiskManager.evaluate`."""

    approved: bool
    shares: int
    reason: str


class RiskManager:
    """Gates orders and tracks halt conditions across the trading day.

    Lifecycle in a typical session:
        rm = RiskManager(...)
        rm.start_of_day(equity=...)
        for each new bar:
            rm.update_equity(equity=...)           # may halt
            decision = rm.evaluate(signal, account)
            if decision.approved: execute(decision.shares)
    """

    def __init__(
        self,
        *,
        max_risk_per_trade: float = 0.01,
        max_total_exposure: float = 0.20,
        daily_loss_limit: float = 0.03,
        max_drawdown: float = 0.15,
    ) -> None:
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_exposure = max_total_exposure
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown

        self._peak_equity: float | None = None
        self._daily_start_equity: float | None = None
        self._halted: bool = False
        self._halt_reason: str = ""

    # ----- state introspection -------------------------------------------

    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def peak_equity(self) -> float | None:
        return self._peak_equity

    # ----- state transitions --------------------------------------------

    def start_of_day(self, *, equity: float) -> None:
        """Reset the day's loss baseline. Call once at session start.

        Doesn't clear an existing halt — drawdown and kill-switch halts
        persist across days by design. Use `resume()` to clear manually
        after review.
        """
        self._daily_start_equity = equity
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

    def update_equity(self, *, equity: float) -> None:
        """Record a new equity reading. Trips a halt if a limit is breached."""
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        if self._halted:
            return

        if self._peak_equity is not None and self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.max_drawdown:
                self.halt(
                    f"max drawdown breached: {drawdown:.2%} from peak "
                    f"{self._peak_equity:.2f} (limit {self.max_drawdown:.0%})"
                )
                return

        if self._daily_start_equity is not None and self._daily_start_equity > 0:
            day_loss = (self._daily_start_equity - equity) / self._daily_start_equity
            if day_loss >= self.daily_loss_limit:
                self.halt(
                    f"daily loss limit breached: down {day_loss:.2%} from "
                    f"open {self._daily_start_equity:.2f} (limit "
                    f"{self.daily_loss_limit:.0%})"
                )

    def halt(self, reason: str) -> None:
        """Trip the kill switch. The next `evaluate` will refuse any order."""
        self._halted = True
        self._halt_reason = reason

    def resume(self) -> None:
        """Clear a halt after manual review. Doesn't reset the daily baseline."""
        self._halted = False
        self._halt_reason = ""

    # ----- the main gate -------------------------------------------------

    def evaluate(self, signal: Signal, account: AccountState) -> RiskDecision:
        """Decide whether to act on a signal, and at what size."""
        if self._halted:
            return RiskDecision(False, 0, f"halted: {self._halt_reason}")

        if signal.action != Action.BUY:
            # SELL/HOLD don't go through sizing here. The execution side
            # closes a known position size; HOLD does nothing.
            return RiskDecision(False, 0, f"no order: signal is {signal.action}")

        if signal.suggested_entry is None or signal.suggested_stop is None:
            return RiskDecision(False, 0, "BUY signal missing entry or stop price")

        if account.equity <= 0:
            return RiskDecision(False, 0, f"account equity is {account.equity}")

        shares = calculate_position_size(
            account_value=account.equity,
            entry_price=signal.suggested_entry,
            stop_loss_price=signal.suggested_stop,
            risk_pct=self.max_risk_per_trade,
        )
        if shares <= 0:
            return RiskDecision(False, 0, "position size rounds to 0 shares")

        new_notional = shares * signal.suggested_entry
        current_exposure = sum(p.market_value for p in account.positions)
        max_dollar_exposure = account.equity * self.max_total_exposure
        if current_exposure + new_notional > max_dollar_exposure:
            return RiskDecision(
                False,
                0,
                f"would exceed max exposure: current {current_exposure:.2f} + "
                f"new {new_notional:.2f} > cap {max_dollar_exposure:.2f}",
            )

        if new_notional > account.cash:
            return RiskDecision(
                False,
                0,
                f"insufficient cash: need {new_notional:.2f}, have {account.cash:.2f}",
            )

        return RiskDecision(
            True,
            shares,
            f"approved: {shares} shares ({new_notional:.2f} notional)",
        )

"""Risk management.

`calculate_position_size` is implemented and tested today. The full
RiskManager (pre-trade checks, daily loss limit, drawdown circuit breaker,
kill switch) lands in a follow-up.
"""

from __future__ import annotations


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

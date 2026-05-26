"""Backtest harness.

Walks a daily OHLCV DataFrame (or a dict of them, one per symbol) bar by
bar, asks the strategy for a signal, runs it through the RiskManager, and
tracks cash + positions + an equity curve. Fills happen at the next bar's
open — no peeking at the close that generated the signal.

Single-symbol or multi-symbol with shared cash. No slippage, no
commissions, no shorts. Good enough to answer "would this have made money"
without pretending to be Backtrader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# 252 NYSE trading days per year, the standard yardstick for daily-bar
# annualization. Used to convert per-bar return stats into yearly numbers.
_TRADING_DAYS_PER_YEAR = 252

from .risk import AccountState, Position, RiskManager
from .strategies import Action
from .strategies.base import Signal
from .strategies.mean_reversion import MeanReversionStrategy


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    qty: int
    pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    starting_equity: float
    ending_equity: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        return self.ending_equity - self.starting_equity

    @property
    def return_pct(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return self.pnl / self.starting_equity

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx = pd.DatetimeIndex([ts for ts, _ in self.equity_curve])
        values = [eq for _, eq in self.equity_curve]
        return pd.Series(values, index=idx, name="equity")

    def monthly_equity(self) -> pd.Series:
        """End-of-month equity readings. Useful for seeing the path."""
        s = self.equity_series()
        if s.empty:
            return s
        return s.resample("ME").last()

    def cagr(self) -> float:
        """Compound annual growth rate, derived from the equity curve span.

        Returns 0.0 if the curve is empty, the span is under a day, or the
        starting equity isn't positive. The span is calendar days between
        first and last equity reading; that beats counting bars when the
        feed has gaps (weekends, holidays).
        """
        if not self.equity_curve or self.starting_equity <= 0:
            return 0.0
        start_ts, _ = self.equity_curve[0]
        end_ts, _ = self.equity_curve[-1]
        days = (end_ts - start_ts).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        ratio = self.ending_equity / self.starting_equity
        if ratio <= 0:
            return -1.0
        return ratio ** (1.0 / years) - 1.0

    def sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio of the daily equity returns.

        Daily returns are simple pct_change on the equity curve. A flat or
        empty curve returns 0.0 rather than NaN. `risk_free_rate` is the
        annual rate (e.g. 0.04 for 4%); it's converted to a per-day rate by
        dividing by 252 before subtracting from each daily return.
        """
        s = self.equity_series()
        if len(s) < 2:
            return 0.0
        returns = s.pct_change().dropna()
        if returns.empty:
            return 0.0
        per_day_rf = risk_free_rate / _TRADING_DAYS_PER_YEAR
        excess = returns - per_day_rf
        std = float(excess.std(ddof=1))
        if std == 0.0 or math.isnan(std):
            return 0.0
        return float(excess.mean()) / std * math.sqrt(_TRADING_DAYS_PER_YEAR)

    def summary(self) -> str:
        lines = [
            f"  start equity:  ${self.starting_equity:>10,.2f}",
            f"  end equity:    ${self.ending_equity:>10,.2f}",
            f"  P/L:           ${self.pnl:>+10,.2f}  ({self.return_pct:+.2%})",
            f"  CAGR:          {self.cagr():>10.2%}",
            f"  Sharpe:        {self.sharpe():>10.2f}",
            f"  trades:        {self.num_trades:>10d}",
            f"  win rate:      {self.win_rate:>10.1%}",
            f"  max drawdown:  {self.max_drawdown:>10.1%}",
        ]
        return "\n".join(lines)


@dataclass
class _SymbolState:
    qty: int = 0
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None
    stop_price: float = 0.0
    bars_held: int | None = None
    pending_action: Action | None = None
    pending_stop: float | None = None


def backtest(
    bars: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    symbol: str | None = None,
    strategy: MeanReversionStrategy | None = None,
    risk_manager: RiskManager | None = None,
    starting_equity: float = 10_000.0,
) -> BacktestResult:
    """Run a backtest over one symbol (DataFrame) or several (dict)."""
    if isinstance(bars, pd.DataFrame):
        if symbol is None:
            raise ValueError("symbol is required when bars is a DataFrame")
        bars_by_symbol = {symbol: bars}
    else:
        bars_by_symbol = bars

    if strategy is None:
        strategy = MeanReversionStrategy()
    if risk_manager is None:
        risk_manager = RiskManager()

    # Union of all timestamps across symbols, sorted.
    all_index = pd.DatetimeIndex(sorted(set().union(*(b.index for b in bars_by_symbol.values()))))

    state: dict[str, _SymbolState] = {sym: _SymbolState() for sym in bars_by_symbol}
    cash = starting_equity
    result = BacktestResult(starting_equity=starting_equity, ending_equity=starting_equity)
    risk_manager.start_of_day(equity=starting_equity)

    min_history = max(strategy.bb_period, strategy.rsi_period, strategy.atr_period) + 1

    for ts in all_index:
        # 1) Execute pending fills + check stops, per symbol.
        for sym, bars_df in bars_by_symbol.items():
            if ts not in bars_df.index:
                continue
            row = bars_df.loc[ts]
            bar_open = float(row["open"])
            bar_low = float(row["low"])
            st = state[sym]

            if st.pending_action == Action.BUY and st.qty == 0:
                equity_now = cash + _open_positions_value(state, bars_by_symbol, ts)
                shares = _size_at_fill(
                    risk_manager=risk_manager,
                    cash=cash,
                    equity=equity_now,
                    open_positions=_position_snapshot(state, bars_by_symbol, ts, skip=sym),
                    symbol=sym,
                    fill_price=bar_open,
                    stop=st.pending_stop or 0.0,
                )
                if shares > 0 and shares * bar_open <= cash:
                    st.qty = shares
                    st.entry_price = bar_open
                    st.entry_time = ts
                    st.stop_price = st.pending_stop or 0.0
                    st.bars_held = 0
                    cash -= shares * bar_open
                st.pending_action = None
                st.pending_stop = None
            elif st.pending_action == Action.SELL and st.qty > 0:
                pnl = (bar_open - st.entry_price) * st.qty
                cash += st.qty * bar_open
                result.trades.append(
                    Trade(
                        symbol=sym,
                        entry_time=st.entry_time,  # type: ignore[arg-type]
                        entry_price=st.entry_price,
                        exit_time=ts,
                        exit_price=bar_open,
                        qty=st.qty,
                        pnl=pnl,
                        exit_reason="signal",
                    )
                )
                st.qty = 0
                st.bars_held = None
                st.pending_action = None

            if st.qty > 0 and bar_low <= st.stop_price:
                pnl = (st.stop_price - st.entry_price) * st.qty
                cash += st.qty * st.stop_price
                result.trades.append(
                    Trade(
                        symbol=sym,
                        entry_time=st.entry_time,  # type: ignore[arg-type]
                        entry_price=st.entry_price,
                        exit_time=ts,
                        exit_price=st.stop_price,
                        qty=st.qty,
                        pnl=pnl,
                        exit_reason="stop",
                    )
                )
                st.qty = 0
                st.bars_held = None

        # 2) Mark to market across all open positions.
        equity = cash + _open_positions_value(state, bars_by_symbol, ts)
        risk_manager.update_equity(equity=equity)
        result.equity_curve.append((ts, equity))

        # 3) Per-symbol signal generation.
        for sym, bars_df in bars_by_symbol.items():
            if ts not in bars_df.index:
                continue
            # Window of bars up to and including today for this symbol.
            window = bars_df.loc[:ts]
            st = state[sym]

            if len(window) < min_history:
                if st.qty > 0 and st.bars_held is not None:
                    st.bars_held += 1
                continue

            signal = strategy.generate_signal(
                window, symbol=sym, bars_in_position=st.bars_held
            )

            if (
                signal.action == Action.BUY
                and st.qty == 0
                and not risk_manager.is_halted()
            ):
                st.pending_action = Action.BUY
                st.pending_stop = signal.suggested_stop
            elif signal.action == Action.SELL and st.qty > 0:
                st.pending_action = Action.SELL

            if st.qty > 0 and st.bars_held is not None:
                st.bars_held += 1

    # Close any open positions at the final available close per symbol.
    for sym, st in state.items():
        if st.qty > 0 and st.entry_time is not None:
            bars_df = bars_by_symbol[sym]
            final_ts = bars_df.index[-1]
            final_price = float(bars_df["close"].iloc[-1])
            pnl = (final_price - st.entry_price) * st.qty
            cash += st.qty * final_price
            result.trades.append(
                Trade(
                    symbol=sym,
                    entry_time=st.entry_time,
                    entry_price=st.entry_price,
                    exit_time=final_ts,
                    exit_price=final_price,
                    qty=st.qty,
                    pnl=pnl,
                    exit_reason="end_of_data",
                )
            )

    result.ending_equity = cash
    return result


def _open_positions_value(
    state: dict[str, _SymbolState],
    bars_by_symbol: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
) -> float:
    total = 0.0
    for sym, st in state.items():
        if st.qty <= 0:
            continue
        bars_df = bars_by_symbol[sym]
        # Use today's close if we have it, else the last known close.
        if ts in bars_df.index:
            price = float(bars_df.loc[ts, "close"])
        else:
            prior = bars_df.loc[:ts]
            if prior.empty:
                continue
            price = float(prior["close"].iloc[-1])
        total += st.qty * price
    return total


def _position_snapshot(
    state: dict[str, _SymbolState],
    bars_by_symbol: dict[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    skip: str | None = None,
) -> tuple[Position, ...]:
    out: list[Position] = []
    for sym, st in state.items():
        if sym == skip or st.qty <= 0:
            continue
        bars_df = bars_by_symbol[sym]
        if ts in bars_df.index:
            price = float(bars_df.loc[ts, "close"])
        else:
            prior = bars_df.loc[:ts]
            if prior.empty:
                continue
            price = float(prior["close"].iloc[-1])
        out.append(
            Position(symbol=sym, qty=st.qty, entry_price=st.entry_price, market_value=st.qty * price)
        )
    return tuple(out)


def _size_at_fill(
    *,
    risk_manager: RiskManager,
    cash: float,
    equity: float,
    open_positions: tuple[Position, ...],
    symbol: str,
    fill_price: float,
    stop: float,
) -> int:
    signal = Signal(
        action=Action.BUY,
        symbol=symbol,
        suggested_entry=fill_price,
        suggested_stop=stop,
    )
    decision = risk_manager.evaluate(
        signal,
        AccountState(equity=equity, cash=cash, positions=open_positions),
    )
    return decision.shares if decision.approved else 0

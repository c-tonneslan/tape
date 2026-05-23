"""Strategy contract.

A strategy is a pure function from market data to a Signal. No API calls, no
order placement, no side effects. The risk manager and execution engine
decide whether and how to act on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    """A strategy's output."""

    action: Action
    symbol: str
    # When action is BUY or SELL, these should be populated. HOLD ignores them.
    suggested_entry: float | None = None
    suggested_stop: float | None = None
    reasoning: str = ""
    # 0.0 to 1.0. Strategies that don't model confidence can leave it at 1.0
    # for BUY/SELL and 0.0 for HOLD.
    confidence: float = 0.0


class Strategy(ABC):
    """Base class for all strategies. Subclasses implement `generate_signal`."""

    @abstractmethod
    def generate_signal(self, market_data: Any) -> Signal:
        """Look at the market data and return a Signal.

        The shape of `market_data` is intentionally generic for now; each
        strategy declares what it expects (a pandas DataFrame of bars, a
        single Bar, etc.). We'll narrow this once we have more than one
        concrete strategy.
        """
        raise NotImplementedError

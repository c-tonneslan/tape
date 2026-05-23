"""Central settings for the bot.

Every numeric trading parameter lives here, never hardcoded elsewhere. That
way parameter sweeps and config changes don't require touching the strategy,
risk, or execution code.

Secrets (API keys) come from environment variables / .env, never from this
file. The bot refuses to run if the Alpaca base URL isn't the paper URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class NotPaperTradingError(RuntimeError):
    """Raised when the configured Alpaca URL isn't the paper URL.

    The bot is paper-only until the CLAUDE.md constraint is lifted, so any
    non-paper URL is treated as a fatal config error rather than a silent
    handoff to live trading.
    """


@dataclass(frozen=True)
class Settings:
    # --- credentials ---
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    anthropic_api_key: str | None = None

    # --- risk parameters ---
    max_risk_per_trade: float = 0.01            # 1% of account
    max_total_exposure: float = 0.20            # 20% of account across positions
    daily_loss_limit: float = 0.03              # 3% of start-of-day equity → halt
    max_drawdown: float = 0.15                  # 15% from peak → halt indefinitely

    # --- stop / sizing ---
    atr_lookback: int = 14
    stop_atr_multiplier: float = 2.0

    # --- universe ---
    watchlist: tuple[str, ...] = ("SPY", "QQQ", "IWM")

    # --- mean reversion strategy parameters ---
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_exit: float = 50.0
    bbands_period: int = 20
    bbands_stddev: float = 1.5
    max_holding_days: int = 5

    # --- data ---
    historical_cache_dir: str = field(default_factory=lambda: ".data_cache")


def load_settings() -> Settings:
    """Read settings from environment / .env, validate, and return Settings."""
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    base = os.getenv("ALPACA_BASE_URL", PAPER_BASE_URL).strip()

    if not api_key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set. Copy .env.example "
            "to .env and fill in your paper-account keys from "
            "https://app.alpaca.markets/paper/dashboard/overview"
        )
    if base != PAPER_BASE_URL:
        raise NotPaperTradingError(
            f"Refusing to start: ALPACA_BASE_URL is {base!r}, not the paper URL "
            f"({PAPER_BASE_URL!r}). The bot is paper-only until CLAUDE.md says "
            "otherwise."
        )

    return Settings(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret,
        alpaca_base_url=base,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
    )

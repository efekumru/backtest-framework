"""
Data handler — OHLCV data feed for the backtest engine.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataHandler:
    """Bar-by-bar data feed. Prevents lookahead by only exposing past data."""

    def __init__(self, ticker: str, data: pd.DataFrame):
        self.ticker = ticker
        self.data = data.copy()
        self.index = 0
        self.total_bars = len(data)

    @classmethod
    def from_csv(cls, ticker: str, filepath: str) -> "DataHandler":
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {ticker} from CSV: {len(df)} bars")
        return cls(ticker, df)

    @classmethod
    def from_yfinance(cls, ticker: str, period: str = "5y") -> "DataHandler":
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        logger.info(f"Downloaded {ticker}: {len(df)} bars")
        return cls(ticker, df)

    def has_next(self) -> bool:
        return self.index < self.total_bars

    def get_next_bar(self) -> Optional[pd.Series]:
        if not self.has_next():
            return None
        bar = self.data.iloc[self.index]
        self.index += 1
        return bar

    def get_current_price(self) -> float:
        if self.index == 0:
            return self.data.iloc[0]["Close"]
        return self.data.iloc[self.index - 1]["Close"]

    def get_current_bar(self) -> Optional[pd.Series]:
        if self.index == 0:
            return None
        return self.data.iloc[self.index - 1]

    def get_history(self, lookback: int) -> pd.DataFrame:
        """Return past N bars only — no future data leakage."""
        end = self.index
        start = max(0, end - lookback)
        return self.data.iloc[start:end].copy()

    def get_all_history(self) -> pd.DataFrame:
        """Return all bars seen so far."""
        return self.data.iloc[:self.index].copy()

    def reset(self) -> None:
        self.index = 0

    @property
    def progress(self) -> float:
        return self.index / self.total_bars if self.total_bars > 0 else 0

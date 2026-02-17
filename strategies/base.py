"""
Abstract base class for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

from engine.data_handler import DataHandler
from engine.events import SignalEvent


class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    @abstractmethod
    def calculate_signals(self, data_handler: DataHandler) -> Optional[SignalEvent]:
        """Analyze current market data and optionally emit a signal.

        Args:
            data_handler: Provides access to historical bars (no future data).

        Returns:
            SignalEvent if a trading opportunity is detected, None otherwise.
        """
        ...

    def reset(self) -> None:
        """Reset strategy state for a new backtest run."""
        pass

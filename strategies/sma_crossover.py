"""
Simple Moving Average crossover strategy.
"""

from typing import Optional

from engine.data_handler import DataHandler
from engine.events import Direction, SignalEvent
from strategies.base import BaseStrategy


class SMACrossover(BaseStrategy):
    """Go long when fast MA crosses above slow MA, exit on cross below."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prev_position: Direction = Direction.EXIT

    def calculate_signals(self, data_handler: DataHandler) -> Optional[SignalEvent]:
        history = data_handler.get_history(self.slow_period + 1)
        if len(history) < self.slow_period:
            return None

        close = history["Close"]
        fast_ma = close.rolling(self.fast_period).mean().iloc[-1]
        slow_ma = close.rolling(self.slow_period).mean().iloc[-1]

        if fast_ma > slow_ma and self.prev_position != Direction.LONG:
            self.prev_position = Direction.LONG
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.LONG,
                strength=1.0,
                timestamp=history.index[-1],
            )
        elif fast_ma < slow_ma and self.prev_position == Direction.LONG:
            self.prev_position = Direction.EXIT
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.EXIT,
                strength=1.0,
                timestamp=history.index[-1],
            )

        return None

    def reset(self) -> None:
        self.prev_position = Direction.EXIT

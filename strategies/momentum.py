"""
Momentum / trend-following strategy.
"""

from typing import Optional

from engine.data_handler import DataHandler
from engine.events import Direction, SignalEvent
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Enter long when N-day return exceeds threshold, exit when it reverses."""

    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.prev_position: Direction = Direction.EXIT

    def calculate_signals(self, data_handler: DataHandler) -> Optional[SignalEvent]:
        history = data_handler.get_history(self.lookback + 1)
        if len(history) < self.lookback + 1:
            return None

        returns = (
            history["Close"].iloc[-1] / history["Close"].iloc[0] - 1
        )

        if returns > self.threshold and self.prev_position != Direction.LONG:
            self.prev_position = Direction.LONG
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.LONG,
                strength=min(returns / self.threshold, 2.0),
                timestamp=history.index[-1],
            )
        elif returns < -self.threshold and self.prev_position == Direction.LONG:
            self.prev_position = Direction.EXIT
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.EXIT,
                timestamp=history.index[-1],
            )

        return None

    def reset(self) -> None:
        self.prev_position = Direction.EXIT

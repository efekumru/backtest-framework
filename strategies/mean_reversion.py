"""
Mean reversion strategy using z-score.
"""

from typing import Optional

from engine.data_handler import DataHandler
from engine.events import Direction, SignalEvent
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Enter long when z-score drops below entry threshold, exit at mean."""

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = -1.5,
        exit_z: float = 0.0,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.in_position = False

    def calculate_signals(self, data_handler: DataHandler) -> Optional[SignalEvent]:
        history = data_handler.get_history(self.lookback + 1)
        if len(history) < self.lookback:
            return None

        close = history["Close"]
        mean = close.rolling(self.lookback).mean().iloc[-1]
        std = close.rolling(self.lookback).std().iloc[-1]

        if std == 0:
            return None

        z_score = (close.iloc[-1] - mean) / std

        if z_score <= self.entry_z and not self.in_position:
            self.in_position = True
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.LONG,
                strength=abs(z_score),
                timestamp=history.index[-1],
            )
        elif z_score >= self.exit_z and self.in_position:
            self.in_position = False
            return SignalEvent(
                ticker=data_handler.ticker,
                direction=Direction.EXIT,
                timestamp=history.index[-1],
            )

        return None

    def reset(self) -> None:
        self.in_position = False

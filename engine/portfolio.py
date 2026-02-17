"""
Portfolio management — tracks positions, cash, equity, and trades.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from engine.events import Direction, FillEvent


@dataclass
class Position:
    """Tracks a single open position."""
    ticker: str
    quantity: int = 0
    avg_price: float = 0.0
    entry_date: str = ""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price

    def update_unrealized(self, current_price: float) -> None:
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity


class Portfolio:
    """Manages positions, cash, and records equity over time."""

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[dict] = []
        self.trades: list[dict] = []
        self.total_commission: float = 0.0

    @property
    def total_value(self) -> float:
        positions_value = sum(
            p.quantity * p.avg_price for p in self.positions.values()
        )
        return self.cash + positions_value

    @property
    def total_return(self) -> float:
        return (self.total_value / self.initial_capital) - 1

    def update_market_value(self, ticker: str, price: float) -> None:
        if ticker in self.positions:
            self.positions[ticker].update_unrealized(price)

    def process_fill(self, fill: FillEvent) -> None:
        """Update portfolio based on a fill event."""
        ticker = fill.ticker
        self.total_commission += fill.total_cost

        if fill.direction == Direction.LONG:
            self._open_position(fill)
        elif fill.direction == Direction.EXIT:
            self._close_position(fill)
        elif fill.direction == Direction.SHORT:
            self._close_position(fill)

    def _open_position(self, fill: FillEvent) -> None:
        ticker = fill.ticker
        cost = fill.fill_price * fill.quantity + fill.total_cost

        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                entry_date=str(fill.timestamp),
            )

        pos = self.positions[ticker]
        total_qty = pos.quantity + fill.quantity
        if total_qty > 0:
            pos.avg_price = (
                (pos.avg_price * pos.quantity)
                + (fill.fill_price * fill.quantity)
            ) / total_qty
        pos.quantity = total_qty
        self.cash -= cost

    def _close_position(self, fill: FillEvent) -> None:
        ticker = fill.ticker
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        pnl = (fill.fill_price - pos.avg_price) * pos.quantity
        ret_pct = (fill.fill_price / pos.avg_price - 1) * 100 if pos.avg_price > 0 else 0

        self.trades.append({
            "ticker": ticker,
            "direction": "LONG",
            "entry_price": round(pos.avg_price, 2),
            "exit_price": round(fill.fill_price, 2),
            "quantity": pos.quantity,
            "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
            "entry_date": pos.entry_date,
            "exit_date": str(fill.timestamp),
            "commission": round(fill.total_cost, 2),
        })

        self.cash += fill.fill_price * pos.quantity - fill.total_cost
        del self.positions[ticker]

    def record_equity(self, timestamp, current_prices: dict[str, float]) -> None:
        positions_value = sum(
            p.quantity * current_prices.get(p.ticker, p.avg_price)
            for p in self.positions.values()
        )
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": self.cash + positions_value,
            "cash": self.cash,
            "positions_value": positions_value,
            "n_positions": len(self.positions),
        })

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def summary(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_value": round(self.total_value, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "total_trades": len(self.trades),
            "total_commission": round(self.total_commission, 2),
            "cash_remaining": round(self.cash, 2),
            "open_positions": len(self.positions),
        }

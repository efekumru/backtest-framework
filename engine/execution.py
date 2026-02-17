"""
Execution handler — simulates order fills with slippage and commission.
"""

import logging
from dataclasses import dataclass

from engine.events import Direction, FillEvent, OrderEvent

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Transaction cost configuration."""
    commission_per_trade: float = 10.0
    slippage_pct: float = 0.001
    market_impact_pct: float = 0.0005

    def calculate_slippage(self, price: float, quantity: int, direction: Direction) -> float:
        base_slippage = price * self.slippage_pct
        impact = price * self.market_impact_pct * (quantity / 1000)
        total = base_slippage + impact
        return total if direction == Direction.LONG else -total


class ExecutionHandler:
    """Simulates order execution with realistic cost modeling."""

    def __init__(self, cost_model: CostModel = None):
        self.cost_model = cost_model or CostModel()
        self.fill_count = 0

    def execute(self, order: OrderEvent, current_price: float) -> FillEvent:
        """Execute an order and return a fill event."""
        slippage = self.cost_model.calculate_slippage(
            current_price, order.quantity, order.direction
        )

        if order.direction == Direction.LONG:
            fill_price = current_price + abs(slippage)
        else:
            fill_price = current_price - abs(slippage)

        slippage_cost = abs(slippage) * order.quantity
        self.fill_count += 1

        fill = FillEvent(
            ticker=order.ticker,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=round(fill_price, 4),
            commission=self.cost_model.commission_per_trade,
            slippage=round(slippage_cost, 4),
            timestamp=order.timestamp,
        )

        logger.debug(f"Fill #{self.fill_count}: {fill}")
        return fill

    def reset(self) -> None:
        self.fill_count = 0
